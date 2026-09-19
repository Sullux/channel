const std = @import("std");
pub const tensor = @import("../tensor.zig");
pub const bf16 = tensor.bf16;
pub const model = @import("../model.zig");
pub const Model = model.Model;
pub const LayerWeights = model.LayerWeights;
pub const ForwardScratch = model.ForwardScratch;
pub const kernels = @import("../kernels.zig");
pub const memory = @import("../memory.zig");
pub const memory_inject = @import("memory_inject.zig");
pub const ring_buffer = @import("../ring_buffer.zig");
pub const DynamicRingBuffer = ring_buffer.DynamicRingBuffer;
pub const quiescence = @import("../quiescence.zig");
pub const QuiescenceTracker = quiescence.QuiescenceTracker;
pub const gpu = @import("../gpu.zig");
pub const GpuModelContext = gpu.model_gpu.GpuModelContext;
pub const ple = @import("ple.zig");

pub fn forwardToken(
    self: *const Model,
    ring: *DynamicRingBuffer,
    scratch: *ForwardScratch,
    token_id: u32,
    clock: usize,
    tp: ?*std.Thread.Pool,
    memory_opt: ?*memory.DiffArchive,
    quiescence_opt: ?*quiescence.QuiescenceTracker,
    gpu_opt: ?*GpuModelContext,
    compute_logits: bool,
) u32 {
    const H = self.config.hidden_size;
    const ple_dim = self.config.hidden_size_per_layer_input;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));

    @memcpy(scratch.prev_x, scratch.x);
    const emb_offset = @as(usize, token_id) * H;
    for (scratch.x, self.embed_tokens[emb_offset .. emb_offset + H]) |*out, e| out.* = e.toF32() * embed_scale;

    var result_token: u32 = 0;

    if (gpu_opt) |g| {
        _ = ring.activateSlot(0, clock);
        const slot_idx = ring.getSlotIndex(clock);
        const active_count = ring.getActiveSlotsLayer(0, clock, false, self.config.sliding_window, scratch.active_slots);
        const sliding_count = ring.getActiveSlotsLayer(0, clock, true, self.config.sliding_window, scratch.sliding_slots);
        const topk_out: ?[]model.TopKCandidate = if (compute_logits) &scratch.topk_candidates else null;
        result_token = gpu.model_dispatch.gpuDispatchForwardToken(g, &self.config, self.layers, scratch.x, topk_out, clock, slot_idx, scratch.active_slots[0..active_count], scratch.sliding_slots[0..sliding_count]);
    } else {
        if (ple_dim > 0) ple.preparePLE(self, scratch, token_id, H, ple_dim, tp);

        for (self.layers, 0..) |l, layer_idx| {
            if (quiescence_opt) |q| {
                if (!q.shouldExecute(layer_idx, clock, scratch.x, scratch.prev_x)) continue;
            }
            @memcpy(scratch.prev_x, scratch.x);
            forwardAttentionCpu(self, l, ring, scratch, layer_idx, clock, H, tp);
            forwardMlpCpu(self, l, scratch, H, tp);
            if (ple_dim > 0) ple.forwardPLE(self, l, scratch, layer_idx, ple_dim, H);
            if (l.layer_scalar) |s| {
                for (scratch.x) |*x_val| x_val.* *= s;
            }
        }

        if (compute_logits) {
            kernels.rmsNorm(scratch.normed_x, scratch.x, self.final_norm, self.config.rms_norm_eps);
            if (tp) |pool| {
                kernels.gemvParallel(scratch.logits, scratch.normed_x, self.embed_tokens, self.config.vocab_size, H, pool);
            } else {
                kernels.gemv(scratch.logits, scratch.normed_x, self.embed_tokens, self.config.vocab_size, H);
            }
            result_token = kernels.sampleArgmax(scratch.logits);
        }
    }

    if (memory_opt) |mem| {
        memory_inject.integrateMemory(self, mem, ring, scratch, clock, H, gpu_opt);
    }
    return result_token;
}

fn runAttentionHeads(self: *const Model, l: LayerWeights, ring: *DynamicRingBuffer, scratch: *ForwardScratch, kv_layer: usize, clock: usize) void {
    const is_sliding = (l.layer_type == .sliding_attention);
    const active_count = ring.getActiveSlotsLayer(kv_layer, clock, is_sliding, self.config.sliding_window, scratch.active_slots);
    const gqa_group_size = self.config.num_attention_heads / l.num_kv_heads;
    const inv_sqrt_dim: f32 = 1.0;

    for (0..self.config.num_attention_heads) |h| {
        const kv_h = h / gqa_group_size;
        const q_head = scratch.q[h * l.head_dim .. (h + 1) * l.head_dim];
        const head_scores = scratch.attn_scores[0..active_count];

        for (0..active_count) |i| {
            const slot = scratch.active_slots[i];
            const slot_kv = ring.getSlotKV(kv_layer, slot, l.kv_dim);
            head_scores[i] = kernels.dotF32(q_head, slot_kv.k[kv_h * l.head_dim .. (kv_h + 1) * l.head_dim]) * inv_sqrt_dim;
        }
        kernels.softmax(head_scores);
        const out_head = scratch.attn_out[h * l.head_dim .. (h + 1) * l.head_dim];
        @memset(out_head, 0);

        for (0..active_count) |i| {
            const slot = scratch.active_slots[i];
            const slot_kv = ring.getSlotKV(kv_layer, slot, l.kv_dim);
            const v_head = slot_kv.v[kv_h * l.head_dim .. (kv_h + 1) * l.head_dim];
            const weight = head_scores[i];
            ring.recordAttention(kv_layer, slot, weight);
            for (out_head, v_head) |*o, v_val| o.* += weight * v_val;
        }
    }
}

pub fn forwardAttentionCpu(self: *const Model, l: LayerWeights, ring: *DynamicRingBuffer, scratch: *ForwardScratch, layer_idx: usize, clock: usize, H: usize, tp: ?*std.Thread.Pool) void {
    kernels.rmsNorm(scratch.normed_x, scratch.x, l.input_layernorm, self.config.rms_norm_eps);
    const first_kv_shared = self.config.num_hidden_layers - self.config.num_kv_shared_layers;
    const is_shared = (self.config.num_kv_shared_layers > 0 and layer_idx >= first_kv_shared);
    const kv_layer = if (is_shared) (if (l.layer_type == .full_attention) @as(usize, 14) else @as(usize, 13)) else layer_idx;

    if (tp) |pool| {
        kernels.gemvParallel(scratch.q[0..l.q_dim], scratch.normed_x, l.q_proj, l.q_dim, H, pool);
        if (!is_shared) {
            kernels.gemvParallel(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H, pool);
            if (!l.k_eq_v and l.v_proj.len > 0) kernels.gemvParallel(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H, pool);
        }
    } else {
        kernels.gemv(scratch.q[0..l.q_dim], scratch.normed_x, l.q_proj, l.q_dim, H);
        if (!is_shared) {
            kernels.gemv(scratch.k[0..l.kv_dim], scratch.normed_x, l.k_proj, l.kv_dim, H);
            if (!l.k_eq_v and l.v_proj.len > 0) kernels.gemv(scratch.v[0..l.kv_dim], scratch.normed_x, l.v_proj, l.kv_dim, H);
        }
    }
    if (l.k_eq_v or l.v_proj.len == 0) @memcpy(scratch.v[0..l.kv_dim], scratch.k[0..l.kv_dim]);

    for (0..self.config.num_attention_heads) |h| {
        const head_q = scratch.q[h * l.head_dim .. (h + 1) * l.head_dim];
        kernels.rmsNorm(head_q, head_q, l.q_norm, self.config.rms_norm_eps);
    }
    const theta = if (l.layer_type == .full_attention) self.config.rope_theta_full else self.config.rope_theta;
    kernels.applyRopePartial(scratch.q[0..l.q_dim], clock, l.head_dim, l.rotary_dim, theta);

    if (!is_shared) {
        for (0..l.num_kv_heads) |kv_h| {
            const head_k = scratch.k[kv_h * l.head_dim .. (kv_h + 1) * l.head_dim];
            kernels.rmsNorm(head_k, head_k, l.k_norm, self.config.rms_norm_eps);
            const head_v = scratch.v[kv_h * l.head_dim .. (kv_h + 1) * l.head_dim];
            kernels.unitRmsNorm(head_v, head_v, self.config.rms_norm_eps);
        }
        kernels.applyRopePartial(scratch.k[0..l.kv_dim], clock, l.head_dim, l.rotary_dim, theta);
        ring.writeKV(kv_layer, clock, scratch.k[0..l.kv_dim], scratch.v[0..l.kv_dim]);
    }

    runAttentionHeads(self, l, ring, scratch, kv_layer, clock);

    if (tp) |pool| {
        kernels.gemvParallel(scratch.mlp_out, scratch.attn_out[0..l.q_dim], l.o_proj, H, l.q_dim, pool);
    } else {
        kernels.gemv(scratch.mlp_out, scratch.attn_out[0..l.q_dim], l.o_proj, H, l.q_dim);
    }
    if (l.post_attention_layernorm) |pal| kernels.rmsNorm(scratch.mlp_out, scratch.mlp_out, pal, self.config.rms_norm_eps);
    for (scratch.x, scratch.mlp_out) |*x_val, attn_v| x_val.* += attn_v;
}

pub fn forwardMlpCpu(self: *const Model, l: LayerWeights, scratch: *ForwardScratch, H: usize, tp: ?*std.Thread.Pool) void {
    kernels.rmsNorm(scratch.normed_x, scratch.x, l.pre_feedforward_layernorm, self.config.rms_norm_eps);
    kernels.gatedMlp(scratch.mlp_out, scratch.normed_x, l.gate_proj, l.up_proj, l.down_proj, H, l.intermediate_dim, scratch.mlp_gate_up, tp);
    if (l.post_feedforward_layernorm) |pfl| kernels.rmsNorm(scratch.mlp_out, scratch.mlp_out, pfl, self.config.rms_norm_eps);
    for (scratch.x, scratch.mlp_out) |*x_val, ffn_v| x_val.* += ffn_v;
}
