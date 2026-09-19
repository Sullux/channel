const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const kernels = @import("kernels.zig");
pub const descriptors = @import("descriptors.zig");
pub const tensor = @import("../tensor.zig");
pub const model = @import("../model.zig");
pub const model_types = @import("../model/types.zig");
pub const quant = @import("../quant.zig");

pub const GpuLayerWeights = struct {
    input_norm: buffer.GpuBuffer, q_norm: buffer.GpuBuffer, k_norm: buffer.GpuBuffer,
    q_proj: buffer.GpuBuffer, k_proj: buffer.GpuBuffer, v_proj: buffer.GpuBuffer, o_proj: buffer.GpuBuffer,
    gate_proj: buffer.GpuBuffer, up_proj: buffer.GpuBuffer, down_proj: buffer.GpuBuffer,
    pre_ffn_norm: buffer.GpuBuffer, post_attn_norm: buffer.GpuBuffer, post_ffn_norm: buffer.GpuBuffer,
    buf_x_prev: buffer.GpuBuffer, buf_k_cache: buffer.GpuBuffer, buf_v_cache: buffer.GpuBuffer,
    layer_scalar: f32, has_post_attn_norm: bool, has_post_ffn_norm: bool, desc: descriptors.LayerDescriptorSets,

    pub fn deinit(self: *GpuLayerWeights) void {
        self.buf_v_cache.deinit(); self.buf_k_cache.deinit(); self.buf_x_prev.deinit();
        self.post_ffn_norm.deinit(); self.post_attn_norm.deinit(); self.pre_ffn_norm.deinit();
        self.down_proj.deinit(); self.up_proj.deinit(); self.gate_proj.deinit(); self.o_proj.deinit();
        self.v_proj.deinit(); self.k_proj.deinit(); self.q_proj.deinit();
        self.k_norm.deinit(); self.q_norm.deinit(); self.input_norm.deinit();
    }
};

pub const GpuModelContext = struct {
    allocator: std.mem.Allocator, ctx: *const context.GpuContext, engine: kernels.GpuEngine,
    desc_mgr: descriptors.DescriptorManager, layers: []GpuLayerWeights,
    embed_tokens: buffer.GpuBuffer, final_norm: buffer.GpuBuffer,
    desc_logits: types.VkDescriptorSet, desc_final_norm: types.VkDescriptorSet, desc_argmax: types.VkDescriptorSet,
    desc_topk_pass1: types.VkDescriptorSet, desc_topk_pass2: types.VkDescriptorSet,
    buf_x: buffer.GpuBuffer, buf_normed_x: buffer.GpuBuffer, buf_q: buffer.GpuBuffer, buf_k: buffer.GpuBuffer, buf_v: buffer.GpuBuffer,
    buf_attn_out: buffer.GpuBuffer, buf_active_slots: buffer.GpuBuffer, buf_gate: buffer.GpuBuffer, buf_up: buffer.GpuBuffer,
    buf_act: buffer.GpuBuffer, buf_mlp_out: buffer.GpuBuffer, buf_logits: buffer.GpuBuffer, buf_sampled_token: buffer.GpuBuffer,
    buf_topk_inter_ids: buffer.GpuBuffer, buf_topk_inter_vals: buffer.GpuBuffer, buf_topk_ids: buffer.GpuBuffer, buf_topk_vals: buffer.GpuBuffer,
    buf_step_params: buffer.GpuBuffer, buf_indirect_cmds: buffer.GpuBuffer,
    cmd_buf_decode: types.VkCommandBuffer, cmd_buf_prefill: types.VkCommandBuffer,
    batch_prefill_ctx: ?*@import("batch_prefill.zig").BatchPrefillContext = null,

    fn createWeightBuffer(ctx: *const context.GpuContext, src: []const tensor.bf16, rows: usize, cols: usize, mode: quant.QuantMode) !buffer.GpuBuffer {
        if (mode == .none or src.len == 0) {
            const byte_size = @as(u64, @intCast(src.len * @sizeOf(tensor.bf16)));
            var buf = try buffer.GpuBuffer.init(ctx, byte_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            if (src.len > 0) @memcpy(buf.asSlice(u8), std.mem.sliceAsBytes(src));
            return buf;
        }
        const byte_size = @as(u64, @intCast(quant.getQuantizedSizeBytes(rows, cols, mode)));
        var buf = try buffer.GpuBuffer.init(ctx, byte_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        quant.quantizeMatrix(buf.asSlice(u32), @as([]const u16, @ptrCast(src)), rows, cols, mode);
        return buf;
    }

    fn createNormBuffer(ctx: *const context.GpuContext, src: []const tensor.bf16, H: usize) !buffer.GpuBuffer {
        var buf = try buffer.GpuBuffer.init(ctx, H * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const slice = buf.asSlice(f32);
        if (src.len == H) {
            for (slice, src) |*dst, s| dst.* = s.toF32();
        } else {
            @memset(slice, 1.0);
        }
        return buf;
    }

    pub fn init(allocator: std.mem.Allocator, ctx: *const context.GpuContext, m: *const model.Model, config: model_types.ModelConfig, mode: quant.QuantMode, quiescence_thresh: f32) !GpuModelContext {
        var engine = try kernels.GpuEngine.init(ctx, mode);
        errdefer engine.deinit();

        const max_sets: u32 = @intCast(m.layers.len * 20 + 32);
        var desc_mgr = try descriptors.DescriptorManager.init(ctx, max_sets);
        errdefer desc_mgr.deinit();

        const H = config.hidden_size;
        const I = config.intermediate_size;
        const V = config.vocab_size;
        const max_slots = 4096;
        const max_head = @max(config.head_dim, config.global_head_dim);
        const max_q = config.num_attention_heads * max_head;
        const max_kv = @max(config.num_key_value_heads, config.num_global_key_value_heads) * max_head;
        const sb = types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | types.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | types.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const ind_sb = sb | types.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

        const buf_x = try buffer.GpuBuffer.init(ctx, H * 4, sb);
        const buf_normed_x = try buffer.GpuBuffer.init(ctx, H * 4, sb);
        const buf_q = try buffer.GpuBuffer.init(ctx, max_q * 4, sb);
        const buf_k = try buffer.GpuBuffer.init(ctx, max_kv * 4, sb);
        const buf_v = try buffer.GpuBuffer.init(ctx, max_kv * 4, sb);
        const buf_attn_out = try buffer.GpuBuffer.init(ctx, max_q * 4, sb);
        const buf_active_slots = try buffer.GpuBuffer.init(ctx, @max(8192, max_slots * 2) * 4, sb);
        const buf_gate = try buffer.GpuBuffer.init(ctx, I * 4, sb);
        const buf_up = try buffer.GpuBuffer.init(ctx, I * 4, sb);
        const buf_act = try buffer.GpuBuffer.init(ctx, I * 4, sb);
        const buf_mlp_out = try buffer.GpuBuffer.init(ctx, H * 4, sb);
        const buf_topk_inter_ids = try buffer.GpuBuffer.init(ctx, 4096 * 4, sb);
        const buf_topk_inter_vals = try buffer.GpuBuffer.init(ctx, 4096 * 4, sb);
        const buf_topk_ids = try buffer.GpuBuffer.init(ctx, 64 * 4, sb);
        const buf_topk_vals = try buffer.GpuBuffer.init(ctx, 64 * 4, sb);
        const buf_logits = try buffer.GpuBuffer.initCached(ctx, V * 4, sb);
        const buf_sampled_token = try buffer.GpuBuffer.init(ctx, 4, sb);
        const buf_step_params = try buffer.GpuBuffer.init(ctx, 64, sb);
        const buf_indirect_cmds = try buffer.GpuBuffer.init(ctx, m.layers.len * 16 * 12, ind_sb);
        const cb_info = types_dispatch.VkCommandBufferAllocateInfo{ .commandPool = engine.cmd_pool, .commandBufferCount = 2 };
        var cmds: [2]types.VkCommandBuffer = .{ null, null };
        if (ctx.api.vkAllocateCommandBuffers(ctx.device, &cb_info, &cmds) != .SUCCESS) return error.VkCmdBufferAllocFailed;

        const attn_m: quant.QuantMode = if (mode == .mixed) .q8 else mode;
        const mlp_m: quant.QuantMode = mode;
        const down_m: quant.QuantMode = if (mode == .mixed) .q8 else mode;

        var gpu_layers = try allocator.alloc(GpuLayerWeights, m.layers.len);
        for (m.layers, 0..) |l, i| {
            const q_dim, const kv_dim = .{ l.q_proj.len / H, l.k_proj.len / H };
            const d = try desc_mgr.allocateLayerSets(&engine);
            var w = GpuLayerWeights{
                .input_norm = try createNormBuffer(ctx, l.input_layernorm, H),
                .q_norm = try createNormBuffer(ctx, l.q_norm, l.head_dim),
                .k_norm = try createNormBuffer(ctx, l.k_norm, l.head_dim),
                .q_proj = try createWeightBuffer(ctx, l.q_proj, q_dim, H, attn_m),
                .k_proj = try createWeightBuffer(ctx, l.k_proj, kv_dim, H, attn_m),
                .v_proj = if (l.v_proj.len > 0) try createWeightBuffer(ctx, l.v_proj, kv_dim, H, attn_m) else try createWeightBuffer(ctx, l.k_proj, kv_dim, H, attn_m),
                .o_proj = try createWeightBuffer(ctx, l.o_proj, H, q_dim, attn_m),
                .gate_proj = try createWeightBuffer(ctx, l.gate_proj, I, H, mlp_m),
                .up_proj = try createWeightBuffer(ctx, l.up_proj, I, H, mlp_m),
                .down_proj = try createWeightBuffer(ctx, l.down_proj, H, I, down_m),
                .pre_ffn_norm = try createNormBuffer(ctx, l.pre_feedforward_layernorm, H),
                .post_attn_norm = try createNormBuffer(ctx, if (l.post_attention_layernorm) |p| p else &.{}, H),
                .post_ffn_norm = try createNormBuffer(ctx, if (l.post_feedforward_layernorm) |p| p else &.{}, H),
                .buf_x_prev = try buffer.GpuBuffer.init(ctx, H * 4, sb),
                .buf_k_cache = try buffer.GpuBuffer.init(ctx, max_slots * max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                .buf_v_cache = try buffer.GpuBuffer.init(ctx, max_slots * max_kv * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                .layer_scalar = l.layer_scalar orelse 1.0, .has_post_attn_norm = (l.post_attention_layernorm != null),
                .has_post_ffn_norm = (l.post_feedforward_layernorm != null), .desc = d,
            };
            desc_mgr.bindBuffers(w.desc.input_norm, &.{ &buf_x, &w.input_norm, &buf_normed_x });
            desc_mgr.bindBuffers(w.desc.fused_qkv, &.{ &w.q_proj, &w.k_proj, &w.v_proj, &buf_normed_x, &buf_q, &buf_k, &buf_v });
            desc_mgr.bindBuffers(w.desc.q_proj, &.{ &w.q_proj, &buf_normed_x, &buf_q });
            desc_mgr.bindBuffers(w.desc.k_proj, &.{ &w.k_proj, &buf_normed_x, &buf_k });
            desc_mgr.bindBuffers(w.desc.v_proj, &.{ &w.v_proj, &buf_normed_x, &buf_v });
            desc_mgr.bindBuffers(w.desc.qkv_rope, &.{ &buf_q, &buf_k, &buf_v, &w.q_norm, &w.k_norm, &buf_q, &w.buf_k_cache, &w.buf_v_cache, &buf_step_params });
            desc_mgr.bindBuffers(w.desc.attn, &.{ &buf_q, &w.buf_k_cache, &w.buf_v_cache, &buf_active_slots, &buf_attn_out, &buf_step_params });
            desc_mgr.bindBuffers(w.desc.o_proj, &.{ &w.o_proj, &buf_attn_out, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.gate_proj, &.{ &w.gate_proj, &buf_normed_x, &buf_gate });
            desc_mgr.bindBuffers(w.desc.up_proj, &.{ &w.up_proj, &buf_normed_x, &buf_up });
            desc_mgr.bindBuffers(w.desc.swiglu, &.{ &buf_gate, &buf_up, &buf_act });
            desc_mgr.bindBuffers(w.desc.down_proj, &.{ &w.down_proj, &buf_act, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.gate_up_swiglu, &.{ &w.gate_proj, &w.up_proj, &buf_normed_x, &buf_act });
            desc_mgr.bindBuffers(w.desc.pre_ffn_norm, &.{ &buf_x, &buf_mlp_out, &w.pre_ffn_norm, &buf_normed_x });
            desc_mgr.bindBuffers(w.desc.post_attn_norm, &.{ &buf_mlp_out, &w.post_attn_norm, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.post_ffn_norm, &.{ &buf_mlp_out, &w.post_ffn_norm, &buf_mlp_out });
            desc_mgr.bindBuffers(w.desc.quiescence_gate, &.{ &buf_x, &w.buf_x_prev, &buf_indirect_cmds });
            gpu_layers[i] = w;
        }

        const embed_tokens = try createWeightBuffer(ctx, m.embed_tokens, V, H, .q8);
        const final_norm = try createNormBuffer(ctx, m.final_norm, H);

        for (gpu_layers, 0..) |*w, i| {
            const next_norm = if (i + 1 < gpu_layers.len) &gpu_layers[i + 1].input_norm else &final_norm;
            desc_mgr.bindBuffers(w.desc.post_ffn_add, &.{ &buf_x, &buf_mlp_out, next_norm, &buf_normed_x });
        }

        const desc_logits = try desc_mgr.allocateSet(engine.gemv_logits_pipe.desc_set_layout);
        const desc_final_norm = try desc_mgr.allocateSet(engine.rmsnorm_pipe.desc_set_layout);
        const desc_argmax = try desc_mgr.allocateSet(engine.argmax_pipe.desc_set_layout);
        const desc_topk_pass1 = try desc_mgr.allocateSet(engine.topk_pass1_pipe.desc_set_layout);
        const desc_topk_pass2 = try desc_mgr.allocateSet(engine.topk_pass2_pipe.desc_set_layout);
        desc_mgr.bindBuffers(desc_final_norm, &.{ &buf_x, &final_norm, &buf_normed_x });
        desc_mgr.bindBuffers(desc_logits, &.{ &embed_tokens, &buf_normed_x, &buf_logits });
        desc_mgr.bindBuffers(desc_argmax, &.{ &buf_logits, &buf_sampled_token });
        desc_mgr.bindBuffers(desc_topk_pass1, &.{ &buf_logits, &buf_topk_inter_ids, &buf_topk_inter_vals });
        desc_mgr.bindBuffers(desc_topk_pass2, &.{ &buf_topk_inter_ids, &buf_topk_inter_vals, &buf_topk_ids, &buf_topk_vals });

        var bp_ptr: ?*@import("batch_prefill.zig").BatchPrefillContext = null;
        if (allocator.create(@import("batch_prefill.zig").BatchPrefillContext)) |bp| {
            if (@import("batch_prefill.zig").BatchPrefillContext.init(allocator, ctx, &config, gpu_layers, &embed_tokens, &final_norm, &buf_logits, 1024, mode)) |res| {
                bp.* = res; bp_ptr = bp;
            } else |_| allocator.destroy(bp);
        } else |_| {}

        var self_ctx = GpuModelContext{
            .allocator = allocator, .ctx = ctx, .engine = engine, .desc_mgr = desc_mgr,
            .layers = gpu_layers, .embed_tokens = embed_tokens, .final_norm = final_norm,
            .desc_logits = desc_logits, .desc_final_norm = desc_final_norm, .desc_argmax = desc_argmax,
            .desc_topk_pass1 = desc_topk_pass1, .desc_topk_pass2 = desc_topk_pass2,
            .buf_x = buf_x, .buf_normed_x = buf_normed_x, .buf_q = buf_q, .buf_k = buf_k, .buf_v = buf_v,
            .buf_attn_out = buf_attn_out, .buf_active_slots = buf_active_slots,
            .buf_gate = buf_gate, .buf_up = buf_up, .buf_act = buf_act, .buf_mlp_out = buf_mlp_out,
            .buf_logits = buf_logits, .buf_sampled_token = buf_sampled_token,
            .buf_topk_inter_ids = buf_topk_inter_ids, .buf_topk_inter_vals = buf_topk_inter_vals,
            .buf_topk_ids = buf_topk_ids, .buf_topk_vals = buf_topk_vals,
            .buf_step_params = buf_step_params, .buf_indirect_cmds = buf_indirect_cmds,
            .cmd_buf_decode = cmds[0], .cmd_buf_prefill = cmds[1], .batch_prefill_ctx = bp_ptr,
        };
        const model_dispatch = @import("model_dispatch.zig");
        model_dispatch.recordForwardGraph(&self_ctx, &m.config, m.layers, cmds[0], true, quiescence_thresh);
        model_dispatch.recordForwardGraph(&self_ctx, &m.config, m.layers, cmds[1], false, 0.0);
        return self_ctx;
    }

    pub fn deinit(self: *GpuModelContext) void {
        if (self.batch_prefill_ctx) |bp| { bp.deinit(); self.allocator.destroy(bp); }
        self.engine.deinit(); self.desc_mgr.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.buf_indirect_cmds.deinit(); self.final_norm.deinit(); self.embed_tokens.deinit();
        self.buf_step_params.deinit(); self.buf_sampled_token.deinit(); self.buf_logits.deinit();
        self.buf_topk_vals.deinit(); self.buf_topk_ids.deinit();
        self.buf_topk_inter_vals.deinit(); self.buf_topk_inter_ids.deinit();
        self.buf_mlp_out.deinit(); self.buf_act.deinit(); self.buf_up.deinit(); self.buf_gate.deinit();
        self.buf_active_slots.deinit(); self.buf_attn_out.deinit(); self.buf_v.deinit();
        self.buf_k.deinit(); self.buf_q.deinit(); self.buf_normed_x.deinit(); self.buf_x.deinit();
    }
};
