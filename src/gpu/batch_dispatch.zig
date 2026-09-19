const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const model_types = @import("../model/types.zig");
pub const model_gpu = @import("model_gpu.zig");
pub const model_dispatch = @import("model_dispatch.zig");
pub const batch_prefill = @import("batch_prefill.zig");
pub const tensor = @import("../tensor.zig");

fn recordBarrier(gpu: *model_gpu.GpuModelContext, cmd: types.VkCommandBuffer) void {
    gpu.engine.recordBarrier(cmd);
}

fn expandEmbeddings(dst: []f32, tokens: []const u32, embed: []const tensor.bf16, H: usize, scale: f32) void {
    for (tokens, 0..) |tok, i| {
        const src = embed[@as(usize, tok) * H ..][0..H];
        const out = dst[i * H ..][0..H];
        for (out, src) |*d, s| {
            const u: u32 = @as(u32, s.bits) << 16;
            d.* = @as(f32, @bitCast(u)) * scale;
        }
    }
}

pub const ProgressFn = *const fn (layer_idx: usize, total_layers: usize, ctx: ?*anyopaque) void;

pub fn gpuDispatchPrefillBatch(
    prefill: *batch_prefill.BatchPrefillContext,
    gpu: *model_gpu.GpuModelContext,
    config: *const model_types.ModelConfig,
    layers: []const model_types.LayerWeights,
    tokens: []const u32,
    embed_tokens: []const tensor.bf16,
    slots: []const u32,
    start_clock: usize,
    num_prev_slots: usize,
    logits_out: []f32,
    progress_fn: ?ProgressFn,
    progress_ctx: ?*anyopaque,
) !void {
    const N: u32 = @intCast(tokens.len);
    if (N == 0 or N > prefill.max_tokens) return error.BatchTooLarge;
    const H = config.hidden_size;
    const inter = config.intermediate_size;
    const eps = config.rms_norm_eps;
    const embed_scale = @sqrt(@as(f32, @floatFromInt(H)));

    expandEmbeddings(prefill.buf_x.asSlice(f32)[0 .. N * H], tokens, embed_tokens, H, embed_scale);
    @memcpy(prefill.buf_slots.asSlice(u32)[0..slots.len], slots);

    const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
    _ = prefill.ctx.api.vkBeginCommandBuffer(prefill.cmd_buf, &begin_info);
    recordBarrier(gpu, prefill.cmd_buf);

    for (gpu.layers, 0..) |l_gpu, i| {
        const l_cpu = layers[i];
        const d = prefill.layers[i];
        const q_dim: u32 = @intCast(l_cpu.q_dim);
        const kv_dim: u32 = @intCast(l_cpu.kv_dim);
        const head_dim: u32 = @intCast(l_cpu.head_dim);
        const rot_dim: u32 = @intCast(l_cpu.rotary_dim);
        const theta: f32 = if (l_cpu.layer_type == .full_attention) config.rope_theta_full else config.rope_theta;

        if (i == 0) {
            const pc_rms = extern struct { H: u32, eps: f32, N: u32, pad: u32 }{ .H = @intCast(H), .eps = eps, .N = N, .pad = 0 };
            prefill.pipe_rmsnorm.record(prefill.cmd_buf, d.input_norm, std.mem.asBytes(&pc_rms), N, 1, 1);
            recordBarrier(gpu, prefill.cmd_buf);
        }

        const n_tiles: u32 = if (prefill.mlp_is_q4) (N + 63) / 64 else (N + 7) / 8;
        const q_tiles: u32 = if (prefill.mlp_is_q4) (q_dim + 31) / 32 else (q_dim + 7) / 8;
        const kv_tiles: u32 = if (prefill.mlp_is_q4) (kv_dim + 31) / 32 else (kv_dim + 7) / 8;
        const h_tiles: u32 = if (prefill.mlp_is_q4) @intCast((H + 31) / 32) else @intCast((H + 7) / 8);
        const inter_tiles: u32 = if (prefill.mlp_is_q4) @intCast((inter + 31) / 32) else @intCast((inter + 7) / 8);

        const pc_q = [4]u32{ N, q_dim, @intCast(H), 0 };
        const pc_kv = [4]u32{ N, kv_dim, @intCast(H), 0 };
        const gemm_pipe = if (prefill.mlp_is_q4) &prefill.pipe_gemm_q4 else &prefill.pipe_gemm_q8;
        gemm_pipe.record(prefill.cmd_buf, d.q_proj, std.mem.sliceAsBytes(&pc_q), q_tiles, n_tiles, 1);
        gemm_pipe.record(prefill.cmd_buf, d.k_proj, std.mem.sliceAsBytes(&pc_kv), kv_tiles, n_tiles, 1);
        gemm_pipe.record(prefill.cmd_buf, d.v_proj, std.mem.sliceAsBytes(&pc_kv), kv_tiles, n_tiles, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        const pc_rope = extern struct {
            num_q_heads: u32, num_kv_heads: u32, head_dim: u32, rotary_dim: u32,
            k_eq_v: u32, rope_theta: f32, eps: f32, N: u32,
            start_clock: u32, slot_offset: u32, pad0: u32, pad1: u32,
        }{
            .num_q_heads = @intCast(config.num_attention_heads), .num_kv_heads = @intCast(l_cpu.num_kv_heads),
            .head_dim = head_dim, .rotary_dim = rot_dim, .k_eq_v = if (l_cpu.k_eq_v) 1 else 0,
            .rope_theta = theta, .eps = eps, .N = N,
            .start_clock = @intCast(start_clock), .slot_offset = @intCast(num_prev_slots), .pad0 = 0, .pad1 = 0,
        };
        prefill.pipe_qkv_rope.record(prefill.cmd_buf, d.qkv_rope, std.mem.asBytes(&pc_rope), @intCast(config.num_attention_heads + l_cpu.num_kv_heads), N, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        const pc_attn = extern struct {
            head_dim: u32, kv_dim: u32, gqa_ratio: u32, inv_sqrt_dim: f32,
            num_q_heads: u32, N: u32, num_prev_slots: u32, is_sliding: u32,
        }{
            .head_dim = head_dim, .kv_dim = kv_dim, .gqa_ratio = @intCast(config.num_attention_heads / l_cpu.num_kv_heads),
            .inv_sqrt_dim = 1.0, .num_q_heads = @intCast(config.num_attention_heads), .N = N,
            .num_prev_slots = @intCast(num_prev_slots), .is_sliding = if (l_cpu.layer_type == .sliding_attention) 1 else 0,
        };
        prefill.pipe_causal_attn.record(prefill.cmd_buf, d.causal_attn, std.mem.asBytes(&pc_attn), @intCast(config.num_attention_heads), N, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        const pc_o = [4]u32{ N, @intCast(H), q_dim, 0 };
        gemm_pipe.record(prefill.cmd_buf, d.o_proj, std.mem.sliceAsBytes(&pc_o), h_tiles, n_tiles, 1);
        if (l_gpu.has_post_attn_norm) {
            recordBarrier(gpu, prefill.cmd_buf);
            const pc_pan = extern struct { H: u32, eps: f32, N: u32, pad: u32 }{ .H = @intCast(H), .eps = eps, .N = N, .pad = 0 };
            prefill.pipe_rmsnorm.record(prefill.cmd_buf, d.post_attn_norm, std.mem.asBytes(&pc_pan), N, 1, 1);
        }
        recordBarrier(gpu, prefill.cmd_buf);

        const pc_pfn = extern struct { H: u32, eps: f32, scalar: f32, N: u32 }{ .H = @intCast(H), .eps = eps, .scalar = 1.0, .N = N };
        prefill.pipe_add_norm.record(prefill.cmd_buf, d.pre_ffn_norm, std.mem.asBytes(&pc_pfn), N, 1, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        const pc_mlp = [4]u32{ N, @intCast(inter), @intCast(H), 0 };
        if (prefill.mlp_is_q4) {
            prefill.pipe_fused_mlp_q4.record(prefill.cmd_buf, d.gate_up_proj, std.mem.sliceAsBytes(&pc_mlp), inter_tiles, n_tiles, 1);
        } else {
            prefill.pipe_fused_mlp_q8.record(prefill.cmd_buf, d.gate_up_proj, std.mem.sliceAsBytes(&pc_mlp), @intCast(inter), N, 1);
        }
        recordBarrier(gpu, prefill.cmd_buf);

        const pc_down = [4]u32{ N, @intCast(H), @intCast(inter), 0 };
        gemm_pipe.record(prefill.cmd_buf, d.down_proj, std.mem.sliceAsBytes(&pc_down), h_tiles, n_tiles, 1);
        if (l_gpu.has_post_ffn_norm) {
            recordBarrier(gpu, prefill.cmd_buf);
            const pc_pfn_norm = extern struct { H: u32, eps: f32, N: u32, pad: u32 }{ .H = @intCast(H), .eps = eps, .N = N, .pad = 0 };
            prefill.pipe_rmsnorm.record(prefill.cmd_buf, d.post_ffn_norm, std.mem.asBytes(&pc_pfn_norm), N, 1, 1);
        }
        recordBarrier(gpu, prefill.cmd_buf);

        const pc_padd = extern struct { H: u32, eps: f32, scalar: f32, N: u32 }{ .H = @intCast(H), .eps = eps, .scalar = l_gpu.layer_scalar, .N = N };
        prefill.pipe_add_norm.record(prefill.cmd_buf, d.post_ffn_add, std.mem.asBytes(&pc_padd), N, 1, 1);
        recordBarrier(gpu, prefill.cmd_buf);

        if ((i + 1) % 4 == 0 or i + 1 == gpu.layers.len) {
            if (i + 1 == gpu.layers.len and logits_out.len > 0) {
                const last_tok_off = (N - 1) * H * 4;
                const copy_region = types_dispatch.VkBufferCopy{ .srcOffset = last_tok_off, .dstOffset = 0, .size = H * 4 };
                prefill.ctx.api.vkCmdCopyBuffer(prefill.cmd_buf, prefill.buf_normed_x.buffer, gpu.buf_normed_x.buffer, 1, (&copy_region)[0..1].ptr);
                recordBarrier(gpu, prefill.cmd_buf);
                gpu.engine.recordGemvLogits(prefill.cmd_buf, gpu.desc_logits, config.vocab_size, H, 0);
                recordBarrier(gpu, prefill.cmd_buf);
                gpu.engine.recordTopK(prefill.cmd_buf, gpu.desc_topk_pass1, gpu.desc_topk_pass2, config.vocab_size);
            }
            _ = prefill.ctx.api.vkEndCommandBuffer(prefill.cmd_buf);
            const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&prefill.cmd_buf)[0..1].ptr };
            const sub_res = prefill.ctx.api.vkQueueSubmit(prefill.ctx.queue, 1, (&submit_info)[0..1].ptr, prefill.fence);
            const wait_res = prefill.ctx.api.vkWaitForFences(prefill.ctx.device, 1, (&prefill.fence)[0..1].ptr, 1, std.math.maxInt(u64));
            _ = prefill.ctx.api.vkResetFences(prefill.ctx.device, 1, (&prefill.fence)[0..1].ptr);
            if (sub_res != .SUCCESS or wait_res != .SUCCESS) {
                std.debug.print("Layer {} failed: sub={}, wait={}\n", .{ i, sub_res, wait_res });
                return error.GpuError;
            }
            if (progress_fn) |pfn| pfn(i + 1, gpu.layers.len, progress_ctx);
            if (i + 1 < gpu.layers.len) {
                _ = prefill.ctx.api.vkBeginCommandBuffer(prefill.cmd_buf, &begin_info);
                recordBarrier(gpu, prefill.cmd_buf);
            }
        }
    }

    if (logits_out.len > 0) {
        @memcpy(logits_out, gpu.buf_logits.asSlice(f32)[0..logits_out.len]);
    }
}
