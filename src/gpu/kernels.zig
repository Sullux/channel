const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const pipeline = @import("pipeline.zig");
pub const shaders = @import("shaders.zig");
pub const quant = @import("../quant.zig");

pub const QuiescenceGatePushConstants = extern struct {
    hidden_size: u32,
    threshold_sq: f32,
    base_cmd_idx: u32,
    num_cmds: u32,
    targets: [16]u32,
};

pub const GpuEngine = struct {
    ctx: *const context.GpuContext,
    mode: quant.QuantMode,
    gemv_attn_pipe: pipeline.ComputePipeline,
    gemv_mlp_pipe: pipeline.ComputePipeline,
    gemv_logits_pipe: pipeline.ComputePipeline,
    swiglu_pipe: pipeline.ComputePipeline,
    gate_up_pipe: pipeline.ComputePipeline,
    add_rmsnorm_pipe: pipeline.ComputePipeline,
    rmsnorm_pipe: pipeline.ComputePipeline,
    fused_qkv_pipe: pipeline.ComputePipeline,
    attn_pipe: pipeline.ComputePipeline,
    qkv_rope_pipe: pipeline.ComputePipeline,
    argmax_pipe: pipeline.ComputePipeline,
    topk_pass1_pipe: pipeline.ComputePipeline,
    topk_pass2_pipe: pipeline.ComputePipeline,
    quiescence_pipe: pipeline.ComputePipeline,
    cmd_pool: types.VkCommandPool,
    cmd_buf: types.VkCommandBuffer,
    fence: types.VkFence,

    pub fn init(ctx: *const context.GpuContext, mode: quant.QuantMode) !GpuEngine {
        const attn_spirv = switch (mode) { .none => &shaders.GEMV_BF16_SPIRV, .q8, .mixed => &shaders.GEMV_Q8_SPIRV, .q4 => &shaders.GEMV_Q4_SPIRV };
        const mlp_spirv = switch (mode) { .none => &shaders.GEMV_BF16_SPIRV, .q8, .mixed => &shaders.GEMV_Q8_SPIRV, .q4 => &shaders.GEMV_Q4_SPIRV };
        var gemv_attn = try pipeline.ComputePipeline.init(ctx, attn_spirv, 3, 16); errdefer gemv_attn.deinit();
        var gemv_mlp = try pipeline.ComputePipeline.init(ctx, mlp_spirv, 3, 16); errdefer gemv_mlp.deinit();
        var gemv_logits = try pipeline.ComputePipeline.init(ctx, &shaders.GEMV_Q8_SPIRV, 3, 16); errdefer gemv_logits.deinit();
        var swiglu = try pipeline.ComputePipeline.init(ctx, &shaders.FUSED_SWIGLU_SPIRV, 3, 4); errdefer swiglu.deinit();
        const gate_up_spirv = switch (mode) { .q4, .mixed => &shaders.FUSED_GATE_UP_SWIGLU_Q4_SPIRV, .q8 => &shaders.FUSED_GATE_UP_SWIGLU_Q8_SPIRV, .none => &shaders.FUSED_GATE_UP_SWIGLU_BF16_SPIRV };
        var gate_up = try pipeline.ComputePipeline.init(ctx, gate_up_spirv, 4, 8); errdefer gate_up.deinit();
        var add_rms = try pipeline.ComputePipeline.init(ctx, &shaders.FUSED_ADD_RMSNORM_SPIRV, 4, 12); errdefer add_rms.deinit();
        var rms = try pipeline.ComputePipeline.init(ctx, &shaders.RMSNORM_SPIRV, 3, 8); errdefer rms.deinit();
        var fused_qkv = try pipeline.ComputePipeline.init(ctx, &shaders.FUSED_QKV_Q4_SPIRV, 7, 16); errdefer fused_qkv.deinit();
        var attn = try pipeline.ComputePipeline.init(ctx, &shaders.DECODE_ATTENTION_SPIRV, 6, 16); errdefer attn.deinit();
        var qkv_rope = try pipeline.ComputePipeline.init(ctx, &shaders.QKV_ROPE_SPIRV, 9, 32); errdefer qkv_rope.deinit();
        var argmax = try pipeline.ComputePipeline.init(ctx, &shaders.ARGMAX_SPIRV, 2, 4); errdefer argmax.deinit();
        var topk_pass1 = try pipeline.ComputePipeline.init(ctx, &shaders.TOPK_PASS1_SPIRV, 3, 4); errdefer topk_pass1.deinit();
        var topk_pass2 = try pipeline.ComputePipeline.init(ctx, &shaders.TOPK_PASS2_SPIRV, 4, 4); errdefer topk_pass2.deinit();
        var quiescence = try pipeline.ComputePipeline.init(ctx, &shaders.QUIESCENCE_GATE_SPIRV, 3, @sizeOf(QuiescenceGatePushConstants)); errdefer quiescence.deinit();

        const cp_info = types_dispatch.VkCommandPoolCreateInfo{ .flags = types.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = ctx.queue_family_index };
        var pool: types.VkCommandPool = null;
        if (ctx.api.vkCreateCommandPool(ctx.device, &cp_info, null, &pool) != .SUCCESS) return error.VkCmdPoolCreationFailed;
        errdefer ctx.api.vkDestroyCommandPool(ctx.device, pool, null);
        const cb_info = types_dispatch.VkCommandBufferAllocateInfo{ .commandPool = pool, .commandBufferCount = 1 };
        var cmd: types.VkCommandBuffer = null;
        if (ctx.api.vkAllocateCommandBuffers(ctx.device, &cb_info, (&cmd)[0..1].ptr) != .SUCCESS) return error.VkCmdBufferAllocFailed;
        const fence_info = types_dispatch.VkFenceCreateInfo{};
        var fence: types.VkFence = null;
        if (ctx.api.vkCreateFence(ctx.device, &fence_info, null, &fence) != .SUCCESS) return error.VkFenceCreationFailed;

        return .{
            .ctx = ctx, .mode = mode, .gemv_attn_pipe = gemv_attn, .gemv_mlp_pipe = gemv_mlp, .gemv_logits_pipe = gemv_logits,
            .swiglu_pipe = swiglu, .gate_up_pipe = gate_up, .add_rmsnorm_pipe = add_rms,
            .rmsnorm_pipe = rms, .fused_qkv_pipe = fused_qkv, .attn_pipe = attn, .qkv_rope_pipe = qkv_rope,
            .argmax_pipe = argmax, .topk_pass1_pipe = topk_pass1, .topk_pass2_pipe = topk_pass2, .quiescence_pipe = quiescence,
            .cmd_pool = pool, .cmd_buf = cmd, .fence = fence,
        };
    }

    pub fn deinit(self: *GpuEngine) void {
        _ = self.ctx.api.vkQueueWaitIdle(self.ctx.queue);
        self.ctx.api.vkDestroyFence(self.ctx.device, self.fence, null);
        self.ctx.api.vkDestroyCommandPool(self.ctx.device, self.cmd_pool, null);
        self.qkv_rope_pipe.deinit(); self.attn_pipe.deinit(); self.fused_qkv_pipe.deinit(); self.rmsnorm_pipe.deinit();
        self.add_rmsnorm_pipe.deinit(); self.gate_up_pipe.deinit(); self.swiglu_pipe.deinit();
        self.gemv_logits_pipe.deinit(); self.gemv_mlp_pipe.deinit(); self.gemv_attn_pipe.deinit(); self.argmax_pipe.deinit();
        self.topk_pass1_pipe.deinit(); self.topk_pass2_pipe.deinit();
        self.quiescence_pipe.deinit();
    }

    pub fn recordGemv(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, m: usize, k: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k), 0, 0 };
        self.gemv_attn_pipe.record(cmd, set, std.mem.sliceAsBytes(&pc), @intCast((m + 7) / 8), 1, 1);
    }
    pub fn recordFusedQkv(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, q_dim: usize, kv_dim: usize, K: usize) void {
        const pc = extern struct { q_dim: u32, kv_dim: u32, K: u32, pad: u32 }{
            .q_dim = @intCast(q_dim), .kv_dim = @intCast(kv_dim), .K = @intCast(K), .pad = 0,
        };
        const total_rows = q_dim + kv_dim + kv_dim;
        const workgroups = (total_rows + 7) / 8;
        self.fused_qkv_pipe.record(cmd, set, std.mem.asBytes(&pc), @intCast(workgroups), 1, 1);
    }
    pub fn recordFusedQkvIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, q_dim: usize, kv_dim: usize, K: usize, ind: types.VkBuffer, off: u64) void {
        const pc = extern struct { q_dim: u32, kv_dim: u32, K: u32, pad: u32 }{
            .q_dim = @intCast(q_dim), .kv_dim = @intCast(kv_dim), .K = @intCast(K), .pad = 0,
        };
        self.fused_qkv_pipe.recordIndirect(cmd, set, std.mem.asBytes(&pc), ind, off);
    }
    pub fn recordGemvMlp(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, m: usize, k: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k), 0, 0 };
        self.gemv_mlp_pipe.record(cmd, set, std.mem.sliceAsBytes(&pc), @intCast((m + 7) / 8), 1, 1);
    }
    pub fn recordGemvIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, m: usize, k: usize, ind: types.VkBuffer, off: u64) void {
        const pc = [_]u32{ @intCast(m), @intCast(k), 0, 0 };
        self.gemv_attn_pipe.recordIndirect(cmd, set, std.mem.sliceAsBytes(&pc), ind, off);
    }
    pub fn recordGemvMlpIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, m: usize, k: usize, ind: types.VkBuffer, off: u64) void {
        const pc = [_]u32{ @intCast(m), @intCast(k), 0, 0 };
        self.gemv_mlp_pipe.recordIndirect(cmd, set, std.mem.sliceAsBytes(&pc), ind, off);
    }
    pub fn recordGemvLogits(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, m: usize, k: usize, x_offset: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k), @intCast(x_offset), 0 };
        self.gemv_logits_pipe.record(cmd, set, std.mem.sliceAsBytes(&pc), @intCast((m + 7) / 8), 1, 1);
    }
    pub fn recordGateUpSwiGlu(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, m: usize, k: usize) void {
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        self.gate_up_pipe.record(cmd, set, std.mem.sliceAsBytes(&pc), @intCast((m + 7) / 8), 1, 1);
    }
    pub fn recordGateUpSwiGluIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, m: usize, k: usize, ind: types.VkBuffer, off: u64) void {
        const pc = [_]u32{ @intCast(m), @intCast(k) };
        self.gate_up_pipe.recordIndirect(cmd, set, std.mem.sliceAsBytes(&pc), ind, off);
    }
    pub fn recordAddRmsNorm(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, H: usize, eps: f32, scalar: f32) void {
        const pc = extern struct { h: u32, eps: f32, scalar: f32 }{ .h = @intCast(H), .eps = eps, .scalar = scalar };
        self.add_rmsnorm_pipe.record(cmd, set, std.mem.asBytes(&pc), 1, 1, 1);
    }
    pub fn recordAddRmsNormIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, H: usize, eps: f32, scalar: f32, ind: types.VkBuffer, off: u64) void {
        const pc = extern struct { h: u32, eps: f32, scalar: f32 }{ .h = @intCast(H), .eps = eps, .scalar = scalar };
        self.add_rmsnorm_pipe.recordIndirect(cmd, set, std.mem.asBytes(&pc), ind, off);
    }
    pub fn recordRmsNorm(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, H: usize, eps: f32) void {
        const pc = extern struct { h: u32, eps: f32 }{ .h = @intCast(H), .eps = eps };
        self.rmsnorm_pipe.record(cmd, set, std.mem.asBytes(&pc), 1, 1, 1);
    }
    pub fn recordRmsNormIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, H: usize, eps: f32, ind: types.VkBuffer, off: u64) void {
        const pc = extern struct { h: u32, eps: f32 }{ .h = @intCast(H), .eps = eps };
        self.rmsnorm_pipe.recordIndirect(cmd, set, std.mem.asBytes(&pc), ind, off);
    }
    pub fn recordDecodeAttn(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, head_dim: usize, kv_dim: usize, gqa_ratio: usize, inv_sqrt_dim: f32, is_sliding: bool, num_q_heads: usize) void {
        const pc = extern struct { head_dim: u32, kv_dim: u32, gqa_ratio: u32, inv_sqrt_dim: f32, is_sliding: u32, pad: u32 }{
            .head_dim = @intCast(head_dim), .kv_dim = @intCast(kv_dim), .gqa_ratio = @intCast(gqa_ratio), .inv_sqrt_dim = inv_sqrt_dim, .is_sliding = if (is_sliding) 1 else 0, .pad = 0,
        };
        self.attn_pipe.record(cmd, set, std.mem.asBytes(&pc), @intCast(num_q_heads), 1, 1);
    }
    pub fn recordDecodeAttnIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, head_dim: usize, kv_dim: usize, gqa_ratio: usize, inv_sqrt_dim: f32, is_sliding: bool, ind: types.VkBuffer, off: u64) void {
        const pc = extern struct { head_dim: u32, kv_dim: u32, gqa_ratio: u32, inv_sqrt_dim: f32, is_sliding: u32, pad: u32 }{
            .head_dim = @intCast(head_dim), .kv_dim = @intCast(kv_dim), .gqa_ratio = @intCast(gqa_ratio), .inv_sqrt_dim = inv_sqrt_dim, .is_sliding = if (is_sliding) 1 else 0, .pad = 0,
        };
        self.attn_pipe.recordIndirect(cmd, set, std.mem.asBytes(&pc), ind, off);
    }
    pub fn recordQkvRope(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, num_q: usize, num_kv: usize, head_dim: usize, rotary_dim: usize, k_eq_v: bool, theta: f32, eps: f32) void {
        const pc = extern struct { num_q: u32, num_kv: u32, head_dim: u32, rotary_dim: u32, k_eq_v: u32, theta: f32, eps: f32, pad: u32 }{
            .num_q = @intCast(num_q), .num_kv = @intCast(num_kv), .head_dim = @intCast(head_dim), .rotary_dim = @intCast(rotary_dim), .k_eq_v = if (k_eq_v) 1 else 0, .theta = theta, .eps = eps, .pad = 0,
        };
        self.qkv_rope_pipe.record(cmd, set, std.mem.asBytes(&pc), @intCast(num_q + num_kv), 1, 1);
    }
    pub fn recordQkvRopeIndirect(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, num_q: usize, num_kv: usize, head_dim: usize, rotary_dim: usize, k_eq_v: bool, theta: f32, eps: f32, ind: types.VkBuffer, off: u64) void {
        const pc = extern struct { num_q: u32, num_kv: u32, head_dim: u32, rotary_dim: u32, k_eq_v: u32, theta: f32, eps: f32, pad: u32 }{
            .num_q = @intCast(num_q), .num_kv = @intCast(num_kv), .head_dim = @intCast(head_dim), .rotary_dim = @intCast(rotary_dim), .k_eq_v = if (k_eq_v) 1 else 0, .theta = theta, .eps = eps, .pad = 0,
        };
        self.qkv_rope_pipe.recordIndirect(cmd, set, std.mem.asBytes(&pc), ind, off);
    }
    pub fn recordQuiescenceGate(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, pc: *const QuiescenceGatePushConstants) void {
        self.quiescence_pipe.record(cmd, set, std.mem.asBytes(pc), 1, 1, 1);
    }
    pub fn recordArgmax(self: *const GpuEngine, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, vocab_size: usize) void {
        const pc = extern struct { v: u32 }{ .v = @intCast(vocab_size) };
        self.argmax_pipe.record(cmd, set, std.mem.asBytes(&pc), 1, 1, 1);
    }
    pub fn recordTopK(self: *const GpuEngine, cmd: types.VkCommandBuffer, desc_p1: types.VkDescriptorSet, desc_p2: types.VkDescriptorSet, vocab_size: usize) void {
        const pc1 = extern struct { v: u32 }{ .v = @intCast(vocab_size) };
        self.topk_pass1_pipe.record(cmd, desc_p1, std.mem.asBytes(&pc1), 64, 1, 1);
        self.recordBarrier(cmd);
        const pc2 = extern struct { g: u32 }{ .g = 64 };
        self.topk_pass2_pipe.record(cmd, desc_p2, std.mem.asBytes(&pc2), 1, 1, 1);
    }
    pub fn recordBarrier(self: *const GpuEngine, cmd: types.VkCommandBuffer) void {
        const b = types_dispatch.VkMemoryBarrier{ .srcAccessMask = 0x00000020 | 0x00000040, .dstAccessMask = 0x00000020 | 0x00000040 };
        self.ctx.api.vkCmdPipelineBarrier(cmd, 0x00000800, 0x00000800, 0, 1, (&b)[0..1].ptr, 0, null, 0, null);
    }
    pub fn submitPreRecorded(self: *const GpuEngine, cmd_buf: types.VkCommandBuffer) !void {
        _ = self.ctx.api.vkResetFences(self.ctx.device, 1, (&self.fence)[0..1].ptr);
        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&cmd_buf)[0..1].ptr };
        if (self.ctx.api.vkQueueSubmit(self.ctx.queue, 1, (&submit_info)[0..1].ptr, self.fence) != .SUCCESS) return error.VkQueueSubmitFailed;
        if (self.ctx.api.vkWaitForFences(self.ctx.device, 1, (&self.fence)[0..1].ptr, 1, 5_000_000_000) != .SUCCESS) return error.VkFenceTimeout;
    }
};
