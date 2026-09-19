const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");

pub const LayerDescriptorSets = struct {
    input_norm: types.VkDescriptorSet,
    fused_qkv: types.VkDescriptorSet,
    q_proj: types.VkDescriptorSet,
    k_proj: types.VkDescriptorSet,
    v_proj: types.VkDescriptorSet,
    qkv_rope: types.VkDescriptorSet,
    attn: types.VkDescriptorSet,
    o_proj: types.VkDescriptorSet,
    gate_proj: types.VkDescriptorSet,
    up_proj: types.VkDescriptorSet,
    swiglu: types.VkDescriptorSet,
    down_proj: types.VkDescriptorSet,
    gate_up_swiglu: types.VkDescriptorSet,
    pre_ffn_norm: types.VkDescriptorSet,
    post_attn_norm: types.VkDescriptorSet,
    post_ffn_norm: types.VkDescriptorSet,
    post_ffn_add: types.VkDescriptorSet,
    quiescence_gate: types.VkDescriptorSet,
};

pub const DescriptorManager = struct {
    ctx: *const context.GpuContext,
    pool: types.VkDescriptorPool,

    pub fn allocateLayerSets(self: *DescriptorManager, engine: anytype) !LayerDescriptorSets {
        return .{
            .input_norm = try self.allocateSet(engine.rmsnorm_pipe.desc_set_layout),
            .fused_qkv = try self.allocateSet(engine.fused_qkv_pipe.desc_set_layout),
            .q_proj = try self.allocateSet(engine.gemv_attn_pipe.desc_set_layout),
            .k_proj = try self.allocateSet(engine.gemv_attn_pipe.desc_set_layout),
            .v_proj = try self.allocateSet(engine.gemv_attn_pipe.desc_set_layout),
            .qkv_rope = try self.allocateSet(engine.qkv_rope_pipe.desc_set_layout),
            .attn = try self.allocateSet(engine.attn_pipe.desc_set_layout),
            .o_proj = try self.allocateSet(engine.gemv_attn_pipe.desc_set_layout),
            .gate_proj = try self.allocateSet(engine.gemv_mlp_pipe.desc_set_layout),
            .up_proj = try self.allocateSet(engine.gemv_mlp_pipe.desc_set_layout),
            .swiglu = try self.allocateSet(engine.swiglu_pipe.desc_set_layout),
            .down_proj = try self.allocateSet(engine.gemv_mlp_pipe.desc_set_layout),
            .gate_up_swiglu = try self.allocateSet(engine.gate_up_pipe.desc_set_layout),
            .pre_ffn_norm = try self.allocateSet(engine.add_rmsnorm_pipe.desc_set_layout),
            .post_attn_norm = try self.allocateSet(engine.rmsnorm_pipe.desc_set_layout),
            .post_ffn_norm = try self.allocateSet(engine.rmsnorm_pipe.desc_set_layout),
            .post_ffn_add = try self.allocateSet(engine.add_rmsnorm_pipe.desc_set_layout),
            .quiescence_gate = try self.allocateSet(engine.quiescence_pipe.desc_set_layout),
        };
    }

    pub fn init(ctx: *const context.GpuContext, max_sets: u32) !DescriptorManager {
        const pool_size = [_]types_dispatch.VkDescriptorPoolSize{
            .{ .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_sets * 16 },
        };
        const pool_info = types_dispatch.VkDescriptorPoolCreateInfo{
            .maxSets = max_sets,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var pool: types.VkDescriptorPool = null;
        if (ctx.api.vkCreateDescriptorPool(ctx.device, &pool_info, null, &pool) != .SUCCESS) return error.VkDescPoolCreationFailed;
        return .{ .ctx = ctx, .pool = pool };
    }

    pub fn deinit(self: *DescriptorManager) void {
        self.ctx.api.vkDestroyDescriptorPool(self.ctx.device, self.pool, null);
    }

    pub fn reset(self: *const DescriptorManager) void {
        _ = self.ctx.api.vkResetDescriptorPool(self.ctx.device, self.pool, 0);
    }

    pub fn allocateSet(self: *const DescriptorManager, layout: types.VkDescriptorSetLayout) !types.VkDescriptorSet {
        const alloc_info = types_dispatch.VkDescriptorSetAllocateInfo{
            .descriptorPool = self.pool,
            .descriptorSetCount = 1,
            .pSetLayouts = (&layout)[0..1].ptr,
        };
        var set: types.VkDescriptorSet = null;
        if (self.ctx.api.vkAllocateDescriptorSets(self.ctx.device, &alloc_info, (&set)[0..1].ptr) != .SUCCESS) return error.VkDescSetAllocFailed;
        return set;
    }

    pub fn bindBuffers(self: *const DescriptorManager, set: types.VkDescriptorSet, bufs: []const *const buffer.GpuBuffer) void {
        var writes: [16]types_dispatch.VkWriteDescriptorSet = undefined;
        var infos: [16]types_dispatch.VkDescriptorBufferInfo = undefined;
        const count = @min(bufs.len, 16);

        for (0..count) |i| {
            infos[i] = .{
                .buffer = bufs[i].buffer,
                .offset = 0,
                .range = 0xFFFFFFFFFFFFFFFF, // VK_WHOLE_SIZE
            };
            writes[i] = .{
                .dstSet = set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = (&infos[i])[0..1].ptr,
            };
        }
        self.ctx.api.vkUpdateDescriptorSets(self.ctx.device, @intCast(count), &writes, 0, null);
    }
};
