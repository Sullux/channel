const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");

pub const ComputePipeline = struct {
    ctx: *const context.GpuContext,
    shader_module: types.VkShaderModule,
    desc_set_layout: types.VkDescriptorSetLayout,
    pipeline_layout: types.VkPipelineLayout,
    pipeline: types.VkPipeline,
    desc_pool: types.VkDescriptorPool,
    desc_set: types.VkDescriptorSet,
    cmd_pool: types.VkCommandPool,
    cmd_buffer: types.VkCommandBuffer,
    fence: types.VkFence,
    num_bindings: u32,

    pub fn init(ctx: *const context.GpuContext, spirv_code: []const u32, num_bindings: u32, push_constant_size: u32) !ComputePipeline {
        const sm_info = types_dispatch.VkShaderModuleCreateInfo{ .codeSize = spirv_code.len * @sizeOf(u32), .pCode = spirv_code.ptr };
        var shader_module: types.VkShaderModule = null;
        if (ctx.api.vkCreateShaderModule(ctx.device, &sm_info, null, &shader_module) != .SUCCESS) return error.VkShaderModuleCreationFailed;
        errdefer ctx.api.vkDestroyShaderModule(ctx.device, shader_module, null);

        var bindings = try ctx.allocator.alloc(types_dispatch.VkDescriptorSetLayoutBinding, num_bindings);
        defer ctx.allocator.free(bindings);
        for (0..num_bindings) |i| {
            bindings[i] = .{ .binding = @intCast(i), .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT };
        }
        const dsl_info = types_dispatch.VkDescriptorSetLayoutCreateInfo{ .bindingCount = num_bindings, .pBindings = bindings.ptr };
        var desc_set_layout: types.VkDescriptorSetLayout = null;
        if (ctx.api.vkCreateDescriptorSetLayout(ctx.device, &dsl_info, null, &desc_set_layout) != .SUCCESS) return error.VkDescLayoutCreationFailed;
        errdefer ctx.api.vkDestroyDescriptorSetLayout(ctx.device, desc_set_layout, null);

        const push_range = types_dispatch.VkPushConstantRange{ .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_constant_size };
        const pl_info = types_dispatch.VkPipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = (&desc_set_layout)[0..1].ptr,
            .pushConstantRangeCount = if (push_constant_size > 0) 1 else 0,
            .pPushConstantRanges = if (push_constant_size > 0) (&push_range)[0..1].ptr else null,
        };
        var pipeline_layout: types.VkPipelineLayout = null;
        if (ctx.api.vkCreatePipelineLayout(ctx.device, &pl_info, null, &pipeline_layout) != .SUCCESS) return error.VkPipelineLayoutCreationFailed;
        errdefer ctx.api.vkDestroyPipelineLayout(ctx.device, pipeline_layout, null);

        const main_name: [:0]const u8 = "main";
        const stage_info = types_dispatch.VkPipelineShaderStageCreateInfo{
            .sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = types.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = main_name.ptr,
            .pSpecializationInfo = null,
        };
        const cp_info = types_dispatch.VkComputePipelineCreateInfo{
            .sType = .COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = stage_info,
            .layout = pipeline_layout,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };
        var pipeline: types.VkPipeline = null;
        if (ctx.api.vkCreateComputePipelines(ctx.device, null, 1, (&cp_info)[0..1].ptr, null, (&pipeline)[0..1].ptr) != .SUCCESS) return error.VkPipelineCreationFailed;
        errdefer ctx.api.vkDestroyPipeline(ctx.device, pipeline, null);

        const pool_size = [_]types_dispatch.VkDescriptorPoolSize{ .{ .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = num_bindings } };
        const dp_info = types_dispatch.VkDescriptorPoolCreateInfo{ .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size };
        var desc_pool: types.VkDescriptorPool = null;
        if (ctx.api.vkCreateDescriptorPool(ctx.device, &dp_info, null, &desc_pool) != .SUCCESS) return error.VkDescPoolCreationFailed;
        errdefer ctx.api.vkDestroyDescriptorPool(ctx.device, desc_pool, null);

        const da_info = types_dispatch.VkDescriptorSetAllocateInfo{ .descriptorPool = desc_pool, .descriptorSetCount = 1, .pSetLayouts = (&desc_set_layout)[0..1].ptr };
        var desc_set: types.VkDescriptorSet = null;
        if (ctx.api.vkAllocateDescriptorSets(ctx.device, &da_info, (&desc_set)[0..1].ptr) != .SUCCESS) return error.VkDescSetAllocFailed;

        const cp_pool_info = types_dispatch.VkCommandPoolCreateInfo{ .queueFamilyIndex = ctx.queue_family_index };
        var cmd_pool: types.VkCommandPool = null;
        if (ctx.api.vkCreateCommandPool(ctx.device, &cp_pool_info, null, &cmd_pool) != .SUCCESS) return error.VkCmdPoolCreationFailed;
        errdefer ctx.api.vkDestroyCommandPool(ctx.device, cmd_pool, null);

        const cb_info = types_dispatch.VkCommandBufferAllocateInfo{ .commandPool = cmd_pool, .commandBufferCount = 1 };
        var cmd_buffer: types.VkCommandBuffer = null;
        if (ctx.api.vkAllocateCommandBuffers(ctx.device, &cb_info, (&cmd_buffer)[0..1].ptr) != .SUCCESS) return error.VkCmdBufferAllocFailed;

        const fence_info = types_dispatch.VkFenceCreateInfo{};
        var fence: types.VkFence = null;
        if (ctx.api.vkCreateFence(ctx.device, &fence_info, null, &fence) != .SUCCESS) return error.VkFenceCreationFailed;

        return .{
            .ctx = ctx,
            .shader_module = shader_module,
            .desc_set_layout = desc_set_layout,
            .pipeline_layout = pipeline_layout,
            .pipeline = pipeline,
            .desc_pool = desc_pool,
            .desc_set = desc_set,
            .cmd_pool = cmd_pool,
            .cmd_buffer = cmd_buffer,
            .fence = fence,
            .num_bindings = num_bindings,
        };
    }

    pub fn deinit(self: *ComputePipeline) void {
        self.ctx.api.vkDestroyFence(self.ctx.device, self.fence, null);
        self.ctx.api.vkDestroyCommandPool(self.ctx.device, self.cmd_pool, null);
        self.ctx.api.vkDestroyDescriptorPool(self.ctx.device, self.desc_pool, null);
        self.ctx.api.vkDestroyPipeline(self.ctx.device, self.pipeline, null);
        self.ctx.api.vkDestroyPipelineLayout(self.ctx.device, self.pipeline_layout, null);
        self.ctx.api.vkDestroyDescriptorSetLayout(self.ctx.device, self.desc_set_layout, null);
        self.ctx.api.vkDestroyShaderModule(self.ctx.device, self.shader_module, null);
    }

    pub fn bindBuffers(self: *const ComputePipeline, buffers: []const *const buffer.GpuBuffer) !void {
        var writes: [8]types_dispatch.VkWriteDescriptorSet = undefined;
        var buf_infos: [8]types_dispatch.VkDescriptorBufferInfo = undefined;
        const count = @min(buffers.len, 8);
        for (0..count) |i| {
            buf_infos[i] = .{ .buffer = buffers[i].buffer, .offset = 0, .range = buffers[i].size };
            writes[i] = .{ .dstSet = self.desc_set, .dstBinding = @intCast(i), .descriptorCount = 1, .descriptorType = types.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = (&buf_infos[i])[0..1].ptr };
        }
        self.ctx.api.vkUpdateDescriptorSets(self.ctx.device, @intCast(count), &writes, 0, null);
    }

    pub fn record(self: *const ComputePipeline, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, pc: ?[]const u8, gx: u32, gy: u32, gz: u32) void {
        self.ctx.api.vkCmdBindPipeline(cmd, 1, self.pipeline);
        self.ctx.api.vkCmdBindDescriptorSets(cmd, 1, self.pipeline_layout, 0, 1, (&set)[0..1].ptr, 0, null);
        if (pc) |p| self.ctx.api.vkCmdPushConstants(cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(p.len), p.ptr);
        self.ctx.api.vkCmdDispatch(cmd, gx, gy, gz);
    }

    pub fn recordIndirect(self: *const ComputePipeline, cmd: types.VkCommandBuffer, set: types.VkDescriptorSet, pc: ?[]const u8, indirect_buf: types.VkBuffer, offset: u64) void {
        self.ctx.api.vkCmdBindPipeline(cmd, 1, self.pipeline);
        self.ctx.api.vkCmdBindDescriptorSets(cmd, 1, self.pipeline_layout, 0, 1, (&set)[0..1].ptr, 0, null);
        if (pc) |p| self.ctx.api.vkCmdPushConstants(cmd, self.pipeline_layout, types.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(p.len), p.ptr);
        self.ctx.api.vkCmdDispatchIndirect(cmd, indirect_buf, offset);
    }

    pub fn dispatch(self: *const ComputePipeline, push_constants: ?[]const u8, gx: u32, gy: u32, gz: u32) !void {
        _ = self.ctx.api.vkResetFences(self.ctx.device, 1, (&self.fence)[0..1].ptr);
        const begin_info = types_dispatch.VkCommandBufferBeginInfo{};
        _ = self.ctx.api.vkBeginCommandBuffer(self.cmd_buffer, &begin_info);
        self.record(self.cmd_buffer, self.desc_set, push_constants, gx, gy, gz);
        _ = self.ctx.api.vkEndCommandBuffer(self.cmd_buffer);

        const submit_info = types_dispatch.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = (&self.cmd_buffer)[0..1].ptr };
        if (self.ctx.api.vkQueueSubmit(self.ctx.queue, 1, (&submit_info)[0..1].ptr, self.fence) != .SUCCESS) return error.VkQueueSubmitFailed;
        var spin: usize = 0;
        while (spin < 500_000) : (spin += 1) {
            if (self.ctx.api.vkGetFenceStatus(self.ctx.device, self.fence) == .SUCCESS) return;
            std.atomic.spinLoopHint();
        }
        if (self.ctx.api.vkWaitForFences(self.ctx.device, 1, (&self.fence)[0..1].ptr, 1, 5_000_000_000) != .SUCCESS) return error.VkFenceTimeout;
    }
};
