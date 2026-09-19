const std = @import("std");
pub const types = @import("types.zig");
pub const types_dispatch = @import("types_dispatch.zig");
pub const context = @import("context.zig");

pub const GpuBuffer = struct {
    ctx: *const context.GpuContext,
    buffer: types.VkBuffer,
    memory: types.VkDeviceMemory,
    size: u64,
    mapped: ?*anyopaque,

    pub fn initCached(ctx: *const context.GpuContext, size: u64, usage: u32) !GpuBuffer {
        const buf_info = types.VkBufferCreateInfo{
            .size = size,
            .usage = usage,
        };

        var buffer: types.VkBuffer = null;
        if (ctx.api.vkCreateBuffer(ctx.device, &buf_info, null, &buffer) != .SUCCESS) return error.VkBufferCreationFailed;
        errdefer ctx.api.vkDestroyBuffer(ctx.device, buffer, null);

        var mem_reqs: types.VkMemoryRequirements = undefined;
        ctx.api.vkGetBufferMemoryRequirements(ctx.device, buffer, &mem_reqs);

        const mem_type = ctx.findMemoryType(
            mem_reqs.memoryTypeBits,
            types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | types.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        ) catch ctx.findMemoryType(
            mem_reqs.memoryTypeBits,
            types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        ) catch |err| return err;

        const alloc_info = types.VkMemoryAllocateInfo{
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type,
        };

        var memory: types.VkDeviceMemory = null;
        if (ctx.api.vkAllocateMemory(ctx.device, &alloc_info, null, &memory) != .SUCCESS) return error.VkMemoryAllocFailed;
        errdefer ctx.api.vkFreeMemory(ctx.device, memory, null);

        if (ctx.api.vkBindBufferMemory(ctx.device, buffer, memory, 0) != .SUCCESS) return error.VkBindBufferFailed;

        var mapped: ?*anyopaque = null;
        if (ctx.api.vkMapMemory(ctx.device, memory, 0, size, 0, &mapped) != .SUCCESS) return error.VkMapMemoryFailed;

        return .{
            .ctx = ctx,
            .buffer = buffer,
            .memory = memory,
            .size = size,
            .mapped = mapped,
        };
    }

    pub fn init(ctx: *const context.GpuContext, size: u64, usage: u32) !GpuBuffer {
        const buf_info = types.VkBufferCreateInfo{
            .size = size,
            .usage = usage,
        };

        var buffer: types.VkBuffer = null;
        if (ctx.api.vkCreateBuffer(ctx.device, &buf_info, null, &buffer) != .SUCCESS) return error.VkBufferCreationFailed;
        errdefer ctx.api.vkDestroyBuffer(ctx.device, buffer, null);

        var mem_reqs: types.VkMemoryRequirements = undefined;
        ctx.api.vkGetBufferMemoryRequirements(ctx.device, buffer, &mem_reqs);

        // Try DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (ideal for UMA)
        const mem_type = ctx.findMemoryType(
            mem_reqs.memoryTypeBits,
            types.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | types.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        ) catch |err| return err;

        const alloc_info = types.VkMemoryAllocateInfo{
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type,
        };

        var memory: types.VkDeviceMemory = null;
        if (ctx.api.vkAllocateMemory(ctx.device, &alloc_info, null, &memory) != .SUCCESS) return error.VkMemoryAllocFailed;
        errdefer ctx.api.vkFreeMemory(ctx.device, memory, null);

        if (ctx.api.vkBindBufferMemory(ctx.device, buffer, memory, 0) != .SUCCESS) return error.VkBindBufferFailed;

        var mapped: ?*anyopaque = null;
        if (ctx.api.vkMapMemory(ctx.device, memory, 0, size, 0, &mapped) != .SUCCESS) return error.VkMapMemoryFailed;

        return .{
            .ctx = ctx,
            .buffer = buffer,
            .memory = memory,
            .size = size,
            .mapped = mapped,
        };
    }

    pub fn deinit(self: *GpuBuffer) void {
        if (self.mapped != null) {
            self.ctx.api.vkUnmapMemory(self.ctx.device, self.memory);
            self.mapped = null;
        }
        self.ctx.api.vkDestroyBuffer(self.ctx.device, self.buffer, null);
        self.ctx.api.vkFreeMemory(self.ctx.device, self.memory, null);
    }

    pub fn asSlice(self: *GpuBuffer, comptime T: type) []T {
        const count = self.size / @sizeOf(T);
        return @as([*]T, @ptrCast(@alignCast(self.mapped.?)))[0..count];
    }
};
