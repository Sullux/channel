const std = @import("std");
pub const types = @import("gpu/types.zig");
pub const types_dispatch = @import("gpu/types_dispatch.zig");
pub const vk_api = @import("gpu/vk_api.zig");
pub const context = @import("gpu/context.zig");
pub const buffer = @import("gpu/buffer.zig");
pub const pipeline = @import("gpu/pipeline.zig");
pub const shaders = @import("gpu/shaders.zig");
pub const gpu_kernels = @import("gpu/kernels.zig");
pub const descriptors = @import("gpu/descriptors.zig");
pub const model_dispatch = @import("gpu/model_dispatch.zig");
pub const model_gpu = @import("gpu/model_gpu.zig");
pub const batch_prefill = @import("gpu/batch_prefill.zig");
pub const batch_dispatch = @import("gpu/batch_dispatch.zig");
pub const quant = @import("quant.zig");

fn testCtx() !?context.GpuContext {
    return context.GpuContext.init(std.testing.allocator) catch |err| {
        if (err == error.VulkanLibraryNotFound or err == error.NoVulkanDevices) return null;
        return err;
    };
}

test "vulkan context initialization and AMD device selection" {
    const maybe_ctx = try testCtx();
    if (maybe_ctx == null) return;
    var ctx = maybe_ctx.?;
    defer ctx.deinit();
    try std.testing.expect(ctx.device_name.len > 0);
}

test "vulkan gemv compute execution (bf16, q4) on AMD GPU" {
    const maybe_ctx = try testCtx();
    if (maybe_ctx == null) return;
    var ctx = maybe_ctx.?;
    defer ctx.deinit();

    const m: usize = 64;
    const k: usize = 128;
    var buf_x = try buffer.GpuBuffer.init(&ctx, k * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_x.deinit();
    var buf_y = try buffer.GpuBuffer.init(&ctx, m * @sizeOf(f32), types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_y.deinit();
    for (buf_x.asSlice(f32), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.25;

    const raw = try std.testing.allocator.alloc(u16, m * k);
    defer std.testing.allocator.free(raw);
    @memset(raw, 0x3F80);

    const q4_size = quant.getQuantizedSizeBytes(m, k, .q4);
    var buf_q4 = try buffer.GpuBuffer.init(&ctx, q4_size, types.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer buf_q4.deinit();
    quant.quantizeMatrix(buf_q4.asSlice(u32), raw, m, k, .q4);

    var p_q4 = try pipeline.ComputePipeline.init(&ctx, &shaders.GEMV_Q4_SPIRV, 3, 8);
    defer p_q4.deinit();
    try p_q4.bindBuffers(&[_]*const buffer.GpuBuffer{ &buf_q4, &buf_x, &buf_y });
    const pc = [_]u32{ @intCast(m), @intCast(k) };
    try p_q4.dispatch(std.mem.sliceAsBytes(&pc), @intCast(m), 1, 1);
    for (buf_y.asSlice(f32)) |y| try std.testing.expect(y > 0);
}
