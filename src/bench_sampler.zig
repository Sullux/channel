const std = @import("std");
const sampler_mod = @import("sampler.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    var s = sampler_mod.Sampler.init(1234, 0.7, 0.95);
    const logits = try alloc.alloc(f32, 256000);
    defer alloc.free(logits);

    for (logits, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.01) * 10.0;
    }

    const recent = [_]u32{ 100, 200, 300, 400, 500 };

    const start = std.time.nanoTimestamp();
    const iters = 1000;
    for (0..iters) |_| {
        _ = s.sample(logits, &recent);
    }
    const elapsed = std.time.nanoTimestamp() - start;
    const ns_per_call = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iters));
    std.debug.print("Sampler.sample: {d:.3} ms per call ({d:.1} calls/sec)\n", .{ ns_per_call / 1e6, 1e9 / ns_per_call });
}
