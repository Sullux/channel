const std = @import("std");
const model = @import("../model.zig");
const model_types = @import("types.zig");
const gpu = @import("../gpu.zig");

pub fn runGpuBenchmark(m: *const model.Model, config: model_types.ModelConfig, g: *gpu.model_gpu.GpuModelContext, stdout: anytype) !void {
    _ = config;
    try stdout.print("\n=== Single-Submission GPU Pipeline Benchmark ({d} layers) ===\n", .{m.layers.len});
    const iter: usize = 20;

    // Warmup
    try g.engine.submitPreRecorded(g.cmd_buf_decode);

    const t0 = std.time.nanoTimestamp();
    for (0..iter) |_| try g.engine.submitPreRecorded(g.cmd_buf_decode);
    const t1 = std.time.nanoTimestamp();

    const ms = (@as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(iter))) / 1_000_000.0;
    const tps = 1000.0 / ms;
    try stdout.print("GPU Pre-Recorded Execution:     {d:.2} ms / token\n", .{ms});
    try stdout.print("Measured GPU Compute Throughput: {d:.1} TOKENS / SEC\n\n", .{tps});
}
