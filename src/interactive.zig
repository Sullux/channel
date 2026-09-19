const std = @import("std");
pub const model = @import("model.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const ring_buffer = @import("ring_buffer.zig");
pub const memory = @import("memory.zig");
pub const quiescence = @import("quiescence.zig");
pub const gpu = @import("gpu.zig");

pub fn printToken(stdout: anytype, token_str: []const u8) !void {
    var buf: [512]u8 = undefined;
    var out_len: usize = 0;
    var i: usize = 0;
    while (i < token_str.len and out_len < buf.len) {
        if (i + 2 < token_str.len and token_str[i] == 0xE2 and token_str[i + 1] == 0x96 and token_str[i + 2] == 0x81) {
            buf[out_len] = ' ';
            out_len += 1;
            i += 3;
        } else {
            buf[out_len] = token_str[i];
            out_len += 1;
            i += 1;
        }
    }
    try stdout.writeAll(buf[0..out_len]);
}

pub fn runInference(m: *const model.Model, tok: *const tokenizer.Tokenizer, ring: *ring_buffer.DynamicRingBuffer, scratch: *model.ForwardScratch, prompt: []const u8, max_tokens: usize, thread_pool: *std.Thread.Pool, stdout: anytype, allocator: std.mem.Allocator, memory_opt: ?*memory.DiffArchive, quiescence_opt: ?*quiescence.QuiescenceTracker, gpu_opt: ?*gpu.model_gpu.GpuModelContext, clock_ptr: *usize, reset_ring: bool) !void {
    const prompt_tokens = try tok.encode(allocator, prompt, true);
    defer allocator.free(prompt_tokens);
    if (reset_ring) ring.reset();

    var current_token: u32 = 0;
    for (prompt_tokens, 0..) |t, i| {
        const is_last = (i == prompt_tokens.len - 1);
        current_token = m.forwardToken(ring, scratch, t, clock_ptr.*, thread_pool, memory_opt, quiescence_opt, gpu_opt, is_last);
        clock_ptr.* += 1;
    }

    const gen_start = std.time.milliTimestamp();
    var gen_count: usize = 0;
    for (0..max_tokens) |_| {
        const token_str = tok.decode(current_token);
        try printToken(stdout, token_str);
        gen_count += 1;
        if (current_token == tok.eos_token_id or current_token == 106) break;
        current_token = m.forwardToken(ring, scratch, current_token, clock_ptr.*, thread_pool, memory_opt, quiescence_opt, gpu_opt, true);
        clock_ptr.* += 1;
    }
    const elapsed_ms = std.time.milliTimestamp() - gen_start;
    if (gen_count > 0 and elapsed_ms > 0) {
        const tps = (@as(f64, @floatFromInt(gen_count)) / @as(f64, @floatFromInt(elapsed_ms))) * 1000.0;
        try stdout.print("\n[{d} tokens, {d:.1} tok/s]", .{ gen_count, tps });
    }
}
