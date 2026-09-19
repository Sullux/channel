const std = @import("std");
pub const kernels = @import("../kernels.zig");
pub const diff = @import("../diff.zig");
pub const memory = @import("../memory.zig");
pub const types = @import("types.zig");
pub const loader = @import("loader.zig");
pub const ring_buffer = @import("../ring_buffer.zig");
pub const gpu = @import("../gpu.zig");

const Model = loader.Model;
const ForwardScratch = types.ForwardScratch;
const DynamicRingBuffer = ring_buffer.DynamicRingBuffer;

pub fn syncRecallToGpu(ring: *const DynamicRingBuffer, gpu_ctx: *gpu.model_gpu.GpuModelContext) void {
    const total_slots = ring.total_slots;
    for (0..gpu_ctx.layers.len) |l| {
        const recall_count = ring.recallSlots(l);
        if (recall_count == 0) continue;
        const r_start = ring.recallStart(l);
        const gpu_k = gpu_ctx.layers[l].buf_k_cache.asSlice(f32);
        const gpu_v = gpu_ctx.layers[l].buf_v_cache.asSlice(f32);
        for (0..recall_count) |r| {
            const slot = r_start + r;
            const r_slot = l * total_slots + slot;
            const r_off = r_slot * ring.max_kv_dim;
            const g_off = slot * ring.max_kv_dim;
            @memcpy(gpu_k[g_off .. g_off + ring.max_kv_dim], ring.k[r_off .. r_off + ring.max_kv_dim]);
            @memcpy(gpu_v[g_off .. g_off + ring.max_kv_dim], ring.v[r_off .. r_off + ring.max_kv_dim]);
        }
    }
}

pub fn primeSubconsciousMemory(
    mem: *memory.DiffArchive,
    ring: *DynamicRingBuffer,
    scratch: *ForwardScratch,
    query_vec: []const f32,
    now_ts: u64,
    gpu_opt: ?*gpu.model_gpu.GpuModelContext,
    current_clock: usize,
    theta: f32,
    head_dim: usize,
    rotary_dim: usize,
    num_kv_heads: usize,
) usize {
    const selected = mem.primeTier3(query_vec, now_ts, ring, scratch.recall_indices, current_clock, theta, head_dim, rotary_dim, num_kv_heads);
    if (selected > 0 and gpu_opt != null) {
        syncRecallToGpu(ring, gpu_opt.?);
    }
    return selected;
}

pub fn integrateMemory(self: *const Model, mem: *memory.DiffArchive, ring: *DynamicRingBuffer, scratch: *ForwardScratch, clock: usize, H: usize, gpu_opt: ?*gpu.model_gpu.GpuModelContext) void {
    _ = H;
    const now_ts: u64 = @intCast(clock);
    const query_vec = if (gpu_opt) |g| g.buf_x.asSlice(f32)[0..scratch.normed_x.len] else scratch.normed_x;
    const theta: f32 = self.config.rope_theta;
    const head_dim: usize = self.config.head_dim;
    const rot_dim: usize = if (self.layers.len > 0) self.layers[0].rotary_dim else self.config.head_dim;
    const num_kv_heads: usize = self.config.num_key_value_heads;
    _ = primeSubconsciousMemory(mem, ring, scratch, query_vec, now_ts, gpu_opt, clock, theta, head_dim, rot_dim, num_kv_heads);
}

pub fn computeKeywordQueryVector(self: *const Model, token_ids: []const u32, out_vector: []f32) bool {
    if (token_ids.len == 0 or out_vector.len != self.config.hidden_size) return false;
    @memset(out_vector, 0);

    var valid_tokens: usize = 0;
    const H = self.config.hidden_size;

    for (token_ids) |tok| {
        if (tok >= self.config.vocab_size) continue;
        const src = self.embed_tokens[tok * H .. (tok + 1) * H];
        for (out_vector, src) |*dst, s| {
            dst.* += s.toF32();
        }
        valid_tokens += 1;
    }

    if (valid_tokens == 0) return false;

    var sum_sq: f32 = 0.0;
    for (out_vector) |v| sum_sq += v * v;
    const inv = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;
    for (out_vector) |*v| v.* *= inv;
    return true;
}

pub fn searchExplicitMemory(mem: *memory.DiffArchive, query: []const f32, now: u64, out_indices: []usize, top_k: usize) usize {
    return mem.scan(query, now, out_indices, top_k);
}

test "primeSubconsciousMemory populates recall slots with valid token clock" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 48, 64, 512, 3456, 128);
    defer ring.deinit();

    var arch = try memory.DiffArchive.initWithKV(std.testing.allocator, 16, 8, 48, 64, .{});
    defer arch.deinit();

    const idx = arch.appendFullMeta(.{
        .episode_id = 1,
        .timestamp = 1740000000000,
        .last_accessed = 1740000000000,
        .start_clock = 500,
        .token_count = 10,
        .salience_norm = 1.0,
    }, &[_]f32{ 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

    var dummy_k: [64]f32 = undefined;
    var dummy_v: [64]f32 = undefined;
    @memset(&dummy_k, 3.14);
    @memset(&dummy_v, 2.71);
    ring.writeKV(16, 500, &dummy_k, &dummy_v);
    arch.copyKVFromRing(idx, &ring, 500, null);

    var recall_idx: [16]usize = undefined;
    const selected = arch.primeTier3(&[_]f32{ 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, 1740000001000, &ring, &recall_idx, 1000, 10000.0, 64, 64, 1);
    try std.testing.expectEqual(@as(usize, 1), selected);

    // Verify recall slot is active at clock 1000 (since start_clock = 500 <= 1000)
    var slots_buf: [4096]usize = undefined;
    const count = ring.getActiveSlotsLayer(16, 1000, false, 1024, &slots_buf);
    try std.testing.expect(count > 0);

    // Verify recall start slot is in the active list
    const r_start = ring.recallStart(16);
    var found = false;
    for (slots_buf[0..count]) |s| {
        if (s == r_start) { found = true; break; }
    }
    try std.testing.expect(found);
}
