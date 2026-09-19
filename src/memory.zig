const std = @import("std");
pub const ring_buffer = @import("ring_buffer.zig");
pub const gpu = @import("gpu.zig");
const DynamicRingBuffer = ring_buffer.DynamicRingBuffer;

pub const SalienceConfig = struct {
    alpha: f32 = 1.0,
    beta: f32 = 0.5,
    gamma: f32 = 0.3,
    delta: f32 = 0.2,
    lambda: f32 = 1.0 / 86400000.0,
};

pub const MemoryMeta = struct {
    episode_id: u64 = 0,
    parent_episode_id: u64 = 0,
    timestamp: u64 = 0,
    last_accessed: u64 = 0,
    start_clock: u64 = 0,
    token_count: u32 = 1,
    access_count: u32 = 0,
    child_count: u32 = 0,
    salience_norm: f32 = 1.0,
    layer_id: u8 = 0,
    is_interrupted: bool = false,
    token_id: u32 = 0,
};

pub const DiffArchive = struct {
    allocator: std.mem.Allocator,
    dim: usize,
    capacity: usize,
    count: usize = 0,
    write_head: usize = 0,
    num_layers: usize,
    max_kv_dim: usize,
    kv_stride: usize,
    config: SalienceConfig,

    vectors: []f32,
    metas: []MemoryMeta,
    scan_scores: []f32,
    scan_indices: []usize,
    kv_cache: []f32,

    pub fn init(allocator: std.mem.Allocator, dim: usize, capacity: usize, config: SalienceConfig) !DiffArchive {
        return initWithKV(allocator, dim, capacity, 0, 0, config);
    }

    pub fn initWithKV(allocator: std.mem.Allocator, dim: usize, capacity: usize, num_layers: usize, max_kv_dim: usize, config: SalienceConfig) !DiffArchive {
        const vectors = try allocator.alloc(f32, capacity * dim);
        errdefer allocator.free(vectors);
        const metas = try allocator.alloc(MemoryMeta, capacity);
        errdefer allocator.free(metas);
        const scores = try allocator.alloc(f32, capacity);
        errdefer allocator.free(scores);
        const indices = try allocator.alloc(usize, capacity);
        errdefer allocator.free(indices);

        const kv_stride = num_layers * max_kv_dim * 2;
        var kv_cache: []f32 = &.{};
        if (kv_stride > 0) {
            kv_cache = try allocator.alloc(f32, capacity * kv_stride);
            @memset(kv_cache, 0);
        }
        errdefer if (kv_cache.len > 0) allocator.free(kv_cache);
        @memset(vectors, 0);

        return .{
            .allocator = allocator,
            .dim = dim,
            .capacity = capacity,
            .num_layers = num_layers,
            .max_kv_dim = max_kv_dim,
            .kv_stride = kv_stride,
            .config = config,
            .vectors = vectors,
            .metas = metas,
            .scan_scores = scores,
            .scan_indices = indices,
            .kv_cache = kv_cache,
        };
    }

    pub fn deinit(self: *DiffArchive) void {
        self.allocator.free(self.vectors);
        self.allocator.free(self.metas);
        self.allocator.free(self.scan_scores);
        self.allocator.free(self.scan_indices);
        if (self.kv_cache.len > 0) self.allocator.free(self.kv_cache);
    }

    pub fn reset(self: *DiffArchive) void {
        self.count = 0;
        self.write_head = 0;
        if (self.kv_cache.len > 0) @memset(self.kv_cache, 0);
    }

    pub fn copyKVFromRing(self: *DiffArchive, idx: usize, ring: *const DynamicRingBuffer, slot: usize, gpu_opt: ?*gpu.model_gpu.GpuModelContext) void {
        if (self.kv_cache.len == 0) return;
        const base = idx * self.kv_stride;
        for (0..self.num_layers) |l| {
            const dst = base + l * self.max_kv_dim * 2;
            if (gpu_opt) |g| {
                if (l < g.layers.len) {
                    const g_k = g.layers[l].buf_k_cache.asSlice(f32);
                    const g_v = g.layers[l].buf_v_cache.asSlice(f32);
                    const g_off = slot * ring.max_kv_dim;
                    if (g_off + self.max_kv_dim <= g_k.len and g_off + self.max_kv_dim <= g_v.len) {
                        @memcpy(self.kv_cache[dst .. dst + self.max_kv_dim], g_k[g_off .. g_off + self.max_kv_dim]);
                        @memcpy(self.kv_cache[dst + self.max_kv_dim .. dst + self.max_kv_dim * 2], g_v[g_off .. g_off + self.max_kv_dim]);
                        continue;
                    }
                }
            }
            const r_slot = l * ring.total_slots + slot;
            const r_off = r_slot * ring.max_kv_dim;
            @memcpy(self.kv_cache[dst .. dst + self.max_kv_dim], ring.k[r_off .. r_off + self.max_kv_dim]);
            @memcpy(self.kv_cache[dst + self.max_kv_dim .. dst + self.max_kv_dim * 2], ring.v[r_off .. r_off + self.max_kv_dim]);
        }
    }

    pub fn copyKVToRing(
        self: *const DiffArchive,
        idx: usize,
        ring: *DynamicRingBuffer,
        rank: usize,
        mem_clock: usize,
        current_clock: usize,
        theta: f32,
        head_dim: usize,
        rotary_dim: usize,
        num_kv_heads: usize,
    ) void {
        if (self.kv_cache.len == 0) return;
        const base = idx * self.kv_stride;
        const target_clock = if (current_clock > rank + 1) current_clock - (rank + 1) else 0;
        const delta_clock = @as(i64, @intCast(target_clock)) - @as(i64, @intCast(mem_clock));

        var k_buf: [4096]f32 = undefined;
        const kv_dim = @min(self.max_kv_dim, k_buf.len);

        for (0..self.num_layers) |l| {
            const src = base + l * self.max_kv_dim * 2;
            const k_src = self.kv_cache[src .. src + kv_dim];
            const v_src = self.kv_cache[src + self.max_kv_dim .. src + self.max_kv_dim + kv_dim];

            if (rotary_dim > 0 and head_dim > 0 and theta > 0.0 and delta_clock != 0 and num_kv_heads > 0) {
                @memcpy(k_buf[0..kv_dim], k_src);
                const half_rot = rotary_dim / 2;
                for (0..num_kv_heads) |h| {
                    const h_off = h * head_dim;
                    if (h_off + rotary_dim > kv_dim) break;
                    for (0..half_rot) |d| {
                        const freq_exp = (2.0 * @as(f32, @floatFromInt(d))) / @as(f32, @floatFromInt(rotary_dim));
                        const freq = 1.0 / std.math.pow(f32, theta, freq_exp);
                        const angle = @as(f32, @floatFromInt(delta_clock)) * freq;
                        const cos_a = @cos(angle);
                        const sin_a = @sin(angle);

                        const idx0 = h_off + d;
                        const idx1 = h_off + d + half_rot;
                        const k0 = k_src[idx0];
                        const k1 = k_src[idx1];
                        k_buf[idx0] = k0 * cos_a - k1 * sin_a;
                        k_buf[idx1] = k0 * sin_a + k1 * cos_a;
                    }
                }
                ring.writeRecallKV(l, rank, k_buf[0..kv_dim], v_src, target_clock);
            } else {
                ring.writeRecallKV(l, rank, k_src, v_src, target_clock);
            }
        }
    }

    pub fn append(self: *DiffArchive, timestamp: u64, salience_norm: f32, layer_id: u8, vector: []const f32) void {
        _ = self.appendWithMeta(timestamp, salience_norm, layer_id, false, 0, vector);
    }

    pub fn appendWithMeta(self: *DiffArchive, timestamp: u64, salience_norm: f32, layer_id: u8, is_interrupted: bool, token_id: u32, vector: []const f32) void {
        _ = self.appendFullMeta(.{
            .timestamp = timestamp,
            .last_accessed = timestamp,
            .salience_norm = salience_norm,
            .layer_id = layer_id,
            .is_interrupted = is_interrupted,
            .token_id = token_id,
        }, vector);
    }

    pub fn appendFullMeta(self: *DiffArchive, meta: MemoryMeta, vector: []const f32) usize {
        std.debug.assert(vector.len == self.dim);
        var sum_sq: f32 = 0.0;
        for (vector) |v| sum_sq += v * v;
        const inv = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;
        const idx = self.write_head;
        const row = self.vectors[idx * self.dim .. (idx + 1) * self.dim];
        for (row, vector) |*dst, v| dst.* = v * inv;

        self.metas[idx] = meta;
        self.write_head = (idx + 1) % self.capacity;
        if (self.count < self.capacity) self.count += 1;
        return idx;
    }

    pub fn scan(self: *DiffArchive, query: []const f32, now: u64, out_indices: []usize, top_k: usize) usize {
        const n = self.count;
        if (n == 0) return 0;
        for (0..n) |i| {
            self.scan_scores[i] = self.score(query, now, i);
            self.scan_indices[i] = i;
        }

        const k = @min(top_k, @min(n, out_indices.len));
        for (0..k) |slot| {
            var best = slot;
            for (slot + 1..n) |j| {
                if (self.scan_scores[j] > self.scan_scores[best]) best = j;
            }
            const tmp_score = self.scan_scores[slot];
            self.scan_scores[slot] = self.scan_scores[best];
            self.scan_scores[best] = tmp_score;
            const tmp_idx = self.scan_indices[slot];
            self.scan_indices[slot] = self.scan_indices[best];
            self.scan_indices[best] = tmp_idx;
            out_indices[slot] = self.scan_indices[slot];
        }

        for (out_indices[0..k]) |idx| {
            self.metas[idx].last_accessed = now;
            self.metas[idx].access_count +%= 1;
        }
        return k;
    }

    pub fn primeTier3(
        self: *DiffArchive,
        query: []const f32,
        now: u64,
        ring: *DynamicRingBuffer,
        scratch_recall_indices: []usize,
        current_clock: usize,
        theta: f32,
        head_dim: usize,
        rotary_dim: usize,
        num_kv_heads: usize,
    ) usize {
        const max_recall = ring_buffer.UPPER_RECALL_SLOTS;
        const to_fetch = @min(max_recall, scratch_recall_indices.len);
        if (to_fetch == 0 or self.count == 0) {
            ring.clearRecall();
            return 0;
        }

        const selected = self.scan(query, now, scratch_recall_indices[0..to_fetch], to_fetch);
        ring.clearRecall();

        for (0..selected) |rank| {
            const mi = scratch_recall_indices[rank];
            const mem_clock = self.metas[mi].start_clock;
            self.copyKVToRing(mi, ring, rank, @intCast(mem_clock), current_clock, theta, head_dim, rotary_dim, num_kv_heads);
        }
        return selected;
    }

    fn score(self: *const DiffArchive, query: []const f32, now: u64, i: usize) f32 {
        const meta = self.metas[i];
        const row = self.vectors[i * self.dim .. (i + 1) * self.dim];
        var dot: f32 = 0.0;
        for (query, row) |q, r| dot += q * r;
        const dt = now -| meta.last_accessed;
        const recency = @exp(-self.config.lambda * @as(f32, @floatFromInt(dt)));
        const prominence = @log(1.0 + @as(f32, @floatFromInt(meta.child_count + meta.access_count)));
        return self.config.alpha * dot + self.config.beta * recency + self.config.gamma * prominence + self.config.delta * meta.salience_norm;
    }
};

test "archive associative recall ranks cosine match above recency" {
    var arch = try DiffArchive.init(std.testing.allocator, 4, 8, .{});
    defer arch.deinit();
    arch.append(0, 0.5, 0, &[_]f32{ 1, 0, 0, 0 });
    arch.append(1, 0.5, 0, &[_]f32{ 0, 1, 0, 0 });
    var idx: [4]usize = undefined;
    const k = arch.scan(&[_]f32{ 1, 0, 0, 0 }, 2, &idx, 2);
    try std.testing.expectEqual(@as(usize, 2), k);
    try std.testing.expectEqual(@as(usize, 0), idx[0]);
}

test "archive wraps circularly and caps count at capacity" {
    var arch = try DiffArchive.init(std.testing.allocator, 2, 3, .{});
    defer arch.deinit();
    arch.append(0, 0, 0, &[_]f32{ 1, 0 });
    arch.append(1, 0, 0, &[_]f32{ 0, 1 });
    arch.append(2, 0, 0, &[_]f32{ 1, 1 });
    arch.append(3, 0, 0, &[_]f32{ -1, 0 });
    try std.testing.expectEqual(@as(usize, 3), arch.count);
    try std.testing.expectEqual(@as(usize, 1), arch.write_head);
}

test "archive multi-factor salience weights child count and prominence" {
    var arch = try DiffArchive.init(std.testing.allocator, 4, 8, .{});
    defer arch.deinit();

    // Node 0: weak match but high child count / foundational schema
    _ = arch.appendFullMeta(.{
        .episode_id = 1,
        .timestamp = 1000,
        .last_accessed = 1000,
        .salience_norm = 1.0,
        .child_count = 50,
        .access_count = 20,
    }, &[_]f32{ 0.7, 0.7, 0.0, 0.0 });

    // Node 1: zero children, ephemeral
    _ = arch.appendFullMeta(.{
        .episode_id = 2,
        .timestamp = 1000,
        .last_accessed = 1000,
        .salience_norm = 0.1,
        .child_count = 0,
        .access_count = 0,
    }, &[_]f32{ 0.7, 0.7, 0.0, 0.0 });

    var idx: [2]usize = undefined;
    const k = arch.scan(&[_]f32{ 0.7, 0.7, 0.0, 0.0 }, 1000, &idx, 2);
    try std.testing.expectEqual(@as(usize, 2), k);
    try std.testing.expectEqual(@as(usize, 0), idx[0]); // Node 0 ranked higher due to prominence
}
