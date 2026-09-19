const std = @import("std");
const memory = @import("memory.zig");
const DiffArchive = memory.DiffArchive;
const DynamicRingBuffer = @import("ring_buffer.zig").DynamicRingBuffer;
const storage_mod = @import("storage.zig");
const PersistentDiffStore = storage_mod.PersistentDiffStore;
const EpisodeHeader = storage_mod.EpisodeHeader;
const EpisodeFlags = storage_mod.EpisodeFlags;
pub const gpu = @import("gpu.zig");

pub const StagedItem = struct {
    timestamp: u64,
    salience_norm: f32,
    layer_id: u8,
    is_interrupted: bool,
    token_id: u32,
    slot_idx: usize,
    vector_offset: usize,
};

pub const Hippocampus = struct {
    allocator: std.mem.Allocator,
    dim: usize,
    capacity: usize,
    count: usize,
    debounce_ms: i64,
    last_activity_ms: i64,
    current_parent_id: u64 = 0,

    vectors: []f32,
    items: []StagedItem,
    centroid_buf: []f32,
    kv_staging_buf: []f32,

    pub fn init(allocator: std.mem.Allocator, dim: usize, capacity: usize, debounce_ms: i64, num_layers: usize, kv_dim: usize) !Hippocampus {
        const vectors = try allocator.alloc(f32, capacity * dim);
        errdefer allocator.free(vectors);
        const items = try allocator.alloc(StagedItem, capacity);
        errdefer allocator.free(items);
        const centroid_buf = try allocator.alloc(f32, dim);
        errdefer allocator.free(centroid_buf);
        const kv_len = num_layers * 2 * capacity * kv_dim;
        const kv_staging_buf = try allocator.alloc(f32, kv_len);
        errdefer allocator.free(kv_staging_buf);

        return .{
            .allocator = allocator,
            .dim = dim,
            .capacity = capacity,
            .count = 0,
            .debounce_ms = debounce_ms,
            .last_activity_ms = 0,
            .current_parent_id = 0,
            .vectors = vectors,
            .items = items,
            .centroid_buf = centroid_buf,
            .kv_staging_buf = kv_staging_buf,
        };
    }

    pub fn deinit(self: *Hippocampus) void {
        self.allocator.free(self.vectors);
        self.allocator.free(self.items);
        self.allocator.free(self.centroid_buf);
        self.allocator.free(self.kv_staging_buf);
    }

    pub fn stage(self: *Hippocampus, vector: []const f32, timestamp: u64, salience: f32, layer: u8, token_id: u32, slot: usize, now_ms: i64) void {
        if (self.count >= self.capacity) return;
        const idx = self.count;
        const v_off = idx * self.dim;
        @memcpy(self.vectors[v_off .. v_off + self.dim], vector);
        self.items[idx] = .{
            .timestamp = timestamp,
            .salience_norm = salience,
            .layer_id = layer,
            .is_interrupted = false,
            .token_id = token_id,
            .slot_idx = slot,
            .vector_offset = v_off,
        };
        self.count += 1;
        self.last_activity_ms = now_ms;
    }

    pub fn setCurrentParent(self: *Hippocampus, parent_id: u64) void {
        self.current_parent_id = parent_id;
    }

    pub fn markInterrupted(self: *Hippocampus) void {
        for (self.items[0..self.count]) |*item| {
            item.is_interrupted = true;
        }
    }

    pub fn shouldFlush(self: *const Hippocampus, now_ms: i64, force: bool) bool {
        if (self.count == 0) return false;
        if (force or self.count >= self.capacity) return true;
        return (now_ms - self.last_activity_ms >= self.debounce_ms);
    }

    pub fn commit(
        self: *Hippocampus,
        archive: ?*DiffArchive,
        ring: ?*const DynamicRingBuffer,
        storage: ?*PersistentDiffStore,
        start_clock: u64,
        gpu_opt: ?*gpu.model_gpu.GpuModelContext,
    ) usize {
        if (self.count == 0) return 0;
        const n = self.count;

        @memset(self.centroid_buf, 0);
        var avg_salience: f32 = 0.0;
        var has_interrupted = false;
        var last_tok: u32 = 0;
        const first_ts = self.items[0].timestamp;
        const last_ts = self.items[n - 1].timestamp;

        for (self.items[0..n]) |item| {
            const row = self.vectors[item.vector_offset .. item.vector_offset + self.dim];
            for (self.centroid_buf, row) |*c, v| c.* += v;
            avg_salience += item.salience_norm;
            if (item.is_interrupted) has_interrupted = true;
            last_tok = item.token_id;
        }
        avg_salience /= @as(f32, @floatFromInt(n));

        var sum_sq: f32 = 0.0;
        for (self.centroid_buf) |v| sum_sq += v * v;
        const inv_norm = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;
        for (self.centroid_buf) |*v| v.* *= inv_norm;

        var flags: u16 = 0;
        if (has_interrupted) flags |= EpisodeFlags.IS_INTERRUPTED;

        if (archive) |a| {
            const a_idx = a.appendFullMeta(.{
                .episode_id = if (storage) |s| @intCast(s.getHeader().total_episodes + 1) else @intCast(a.count + 1),
                .parent_episode_id = self.current_parent_id,
                .timestamp = first_ts,
                .last_accessed = last_ts,
                .start_clock = start_clock,
                .token_count = @intCast(n),
                .salience_norm = avg_salience,
                .is_interrupted = has_interrupted,
                .token_id = last_tok,
            }, self.centroid_buf);
            if (ring) |r| {
                const rep_slot = self.items[n / 2].slot_idx;
                a.copyKVFromRing(a_idx, r, rep_slot, gpu_opt);
            }
        }

        if (storage) |s| {
            var kv_slice: ?[]const f32 = null;
            if (ring) |r| {
                const num_layers = s.num_layers;
                const kv_dim = s.kv_dim;
                for (0..num_layers) |l| {
                    const l_base = l * 2 * n * kv_dim;
                    for (0..n) |t| {
                        const s_idx = self.items[t].slot_idx;
                        const k_dst = l_base + 0 * (n * kv_dim) + t * kv_dim;
                        const v_dst = l_base + 1 * (n * kv_dim) + t * kv_dim;
                        if (gpu_opt) |g| {
                            if (l < g.layers.len) {
                                const g_k = g.layers[l].buf_k_cache.asSlice(f32);
                                const g_v = g.layers[l].buf_v_cache.asSlice(f32);
                                const g_off = s_idx * r.max_kv_dim;
                                if (g_off + kv_dim <= g_k.len and g_off + kv_dim <= g_v.len) {
                                    @memcpy(self.kv_staging_buf[k_dst .. k_dst + kv_dim], g_k[g_off .. g_off + kv_dim]);
                                    @memcpy(self.kv_staging_buf[v_dst .. v_dst + kv_dim], g_v[g_off .. g_off + kv_dim]);
                                    continue;
                                }
                            }
                        }
                        const r_slot = l * r.total_slots + s_idx;
                        const r_off = r_slot * r.max_kv_dim;
                        @memcpy(self.kv_staging_buf[k_dst .. k_dst + kv_dim], r.k[r_off .. r_off + kv_dim]);
                        @memcpy(self.kv_staging_buf[v_dst .. v_dst + kv_dim], r.v[r_off .. r_off + kv_dim]);
                    }
                }
                kv_slice = self.kv_staging_buf[0 .. num_layers * 2 * n * kv_dim];
            }

            const ep = EpisodeHeader{
                .episode_id = @intCast(s.getHeader().total_episodes + 1),
                .parent_episode_id = self.current_parent_id,
                .created_timestamp = first_ts,
                .last_accessed = last_ts,
                .start_clock = start_clock,
                .token_count = @intCast(n),
                .access_count = 0,
                .child_count = 0,
                .salience_norm = avg_salience,
                .continuation_token = last_tok,
                .flags = flags,
                .summary_len = 0,
            };
            s.appendEpisode(ep, self.centroid_buf, "", kv_slice);

            if (self.current_parent_id != 0) {
                s.incrementChildCount(self.current_parent_id);
            }
        }

        self.count = 0;
        return n;
    }
};

test "hippocampus stages and flushes after debounce" {
    var arch = try DiffArchive.init(std.testing.allocator, 4, 8, .{});
    defer arch.deinit();
    var hippo = try Hippocampus.init(std.testing.allocator, 4, 16, 6000, 2, 2);
    defer hippo.deinit();

    hippo.stage(&[_]f32{ 1, 0, 0, 0 }, 100, 0.8, 0, 42, 0, 1000);
    try std.testing.expectEqual(@as(usize, 1), hippo.count);
    try std.testing.expect(!hippo.shouldFlush(3000, false));
    try std.testing.expect(hippo.shouldFlush(7500, false));

    const flushed = hippo.commit(&arch, null, null, 0, null);
    try std.testing.expectEqual(@as(usize, 1), flushed);
    try std.testing.expectEqual(@as(usize, 0), hippo.count);
    try std.testing.expectEqual(@as(usize, 1), arch.count);
}

test "hippocampus consolidates KV slab and lineage to persistent storage" {
    const test_path = "/tmp/test_hippo_store.mem";
    std.fs.cwd().deleteFile(test_path) catch {};
    defer std.fs.cwd().deleteFile(test_path) catch {};

    var arch = try DiffArchive.init(std.testing.allocator, 4, 8, .{});
    defer arch.deinit();
    var store = try PersistentDiffStore.open(test_path, 4, 2, 4, 2, 4);
    defer store.close();
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 2, 2, 4, 16, 4);
    defer ring.deinit();

    // Populate ring buffer slots
    const s0 = ring.activateSlot(0, 0);
    const s1 = ring.activateSlot(0, 1);
    _ = ring.activateSlot(1, 0);
    _ = ring.activateSlot(1, 1);
    ring.k[0..4].* = [_]f32{ 1.1, 1.2, 2.1, 2.2 };
    ring.v[0..4].* = [_]f32{ 3.1, 3.2, 4.1, 4.2 };

    var hippo = try Hippocampus.init(std.testing.allocator, 4, 16, 6000, 2, 2);
    defer hippo.deinit();

    // Stage 2 tokens
    hippo.stage(&[_]f32{ 1.0, 0.0, 0.0, 0.0 }, 1000, 0.8, 0, 10, s0, 1000);
    hippo.stage(&[_]f32{ 0.0, 1.0, 0.0, 0.0 }, 1050, 0.9, 0, 11, s1, 1050);

    const committed = hippo.commit(&arch, &ring, &store, 0, null);
    try std.testing.expectEqual(@as(usize, 2), committed);
    try std.testing.expectEqual(@as(u64, 1), store.getHeader().total_episodes);

    const ep = store.getEpisodeHeader(0);
    try std.testing.expectEqual(@as(u32, 2), ep.token_count);
    try std.testing.expectEqual(@as(u32, 11), ep.continuation_token);

    var kv_out: [16]f32 = undefined;
    const kv_read = store.getEpisodeKVSlab(0, &kv_out);
    try std.testing.expectEqual(@as(usize, 16), kv_read);
}
