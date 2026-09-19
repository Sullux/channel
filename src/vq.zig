const std = @import("std");
pub const memory = @import("memory.zig");

pub const Centroid = struct {
    count: u32 = 0,
    first_seen: u64 = 0,
    last_seen: u64 = 0,
    salience_norm: f32 = 0.0,
    active: bool = false,
};

/// Bounded Spherical Vector Quantization (VQ) Codebook for clustering
/// historical diff archives into compact semantic landmark centroids.
pub const CentroidCodebook = struct {
    allocator: std.mem.Allocator,
    num_centroids: usize,
    dim: usize,
    active_count: usize = 0,
    centroids: []Centroid,
    vectors: []f32, // num_centroids * dim, unit-normalized

    pub fn init(allocator: std.mem.Allocator, num_centroids: usize, dim: usize) !CentroidCodebook {
        const centroids = try allocator.alloc(Centroid, num_centroids);
        errdefer allocator.free(centroids);
        const vectors = try allocator.alloc(f32, num_centroids * dim);
        errdefer allocator.free(vectors);
        @memset(centroids, Centroid{});
        @memset(vectors, 0);

        return .{ .allocator = allocator, .num_centroids = num_centroids, .dim = dim, .centroids = centroids, .vectors = vectors };
    }

    pub fn deinit(self: *CentroidCodebook) void {
        self.allocator.free(self.centroids);
        self.allocator.free(self.vectors);
    }

    pub fn reset(self: *CentroidCodebook) void {
        @memset(self.centroids, Centroid{});
        @memset(self.vectors, 0);
        self.active_count = 0;
    }

    pub fn getVector(self: *const CentroidCodebook, idx: usize) []const f32 {
        return self.vectors[idx * self.dim .. (idx + 1) * self.dim];
    }

    /// Online spherical update: merges vector into nearest centroid or allocates new slot.
    pub fn updateOnline(self: *CentroidCodebook, timestamp: u64, salience_norm: f32, vector: []const f32, sim_thresh: f32) usize {
        std.debug.assert(vector.len == self.dim);
        if (self.active_count > 0) {
            const match = self.findNearest(vector);
            if (match.similarity >= sim_thresh) {
                const idx = match.index;
                const c_vec = self.vectors[idx * self.dim .. (idx + 1) * self.dim];
                const count_f = @as(f32, @floatFromInt(self.centroids[idx].count));
                var sum_sq: f32 = 0.0;
                for (c_vec, vector) |*c, v| {
                    c.* = c.* * count_f + v;
                    sum_sq += c.* * c.*;
                }
                const inv = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;
                for (c_vec) |*c| c.* *= inv;
                self.centroids[idx].count += 1;
                self.centroids[idx].last_seen = timestamp;
                self.centroids[idx].salience_norm = @max(self.centroids[idx].salience_norm, salience_norm);
                return idx;
            }
        }

        const slot = if (self.active_count < self.num_centroids) self.active_count else self.findLeastSalient();
        if (slot == self.active_count) self.active_count += 1;
        const c_vec = self.vectors[slot * self.dim .. (slot + 1) * self.dim];
        var sum_sq: f32 = 0.0;
        for (vector) |v| sum_sq += v * v;
        const inv = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;
        for (c_vec, vector) |*dst, v| dst.* = v * inv;

        self.centroids[slot] = .{
            .count = 1,
            .first_seen = timestamp,
            .last_seen = timestamp,
            .salience_norm = salience_norm,
            .active = true,
        };
        return slot;
    }

    /// Compress entire DiffArchive into K centroids using Spherical K-Means.
    pub fn compressArchive(self: *CentroidCodebook, archive: *const memory.DiffArchive, iterations: usize) void {
        const n = archive.count;
        if (n == 0) return;
        const k = @min(self.num_centroids, n);
        self.reset();
        self.active_count = k;

        const stride = n / k;
        for (0..k) |c| {
            const src_idx = c * stride;
            const src = archive.vectors[src_idx * self.dim .. (src_idx + 1) * self.dim];
            @memcpy(self.vectors[c * self.dim .. (c + 1) * self.dim], src);
            self.centroids[c] = .{
                .first_seen = archive.metas[src_idx].timestamp,
                .last_seen = archive.metas[src_idx].timestamp,
                .salience_norm = archive.metas[src_idx].salience_norm,
                .active = true,
            };
        }

        for (0..iterations) |_| {
            for (self.centroids[0..k]) |*c| c.count = 0;
            for (0..n) |i| {
                const vec = archive.vectors[i * self.dim .. (i + 1) * self.dim];
                const c_idx = self.findNearest(vec).index;
                const c_vec = self.vectors[c_idx * self.dim .. (c_idx + 1) * self.dim];
                const meta = archive.metas[i];
                if (self.centroids[c_idx].count == 0) {
                    @memcpy(c_vec, vec);
                    self.centroids[c_idx].first_seen = meta.timestamp;
                    self.centroids[c_idx].last_seen = meta.timestamp;
                    self.centroids[c_idx].salience_norm = meta.salience_norm;
                } else {
                    for (c_vec, vec) |*c_val, v| c_val.* += v;
                    self.centroids[c_idx].last_seen = @max(self.centroids[c_idx].last_seen, meta.timestamp);
                    self.centroids[c_idx].salience_norm = @max(self.centroids[c_idx].salience_norm, meta.salience_norm);
                }
                self.centroids[c_idx].count += 1;
            }

            for (0..k) |c| {
                const c_vec = self.vectors[c * self.dim .. (c + 1) * self.dim];
                var sum_sq: f32 = 0.0;
                for (c_vec) |v| sum_sq += v * v;
                const inv = if (sum_sq > 1e-12) 1.0 / @sqrt(sum_sq) else 0.0;
                for (c_vec) |*v| v.* *= inv;
            }
        }
    }

    pub fn findNearest(self: *const CentroidCodebook, query: []const f32) struct { index: usize, similarity: f32 } {
        var best_idx: usize = 0;
        var best_sim: f32 = -std.math.inf(f32);
        for (0..self.active_count) |i| {
            const c_vec = self.vectors[i * self.dim .. (i + 1) * self.dim];
            var dot: f32 = 0.0;
            for (query, c_vec) |q, c| dot += q * c;
            if (dot > best_sim) {
                best_sim = dot;
                best_idx = i;
            }
        }
        return .{ .index = best_idx, .similarity = best_sim };
    }

    fn findLeastSalient(self: *const CentroidCodebook) usize {
        var min_idx: usize = 0;
        var min_score: f32 = std.math.inf(f32);
        for (self.centroids[0..self.active_count], 0..) |c, i| {
            const score = c.salience_norm * @as(f32, @floatFromInt(c.count));
            if (score < min_score) {
                min_score = score;
                min_idx = i;
            }
        }
        return min_idx;
    }
};

test "vq codebook online clustering merges similar vectors" {
    var cb = try CentroidCodebook.init(std.testing.allocator, 4, 3);
    defer cb.deinit();
    const idx1 = cb.updateOnline(10, 0.8, &[_]f32{ 1.0, 0.01, 0.0 }, 0.90);
    const idx2 = cb.updateOnline(11, 0.9, &[_]f32{ 0.99, 0.02, 0.0 }, 0.90);
    const idx3 = cb.updateOnline(12, 0.7, &[_]f32{ 0.0, 1.0, 0.0 }, 0.90);
    try std.testing.expectEqual(idx1, idx2);
    try std.testing.expect(idx1 != idx3);
    try std.testing.expectEqual(@as(usize, 2), cb.active_count);
    try std.testing.expectEqual(@as(u32, 2), cb.centroids[idx1].count);
}

test "vq codebook compressArchive clusters into K centroids" {
    var arch = try memory.DiffArchive.init(std.testing.allocator, 2, 8, .{});
    defer arch.deinit();
    arch.append(1, 0.5, 0, &[_]f32{ 1.0, 0.0 });
    arch.append(2, 0.5, 0, &[_]f32{ 0.9, 0.1 });
    arch.append(3, 0.5, 0, &[_]f32{ 0.0, 1.0 });
    arch.append(4, 0.5, 0, &[_]f32{ 0.1, 0.9 });

    var cb = try CentroidCodebook.init(std.testing.allocator, 2, 2);
    defer cb.deinit();
    cb.compressArchive(&arch, 5);
    try std.testing.expectEqual(@as(usize, 2), cb.active_count);
    const nearest_x = cb.findNearest(&[_]f32{ 1.0, 0.0 });
    try std.testing.expect(nearest_x.similarity > 0.95);
}
