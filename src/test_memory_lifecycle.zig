const std = @import("std");
const memory = @import("memory.zig");
const storage = @import("storage.zig");
const hippocampus = @import("hippocampus.zig");
const ring_buffer = @import("ring_buffer.zig");

test "end-to-end multi-turn episodic lifecycle, provenance DAG, cold restart, and latent rehydration" {
    const allocator = std.testing.allocator;
    const test_path = "/tmp/test_full_lifecycle.mem";
    std.fs.cwd().deleteFile(test_path) catch {};
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const H: usize = 16;
    const max_episodes: usize = 32;
    const max_tokens_per_ep: usize = 8;
    const num_layers: usize = 4;
    const kv_dim: usize = 4;

    // === PHASE 1: Conversation Turn 1 (Foundational Schema) ===
    {
        var store = try storage.PersistentDiffStore.open(test_path, max_episodes, num_layers, H, kv_dim, max_tokens_per_ep);
        defer store.close();
        var arch = try memory.DiffArchive.init(allocator, H, max_episodes, .{});
        defer arch.deinit();
        var ring = try ring_buffer.DynamicRingBuffer.init(allocator, num_layers, kv_dim, 4, 16, 4);
        defer ring.deinit();
        var hippo = try hippocampus.Hippocampus.init(allocator, H, max_tokens_per_ep, 6000, num_layers, kv_dim);
        defer hippo.deinit();

        // Populate ring buffer with Turn 1 KV activations (tokens 0..3)
        for (0..4) |t| {
            for (0..num_layers) |l| {
                const slot = ring.activateSlot(l, t);
                const r_slot = l * ring.total_slots + slot;
                const r_off = r_slot * ring.max_kv_dim;
                for (0..kv_dim) |d| {
                    ring.k[r_off + d] = @floatFromInt(100 + l * 10 + t * 2 + d);
                    ring.v[r_off + d] = @floatFromInt(200 + l * 10 + t * 2 + d);
                }
            }
            // Stage hidden vector for each token
            var h_vec: [16]f32 = undefined;
            for (&h_vec, 0..) |*v, d| v.* = if (d < 4) 1.0 else 0.0;
            hippo.stage(&h_vec, @intCast(1000 + t * 50), 0.9, 3, @intCast(10 + t), t, @intCast(1000 + t * 50));
        }

        // Commit Turn 1 (Episode ID 1, Parent ID 0)
        hippo.setCurrentParent(0);
        const flushed1 = hippo.commit(&arch, &ring, &store, 0, null);
        try std.testing.expectEqual(@as(usize, 4), flushed1);
        try std.testing.expectEqual(@as(u64, 1), store.getHeader().total_episodes);

        const ep1 = store.getEpisodeHeader(0);
        try std.testing.expectEqual(@as(u64, 1), ep1.episode_id);
        try std.testing.expectEqual(@as(u64, 0), ep1.parent_episode_id);
        try std.testing.expectEqual(@as(u32, 4), ep1.token_count);
        try std.testing.expectEqual(@as(u32, 13), ep1.continuation_token);

        // === PHASE 2: Conversation Turn 2 (Branching Child Reasoning) ===
        // Populate Turn 2 KV activations (tokens 4..7)
        for (4..8) |t| {
            for (0..num_layers) |l| {
                const slot = ring.activateSlot(l, t);
                const r_slot = l * ring.total_slots + slot;
                const r_off = r_slot * ring.max_kv_dim;
                for (0..kv_dim) |d| {
                    ring.k[r_off + d] = @floatFromInt(300 + l * 10 + (t - 4) * 2 + d);
                    ring.v[r_off + d] = @floatFromInt(400 + l * 10 + (t - 4) * 2 + d);
                }
            }
            var h_vec: [16]f32 = undefined;
            for (&h_vec, 0..) |*v, d| v.* = if (d >= 4 and d < 8) 1.0 else 0.0;
            hippo.stage(&h_vec, @intCast(2000 + (t - 4) * 50), 0.85, 3, @intCast(20 + (t - 4)), t, @intCast(2000 + (t - 4) * 50));
        }

        // Turn 2 is a child of Turn 1 (Parent ID 1)
        hippo.setCurrentParent(1);
        const flushed2 = hippo.commit(&arch, &ring, &store, 4, null);
        try std.testing.expectEqual(@as(usize, 4), flushed2);
        try std.testing.expectEqual(@as(u64, 2), store.getHeader().total_episodes);

        const ep2 = store.getEpisodeHeader(1);
        try std.testing.expectEqual(@as(u64, 2), ep2.episode_id);
        try std.testing.expectEqual(@as(u64, 1), ep2.parent_episode_id);

        // Verify that parent episode 1 had its child_count incremented to 1
        const ep1_updated = store.getEpisodeHeader(0);
        try std.testing.expectEqual(@as(u32, 1), ep1_updated.child_count);

        // === PHASE 3: Conversation Turn 3 (Interrupted Trajectory) ===
        // Stage 2 tokens before an abort signal
        for (8..10) |t| {
            for (0..num_layers) |l| {
                const slot = ring.activateSlot(l, t);
                const r_slot = l * ring.total_slots + slot;
                const r_off = r_slot * ring.max_kv_dim;
                for (0..kv_dim) |d| {
                    ring.k[r_off + d] = @floatFromInt(500 + l * 10 + (t - 8) * 2 + d);
                    ring.v[r_off + d] = @floatFromInt(600 + l * 10 + (t - 8) * 2 + d);
                }
            }
            var h_vec: [16]f32 = undefined;
            for (&h_vec, 0..) |*v, d| v.* = if (d >= 8 and d < 12) 1.0 else 0.0;
            hippo.stage(&h_vec, @intCast(3000 + (t - 8) * 50), 0.7, 3, @intCast(30 + (t - 8)), t, @intCast(3000 + (t - 8) * 50));
        }
        hippo.setCurrentParent(2);
        hippo.markInterrupted();
        const flushed3 = hippo.commit(&arch, &ring, &store, 8, null);
        try std.testing.expectEqual(@as(usize, 2), flushed3);
        try std.testing.expectEqual(@as(u64, 3), store.getHeader().total_episodes);

        const ep3 = store.getEpisodeHeader(2);
        try std.testing.expectEqual(@as(u64, 3), ep3.episode_id);
        try std.testing.expectEqual(@as(u64, 2), ep3.parent_episode_id);
        try std.testing.expect(ep3.flags & storage.EpisodeFlags.IS_INTERRUPTED != 0);

        // Verify parent episode 2 child_count incremented to 1
        const ep2_updated = store.getEpisodeHeader(1);
        try std.testing.expectEqual(@as(u32, 1), ep2_updated.child_count);
    }

    // === PHASE 4: Cold Engine Restart & Resident Cache Rehydration ===
    {
        // Re-open store from scratch
        var store = try storage.PersistentDiffStore.open(test_path, max_episodes, num_layers, H, kv_dim, max_tokens_per_ep);
        defer store.close();

        try std.testing.expectEqual(@as(u64, 3), store.getHeader().total_episodes);

        // Load resident memory archive
        var arch = try memory.DiffArchive.init(allocator, H, max_episodes, .{});
        defer arch.deinit();

        const loaded = store.loadIntoArchive(&arch);
        try std.testing.expectEqual(@as(usize, 3), loaded);

        // Verify provenance and metadata in archive
        try std.testing.expectEqual(@as(u64, 1), arch.metas[0].episode_id);
        try std.testing.expectEqual(@as(u32, 1), arch.metas[0].child_count);
        try std.testing.expectEqual(@as(u64, 2), arch.metas[1].episode_id);
        try std.testing.expectEqual(@as(u32, 1), arch.metas[1].child_count);
        try std.testing.expectEqual(@as(u64, 3), arch.metas[2].episode_id);
        try std.testing.expect(arch.metas[2].is_interrupted);

        // Scan memory for query matching Turn 1
        var q_vec: [16]f32 = undefined;
        for (&q_vec, 0..) |*v, d| v.* = if (d < 4) 1.0 else 0.0;

        var scan_indices: [4]usize = undefined;
        const matched = arch.scan(&q_vec, 4000, &scan_indices, 2);
        try std.testing.expectEqual(@as(usize, 2), matched);
        try std.testing.expectEqual(@as(usize, 0), scan_indices[0]); // Episode 1 ranked #1

        // === PHASE 5: Explicit Latent Rehydration of KV Slabs ===
        var fresh_ring = try ring_buffer.DynamicRingBuffer.init(allocator, num_layers, kv_dim, 4, 16, 4);
        defer fresh_ring.deinit();

        // Rehydrate Episode 1 (tokens 0..3) into ring buffer starting at virtual clock 100
        var temp_kv_buf: [4 * 2 * 8 * 4]f32 = undefined;
        const rehydrated = store.rehydrateEpisode(0, &fresh_ring, 100, &temp_kv_buf);
        try std.testing.expectEqual(@as(usize, 4), rehydrated);

        // Verify that exact FP16 KV values across all 4 layers were restored into fresh_ring
        for (0..4) |t| {
            const c = 100 + t;
            const slot = fresh_ring.getSlotIndex(c);
            for (0..num_layers) |l| {
                const r_slot = l * fresh_ring.total_slots + slot;
                const r_off = r_slot * fresh_ring.max_kv_dim;
                for (0..kv_dim) |d| {
                    const expected_k: f32 = @floatFromInt(100 + l * 10 + t * 2 + d);
                    const expected_v: f32 = @floatFromInt(200 + l * 10 + t * 2 + d);
                    try std.testing.expectApproxEqAbs(expected_k, fresh_ring.k[r_off + d], 0.05);
                    try std.testing.expectApproxEqAbs(expected_v, fresh_ring.v[r_off + d], 0.05);
                }
            }
        }
    }
}
