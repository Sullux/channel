const std = @import("std");

pub const TOTAL_SLOTS: usize = 4096;
pub const UPPER_RECALL_SLOTS: usize = 128;
pub const SPLIT_LAYER: usize = 16;

pub const SpanType = enum(u8) {
    system = 0,
    user = 1,
    thought = 2,
    tool_call = 3,
    tool_result = 4,
    response_sentence = 5,
    turn_end = 6,
};

/// Fixed 4,096-slot per-layer buffer with uniform 3-tier partitioning across all layers:
///   Tier 1: [0 .. num_anchors)                           Immutable Dynamic Anchors
///   Tier 2: [num_anchors .. recallStart(l))             Sliding FIFO Ring Window
///   Tier 3: [recallStart(l) .. TOTAL_SLOTS)              Associative Recall
pub const DynamicRingBuffer = struct {
    pub const MAX_BOUNDARIES: usize = 1024;

    allocator: std.mem.Allocator,
    num_layers: usize,
    max_kv_dim: usize,
    num_anchors: usize,
    total_slots: usize,

    k: []f32,
    v: []f32,
    clocks: []usize,
    active: []bool,
    attention_mass: []f32,
    total_ingested: usize,
    turn_boundaries: [MAX_BOUNDARIES]usize = [_]usize{0} ** MAX_BOUNDARIES,
    boundary_types: [MAX_BOUNDARIES]SpanType = [_]SpanType{.turn_end} ** MAX_BOUNDARIES,
    boundary_salience: [MAX_BOUNDARIES]f32 = [_]f32{1.0} ** MAX_BOUNDARIES,
    num_turn_boundaries: usize = 0,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, max_kv_dim: usize, num_anchors: usize, window_size: usize, num_recall: usize) !DynamicRingBuffer {
        _ = window_size; _ = num_recall;
        const slots = TOTAL_SLOTS;
        const total_elements = num_layers * slots * max_kv_dim;

        const k_buf = try allocator.alloc(f32, total_elements);
        const v_buf = try allocator.alloc(f32, total_elements);
        const clocks_buf = try allocator.alloc(usize, num_layers * slots);
        const active_buf = try allocator.alloc(bool, num_layers * slots);
        const attn_buf = try allocator.alloc(f32, num_layers * slots);

        @memset(k_buf, 0); @memset(v_buf, 0); @memset(clocks_buf, 0); @memset(active_buf, false);
        @memset(attn_buf, 0);

        return .{
            .allocator = allocator, .num_layers = num_layers, .max_kv_dim = max_kv_dim,
            .num_anchors = num_anchors, .total_slots = slots, .k = k_buf, .v = v_buf,
            .clocks = clocks_buf, .active = active_buf, .attention_mass = attn_buf, .total_ingested = 0,
        };
    }

    pub fn deinit(self: *DynamicRingBuffer) void {
        self.allocator.free(self.k); self.allocator.free(self.v);
        self.allocator.free(self.clocks); self.allocator.free(self.active);
        self.allocator.free(self.attention_mass);
    }
    pub fn reset(self: *DynamicRingBuffer) void {
        @memset(self.k, 0); @memset(self.v, 0);
        @memset(self.clocks, 0); @memset(self.active, false);
        @memset(self.attention_mass, 0);
        self.total_ingested = 0;
        self.num_turn_boundaries = 0;
    }

    pub fn markTurnBoundary(self: *DynamicRingBuffer, clock: usize) void {
        self.markBoundary(clock, .turn_end, 1.0);
    }

    pub fn markBoundary(self: *DynamicRingBuffer, clock: usize, span_type: SpanType, salience: f32) void {
        if (self.num_turn_boundaries > 0 and self.turn_boundaries[(self.num_turn_boundaries - 1) % MAX_BOUNDARIES] == clock) {
            self.boundary_types[(self.num_turn_boundaries - 1) % MAX_BOUNDARIES] = span_type;
            self.boundary_salience[(self.num_turn_boundaries - 1) % MAX_BOUNDARIES] = salience;
            return;
        }
        const idx = self.num_turn_boundaries % MAX_BOUNDARIES;
        self.turn_boundaries[idx] = clock;
        self.boundary_types[idx] = span_type;
        self.boundary_salience[idx] = salience;
        self.num_turn_boundaries += 1;
    }

    pub fn snapToBoundary(self: *const DynamicRingBuffer, raw_min_clock: usize) usize {
        if (raw_min_clock <= self.num_anchors or self.num_turn_boundaries == 0) return raw_min_clock;
        const start = if (self.num_turn_boundaries > MAX_BOUNDARIES) self.num_turn_boundaries - MAX_BOUNDARIES else 0;
        for (start..self.num_turn_boundaries) |i| {
            const b = self.turn_boundaries[i % MAX_BOUNDARIES];
            if (b >= raw_min_clock) return b;
        }
        return raw_min_clock;
    }
    pub fn setNumAnchors(self: *DynamicRingBuffer, n: usize) void {
        self.num_anchors = @min(n, self.total_slots - UPPER_RECALL_SLOTS - 64);
    }
    pub inline fn recallSlots(self: *const DynamicRingBuffer, layer: usize) usize {
        _ = layer;
        return if (self.total_slots < UPPER_RECALL_SLOTS * 2) 0 else UPPER_RECALL_SLOTS;
    }
    pub inline fn recallStart(self: *const DynamicRingBuffer, layer: usize) usize {
        return self.total_slots - self.recallSlots(layer);
    }
    pub inline fn windowSize(self: *const DynamicRingBuffer, layer: usize) usize {
        const start = self.recallStart(layer);
        return if (start > self.num_anchors) start - self.num_anchors else 1;
    }

    pub fn getSlotIndex(self: *const DynamicRingBuffer, clock: usize) usize { return self.getSlotIndexLayer(0, clock); }

    pub fn getSlotIndexLayer(self: *const DynamicRingBuffer, layer: usize, clock: usize) usize {
        if (clock < self.num_anchors) return clock;
        return self.num_anchors + ((clock - self.num_anchors) % self.windowSize(layer));
    }

    pub fn activateSlot(self: *DynamicRingBuffer, layer: usize, clock: usize) usize {
        const slot = self.getSlotIndexLayer(layer, clock);
        const slot_idx = layer * self.total_slots + slot;
        self.clocks[slot_idx] = clock;
        self.active[slot_idx] = true;
        self.attention_mass[slot_idx] = 0.0;
        if (layer == 0 and clock >= self.total_ingested) self.total_ingested = clock + 1;
        return slot;
    }

    pub fn deactivateSlot(self: *DynamicRingBuffer, layer: usize, clock: usize) void {
        const slot = self.getSlotIndexLayer(layer, clock);
        const slot_idx = layer * self.total_slots + slot;
        if (self.clocks[slot_idx] == clock) {
            self.active[slot_idx] = false;
            self.clocks[slot_idx] = 0;
            self.attention_mass[slot_idx] = 0.0;
        }
    }

    pub fn rollbackClock(self: *DynamicRingBuffer, target_clock: usize) void {
        if (target_clock >= self.total_ingested) return;
        for (target_clock..self.total_ingested) |c| {
            for (0..self.num_layers) |l| {
                self.deactivateSlot(l, c);
            }
        }
        self.total_ingested = target_clock;
    }

    pub fn recordAttention(self: *DynamicRingBuffer, layer: usize, slot: usize, mass: f32) void {
        if (layer < self.num_layers and slot < self.total_slots) {
            self.attention_mass[layer * self.total_slots + slot] += mass;
        }
    }

    pub fn getAttentionMass(self: *const DynamicRingBuffer, layer: usize, slot: usize) f32 {
        return if (layer < self.num_layers and slot < self.total_slots) self.attention_mass[layer * self.total_slots + slot] else 0.0;
    }

    pub fn getSlotSalience(self: *const DynamicRingBuffer, slot: usize) f32 {
        if (slot >= self.total_slots) return 0.0;
        var sum: f32 = 0;
        for (0..self.num_layers) |l| sum += self.attention_mass[l * self.total_slots + slot];
        return sum / @as(f32, @floatFromInt(self.num_layers));
    }

    pub fn getWorkingSetGini(self: *const DynamicRingBuffer) f32 {
        var saliences: [4096]f32 = undefined;
        var count: usize = 0;
        var total_sum: f32 = 0.0;

        const w_end = self.recallStart(0);
        for (self.num_anchors..w_end) |slot| {
            if (self.active[slot]) {
                const s = self.getSlotSalience(slot);
                saliences[count] = s;
                total_sum += s;
                count += 1;
            }
        }

        if (count < 2 or total_sum <= 0.0) return 1.0;

        std.mem.sort(f32, saliences[0..count], {}, std.sort.asc(f32));

        var weighted_sum: f32 = 0.0;
        for (0..count) |i| {
            weighted_sum += @as(f32, @floatFromInt(i + 1)) * saliences[i];
        }

        const n_f = @as(f32, @floatFromInt(count));
        const gini = (2.0 * weighted_sum) / (n_f * total_sum) - (n_f + 1.0) / n_f;
        return @max(0.0, @min(1.0, gini));
    }

    pub fn isWorkingSetSaturated(self: *const DynamicRingBuffer, capacity_threshold: f32, gini_threshold: f32) bool {
        const w_end = self.recallStart(0);
        const tier2_cap = if (w_end > self.num_anchors) w_end - self.num_anchors else 1;
        var active_count: usize = 0;

        for (self.num_anchors..w_end) |slot| {
            if (self.active[slot]) active_count += 1;
        }

        const fill_ratio = @as(f32, @floatFromInt(active_count)) / @as(f32, @floatFromInt(tier2_cap));
        if (fill_ratio < capacity_threshold) return false;

        const gini = self.getWorkingSetGini();
        return gini < gini_threshold;
    }

    pub fn evictLowSalienceSlots(self: *DynamicRingBuffer, min_clock: usize, max_clock: usize, evict_count: usize) usize {
        if (min_clock >= max_clock or evict_count == 0) return 0;

        const Candidate = struct { slot: usize, salience: f32 };
        var candidates: [1024]Candidate = undefined;
        var cand_count: usize = 0;

        for (min_clock..max_clock + 1) |c| {
            if (c < self.num_anchors) continue;
            const slot = self.getSlotIndexLayer(0, c);
            const g_idx = slot;
            if (self.active[g_idx] and self.clocks[g_idx] == c) {
                if (cand_count < candidates.len) {
                    candidates[cand_count] = .{ .slot = slot, .salience = self.getSlotSalience(slot) };
                    cand_count += 1;
                }
            }
        }

        if (cand_count == 0) return 0;

        std.mem.sort(Candidate, candidates[0..cand_count], {}, struct {
            fn lessThan(_: void, a: Candidate, b: Candidate) bool {
                return a.salience < b.salience;
            }
        }.lessThan);

        const to_evict = @min(evict_count, cand_count);
        for (0..to_evict) |i| {
            const slot = candidates[i].slot;
            for (0..self.num_layers) |l| {
                const idx = l * self.total_slots + slot;
                self.active[idx] = false;
            }
        }

        return to_evict;
    }

    pub fn writeKV(self: *DynamicRingBuffer, layer: usize, clock: usize, k_src: []const f32, v_src: []const f32) void {
        const slot = self.getSlotIndexLayer(layer, clock);
        const slot_idx = layer * self.total_slots + slot;
        const kv_offset = slot_idx * self.max_kv_dim;

        @memcpy(self.k[kv_offset .. kv_offset + k_src.len], k_src);
        @memcpy(self.v[kv_offset .. kv_offset + v_src.len], v_src);
        self.clocks[slot_idx] = clock;
        self.active[slot_idx] = true;
        if (layer == 0 and clock >= self.total_ingested) self.total_ingested = clock + 1;
    }

    pub fn clearRecall(self: *DynamicRingBuffer) void {
        for (0..self.num_layers) |l| {
            const base = l * self.total_slots + self.recallStart(l);
            @memset(self.active[base .. base + self.recallSlots(l)], false);
        }
    }

    pub fn writeRecallKV(self: *DynamicRingBuffer, layer: usize, rank: usize, k_src: []const f32, v_src: []const f32, clock: usize) void {
        const slot = self.recallStart(layer) + rank;
        if (slot >= self.total_slots) return;
        const slot_idx = layer * self.total_slots + slot;
        const kv_offset = slot_idx * self.max_kv_dim;
        @memcpy(self.k[kv_offset .. kv_offset + k_src.len], k_src);
        @memcpy(self.v[kv_offset .. kv_offset + v_src.len], v_src);
        self.clocks[slot_idx] = clock; self.active[slot_idx] = true;
    }

    pub fn getActiveSlots(self: *const DynamicRingBuffer, layer: usize, curr_clock: usize, out_slots: []usize) usize {
        return self.getActiveSlotsLayer(layer, curr_clock, false, 1024, out_slots);
    }

    pub fn getActiveSlotsLayer(self: *const DynamicRingBuffer, layer: usize, curr_clock: usize, is_sliding: bool, sliding_window: usize, out_slots: []usize) usize {
        const layer_offset = layer * self.total_slots;
        var count: usize = 0;

        const effective_window = if (is_sliding) sliding_window else self.windowSize(layer);
        const raw_min: usize = if (curr_clock >= effective_window) curr_clock - effective_window + 1 else 0;
        const min_valid_clock = if (is_sliding) raw_min else self.snapToBoundary(raw_min);

        // 1. Anchors (clock = s)
        const anchor_limit = @min(curr_clock + 1, self.num_anchors);
        for (0..anchor_limit) |s| {
            if (self.active[layer_offset + s]) {
                if (!is_sliding or s >= min_valid_clock) {
                    out_slots[count] = s;
                    count += 1;
                }
            }
        }

        // 2. Dynamic Window in strict chronological order
        const dynamic_start = @max(min_valid_clock, self.num_anchors);
        if (curr_clock >= dynamic_start) {
            for (dynamic_start..curr_clock + 1) |c| {
                const slot = self.getSlotIndexLayer(layer, c);
                const global_idx = layer_offset + slot;
                if (self.active[global_idx] and self.clocks[global_idx] == c) {
                    out_slots[count] = slot;
                    count += 1;
                }
            }
        }

        // 3. Recall slots (Full attention layers only)
        if (!is_sliding) {
            const w_start = self.recallStart(layer);
            for (w_start..self.total_slots) |s| {
                const global_idx = layer_offset + s;
                if (!self.active[global_idx]) continue;
                if (self.clocks[global_idx] <= curr_clock) {
                    out_slots[count] = s;
                    count += 1;
                }
            }
        }

        return count;
    }

    pub fn getPrefillPrevSlots(self: *const DynamicRingBuffer, layer: usize, start_clock: usize, chunk_len: usize, out_slots: []usize) usize {
        const layer_offset = layer * self.total_slots;
        var count: usize = 0;

        if (start_clock == 0 or chunk_len == 0) return 0;

        const end_clock = start_clock + chunk_len - 1;
        const w_size = self.windowSize(layer);
        const raw_min: usize = if (end_clock >= w_size) end_clock - w_size + 1 else 0;
        const min_valid_clock = self.snapToBoundary(raw_min);

        // 1. Anchors (clock = s)
        const anchor_limit = @min(start_clock, self.num_anchors);
        for (0..anchor_limit) |s| {
            if (self.active[layer_offset + s]) {
                out_slots[count] = s;
                count += 1;
            }
        }

        // 2. Dynamic Window in strict chronological order
        const dynamic_start = @max(min_valid_clock, self.num_anchors);
        if (start_clock > dynamic_start) {
            for (dynamic_start..start_clock) |c| {
                const slot = self.getSlotIndexLayer(layer, c);
                const global_idx = layer_offset + slot;
                if (self.active[global_idx] and self.clocks[global_idx] == c) {
                    out_slots[count] = slot;
                    count += 1;
                }
            }
        }

        // 3. Recall slots
        const w_start = self.recallStart(layer);
        for (w_start..self.total_slots) |s| {
            const global_idx = layer_offset + s;
            if (!self.active[global_idx]) continue;
            if (self.clocks[global_idx] < start_clock) {
                out_slots[count] = s;
                count += 1;
            }
        }

        return count;
    }

    pub fn getSlotKV(self: *const DynamicRingBuffer, layer: usize, slot: usize, kv_dim: usize) struct { k: []const f32, v: []const f32, clock: usize } {
        const slot_idx = layer * self.total_slots + slot;
        const offset = slot_idx * self.max_kv_dim;
        return .{
            .k = self.k[offset .. offset + kv_dim],
            .v = self.v[offset .. offset + kv_dim],
            .clock = self.clocks[slot_idx],
        };
    }
};

test "ring buffer tiers and window eviction" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 2, 64, 4, 8, 4);
    defer ring.deinit();

    try std.testing.expectEqual(@as(usize, 4096), ring.total_slots);
    var dummy_k: [64]f32 = undefined;
    var dummy_v: [64]f32 = undefined;
    @memset(&dummy_k, 1.0);
    @memset(&dummy_v, 2.0);

    for (0..20) |c| ring.writeKV(0, c, &dummy_k, &dummy_v);

    var slots_buf: [32]usize = undefined;
    const count = ring.getActiveSlots(0, 19, &slots_buf);
    try std.testing.expectEqual(@as(usize, 20), count);
    try std.testing.expectEqual(@as(usize, 0), slots_buf[0]);
    try std.testing.expectEqual(@as(usize, 3), slots_buf[3]);
}

test "recall slots are written, cleared, and included in active set" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 32, 8, 2, 4, 3);
    defer ring.deinit();

    var k_src: [8]f32 = undefined;
    var v_src: [8]f32 = undefined;
    @memset(&k_src, 1.0);
    @memset(&v_src, 2.0);

    ring.writeRecallKV(16, 0, &k_src, &v_src, 100);
    ring.writeRecallKV(16, 1, &k_src, &v_src, 50);

    var slots_buf: [16]usize = undefined;
    const count = ring.getActiveSlots(16, 101, &slots_buf);
    try std.testing.expectEqual(@as(usize, 2), count);

    ring.clearRecall();
    const after_clear = ring.getActiveSlots(16, 101, &slots_buf);
    try std.testing.expectEqual(@as(usize, 0), after_clear);
}

test "prefill slot calculation strictly respects physical ring buffer bounds" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 2, 64, 512, 3456, 128);
    defer ring.deinit();

    var dummy_k: [64]f32 = undefined;
    var dummy_v: [64]f32 = undefined;
    @memset(&dummy_k, 1.0);
    @memset(&dummy_v, 2.0);

    // Simulate 5000 tokens ingested over multiple turns
    for (0..5000) |c| {
        _ = ring.activateSlot(0, c);
        ring.writeKV(0, c, &dummy_k, &dummy_v);
    }

    var slots_buf: [4096]usize = undefined;
    const chunk_len = 226;
    const prev_count = ring.getPrefillPrevSlots(0, 5000, chunk_len, &slots_buf);

    // Combined total must never exceed TOTAL_SLOTS (4096)
    try std.testing.expect(prev_count + chunk_len <= ring.total_slots);

    // Anchors must be preserved
    try std.testing.expect(prev_count >= 512);

    // Slots returned must be in strictly monotonic increasing clock order
    var last_c: usize = 0;
    for (0..prev_count) |i| {
        const slot = slots_buf[i];
        const clock = ring.clocks[slot];
        if (i > 0) {
            try std.testing.expect(clock > last_c);
        }
        last_c = clock;
    }
}

test "attention mass accumulation and salience scoring" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 4, 64, 4, 8, 4);
    defer ring.deinit();

    _ = ring.activateSlot(0, 0);
    _ = ring.activateSlot(1, 0);
    _ = ring.activateSlot(2, 0);
    _ = ring.activateSlot(3, 0);

    ring.recordAttention(0, 0, 0.5);
    ring.recordAttention(1, 0, 0.7);
    ring.recordAttention(2, 0, 0.4);
    ring.recordAttention(3, 0, 0.8);

    try std.testing.expectEqual(@as(f32, 0.5), ring.getAttentionMass(0, 0));
    try std.testing.expectEqual(@as(f32, 0.6), ring.getSlotSalience(0));
}

test "evictLowSalienceSlots evicts low salience tokens and preserves heavy hitters" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 2, 64, 4, 8, 4);
    defer ring.deinit();

    var dummy_k: [64]f32 = undefined;
    var dummy_v: [64]f32 = undefined;
    @memset(&dummy_k, 1.0);
    @memset(&dummy_v, 2.0);

    // Ingest slots for clocks 4..9 (Tier 2 dynamic window)
    for (4..10) |c| {
        ring.writeKV(0, c, &dummy_k, &dummy_v);
        ring.writeKV(1, c, &dummy_k, &dummy_v);
    }

    // Assign high attention mass to clock 6 (heavy hitter) and clocks 7..9
    const slot6 = ring.getSlotIndex(6);
    const slot4 = ring.getSlotIndex(4);
    const slot5 = ring.getSlotIndex(5);

    ring.recordAttention(0, slot6, 10.0);
    ring.recordAttention(1, slot6, 10.0);
    for (7..10) |c| {
        const sc = ring.getSlotIndex(c);
        ring.recordAttention(0, sc, 1.0);
        ring.recordAttention(1, sc, 1.0);
    }
    ring.recordAttention(0, slot4, 0.1);
    ring.recordAttention(0, slot5, 0.2);

    // Evict 2 lowest salience slots in range 4..9 (slot4=0.05, slot5=0.10)
    const evicted = ring.evictLowSalienceSlots(4, 9, 2);
    try std.testing.expectEqual(@as(usize, 2), evicted);

    // slot4 and slot5 should now be inactive
    try std.testing.expectEqual(false, ring.active[slot4]);
    try std.testing.expectEqual(false, ring.active[slot5]);

    // slot6 (heavy hitter) must still be active!
    try std.testing.expectEqual(true, ring.active[slot6]);
}

test "getWorkingSetGini distinguishes focused vs diffuse attention distributions" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 2, 64, 4, 8, 4);
    defer ring.deinit();

    var dummy_k: [64]f32 = undefined;
    var dummy_v: [64]f32 = undefined;
    @memset(&dummy_k, 1.0);
    @memset(&dummy_v, 2.0);

    for (4..8) |c| {
        ring.writeKV(0, c, &dummy_k, &dummy_v);
    }

    // 1. Perfectly flat/diffuse distribution -> low Gini
    for (4..8) |c| {
        const slot = ring.getSlotIndex(c);
        ring.recordAttention(0, slot, 1.0);
    }
    const flat_gini = ring.getWorkingSetGini();
    try std.testing.expect(flat_gini < 0.10);
    // 4 active slots out of 4092 tier2 capacity ~ 0.00097 fill ratio
    try std.testing.expect(ring.isWorkingSetSaturated(0.0005, 0.35));

    // 2. Skewed / concentrated distribution -> high Gini
    var ring2 = try DynamicRingBuffer.init(std.testing.allocator, 2, 64, 4, 8, 4);
    defer ring2.deinit();

    for (4..8) |c| {
        ring2.writeKV(0, c, &dummy_k, &dummy_v);
    }
    const slot4 = ring2.getSlotIndex(4);
    const slot5 = ring2.getSlotIndex(5);
    const slot6 = ring2.getSlotIndex(6);
    const slot7 = ring2.getSlotIndex(7);

    ring2.recordAttention(0, slot4, 0.05);
    ring2.recordAttention(0, slot5, 0.05);
    ring2.recordAttention(0, slot6, 0.10);
    ring2.recordAttention(0, slot7, 10.00); // 1 dominant anchor

    const concentrated_gini = ring2.getWorkingSetGini();
    try std.testing.expect(concentrated_gini > 0.60);
    try std.testing.expect(!ring2.isWorkingSetSaturated(0.0005, 0.35));
}

test "ring buffer rollbackClock cleanly deactivates transient probe slots" {
    var ring = try DynamicRingBuffer.init(std.testing.allocator, 2, 64, 4, 8, 4);
    defer ring.deinit();

    // Ingest 10 base tokens
    for (0..10) |c| {
        for (0..2) |l| _ = ring.activateSlot(l, c);
    }
    try std.testing.expectEqual(@as(usize, 10), ring.total_ingested);

    // Ingest 3 transient probe tokens
    for (10..13) |c| {
        for (0..2) |l| _ = ring.activateSlot(l, c);
    }
    try std.testing.expectEqual(@as(usize, 13), ring.total_ingested);
    const slot11 = ring.getSlotIndex(11);
    try std.testing.expectEqual(true, ring.active[slot11]);

    // Rollback to base clock 10
    ring.rollbackClock(10);
    try std.testing.expectEqual(@as(usize, 10), ring.total_ingested);
    try std.testing.expectEqual(false, ring.active[slot11]);
}
