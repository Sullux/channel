const std = @import("std");
const kernels = @import("kernels.zig");
pub const TopKCandidate = @import("model/types.zig").TopKCandidate;

pub const IndexedLogit = struct {
    id: u32,
    val: f32,
};

fn pushTopK(heap: []IndexedLogit, size: *usize, max_k: usize, item: IndexedLogit) void {
    if (size.* < max_k) {
        heap[size.*] = item;
        size.* += 1;
        var idx = size.* - 1;
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (heap[idx].val < heap[parent].val) {
                const tmp = heap[idx];
                heap[idx] = heap[parent];
                heap[parent] = tmp;
                idx = parent;
            } else break;
        }
        return;
    }
    if (item.val <= heap[0].val) return;
    heap[0] = item;
    var idx: usize = 0;
    while (true) {
        const left = 2 * idx + 1;
        const right = 2 * idx + 2;
        var smallest = idx;
        if (left < max_k and heap[left].val < heap[smallest].val) smallest = left;
        if (right < max_k and heap[right].val < heap[smallest].val) smallest = right;
        if (smallest == idx) break;
        const tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

inline fn isCritiqueToken(tok: u32) bool {
    return switch (tok) {
        // "Wait", " Wait", "wait", " wait", "WAIT", " WAIT"
        20470, 29202, 12429, 4491, 83675, 213110,
        // "Actually", " Actually", "actually", " actually"
        68755, 38403, 89429, 3643,
        // "But", "but"
        4573, 5503,
        // "However", "however"
        9675, 97485 => true,
        else => false,
    };
}

pub const Sampler = struct {
    prng: std.Random.DefaultPrng,
    top_k: u32 = 64,
    repeat_penalty: f32 = 1.1,
    frequency_penalty: f32 = 0.1,
    presence_penalty: f32 = 0.1,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    temp: f32 = 1.0,
    suppress_thinking: bool = false,
    suppress_critique: bool = false,
    suppress_channel_close: bool = false,

    pub fn init(seed: u64, temp: f32, top_p: f32) Sampler {
        return .{
            .prng = std.Random.DefaultPrng.init(seed),
            .temp = temp,
            .top_p = top_p,
            .top_k = 64,
            .repeat_penalty = 1.1,
            .frequency_penalty = 0.1,
            .presence_penalty = 0.1,
            .min_p = 0.05,
            .suppress_thinking = false,
            .suppress_critique = false,
            .suppress_channel_close = false,
        };
    }

    pub fn sample(self: *Sampler, logits: []f32, recent_tokens: ?[]const u32) u32 {
        const suppress = [_]u32{ 0, 258882, 258883, 255999, 256000, 256001, 255995, 255996, 255997, 255998 };
        for (suppress) |sup| {
            if (sup < logits.len) logits[sup] = -1e9;
        }
        if (self.suppress_thinking and 100 < logits.len) logits[100] = -1e9;
        if (self.suppress_channel_close and 101 < logits.len) logits[101] = -1e9;
        if (self.suppress_critique) {
            const critique_tokens = [_]u32{
                20470, 29202, 12429, 4491, 83675, 213110,
                68755, 38403, 89429, 3643,
                4573, 5503,
                9675, 97485,
            };
            for (critique_tokens) |ct| {
                if (ct < logits.len) logits[ct] = -1e9;
            }
        }

        const cap: f32 = 30.0;
        const inv_cap: f32 = 1.0 / cap;
        for (logits) |*l| {
            if (l.* > -1e8) {
                l.* = std.math.tanh(l.* * inv_cap) * cap;
            }
        }

        if (recent_tokens != null and (self.repeat_penalty > 1.0 or self.frequency_penalty > 0.0 or self.presence_penalty > 0.0)) {
            for (recent_tokens.?, 0..) |tok, i| {
                var already_seen = false;
                for (recent_tokens.?[0..i]) |prev| {
                    if (prev == tok) { already_seen = true; break; }
                }
                if (already_seen) continue;

                var count: usize = 0;
                for (recent_tokens.?) |rt| {
                    if (rt == tok) count += 1;
                }

                if (tok < logits.len) {
                    if (self.repeat_penalty > 1.0) {
                        if (logits[tok] > 0.0) {
                            logits[tok] /= self.repeat_penalty;
                        } else {
                            logits[tok] *= self.repeat_penalty;
                        }
                    }
                    logits[tok] -= self.presence_penalty + (@as(f32, @floatFromInt(count)) * self.frequency_penalty);
                }
            }
        }

        if (self.temp <= 0.0) return kernels.sampleArgmax(logits);

        var top_heap: [64]IndexedLogit = undefined;
        var heap_size: usize = 0;
        const max_k = @min(64, @as(usize, self.top_k));

        for (logits, 0..) |v, i| {
            if (v > -1e8) {
                pushTopK(&top_heap, &heap_size, max_k, .{ .id = @intCast(i), .val = v });
            }
        }
        if (heap_size == 0) return kernels.sampleArgmax(logits);

        const candidates = top_heap[0..heap_size];
        std.mem.sort(IndexedLogit, candidates, {}, struct {
            fn cmp(_: void, a: IndexedLogit, b: IndexedLogit) bool { return a.val > b.val; }
        }.cmp);

        var probs: [64]f32 = undefined;
        var max_scaled: f32 = -1e9;
        const t = if (self.temp > 0.0) self.temp else 1.0;
        for (candidates, 0..) |cand, i| {
            const scaled = cand.val / t;
            probs[i] = scaled;
            if (scaled > max_scaled) max_scaled = scaled;
        }

        var sum: f32 = 0.0;
        for (0..heap_size) |i| {
            probs[i] = @exp(probs[i] - max_scaled);
            sum += probs[i];
        }
        for (0..heap_size) |i| probs[i] /= sum;

        // Min-P filter
        const max_p = probs[0];
        const p_thresh = max_p * self.min_p;
        var valid_k: usize = 0;
        var valid_sum: f32 = 0.0;
        for (0..heap_size) |i| {
            if (probs[i] >= p_thresh) {
                valid_k = i + 1;
                valid_sum += probs[i];
            } else break;
        }
        if (valid_k == 0) valid_k = 1;
        for (0..valid_k) |i| probs[i] /= valid_sum;

        // Top-P filter
        var cum_p: f32 = 0.0;
        var cutoff: usize = 0;
        for (0..valid_k) |i| {
            cum_p += probs[i];
            cutoff = i + 1;
            if (cum_p >= self.top_p) break;
        }

        const r = self.prng.random().float(f32) * cum_p;
        var acc: f32 = 0.0;
        var chosen_id = candidates[0].id;
        for (0..cutoff) |i| {
            acc += probs[i];
            if (acc >= r) {
                chosen_id = candidates[i].id;
                break;
            }
        }

        return chosen_id;
    }

    pub fn sampleTopK(self: *Sampler, candidates: []const TopKCandidate, recent_tokens: ?[]const u32) u32 {
        if (candidates.len == 0) return 0;
        const has_penalties = (self.repeat_penalty > 1.0 or self.frequency_penalty > 0.0 or self.presence_penalty > 0.0);
        if (self.temp <= 0.0 and (!has_penalties or recent_tokens == null) and !self.suppress_thinking and !self.suppress_critique and !self.suppress_channel_close) return candidates[0].id;

        var items: [64]TopKCandidate = undefined;
        var K: usize = 0;
        for (candidates) |cand| {
            if (cand.id == 0 or cand.id == 258882 or cand.id == 258883 or cand.id == 255999 or cand.id == 256000 or cand.id == 256001 or cand.id == 255995 or cand.id == 255996 or cand.id == 255997 or cand.id == 255998) continue;
            if (self.suppress_thinking and cand.id == 100) continue;
            if (self.suppress_channel_close and cand.id == 101) continue;
            if (self.suppress_critique and isCritiqueToken(cand.id)) continue;
            items[K] = cand;
            K += 1;
            if (K == 64) break;
        }
        if (K == 0) return candidates[0].id;

        if (recent_tokens != null and has_penalties) {
            for (recent_tokens.?, 0..) |tok, i| {
                var already_seen = false;
                for (recent_tokens.?[0..i]) |prev| {
                    if (prev == tok) {
                        already_seen = true;
                        break;
                    }
                }
                if (already_seen) continue;

                var count: usize = 0;
                for (recent_tokens.?) |rt| {
                    if (rt == tok) count += 1;
                }

                for (items[0..K]) |*item| {
                    if (item.id == tok) {
                        if (self.repeat_penalty > 1.0) {
                            if (item.val > 0.0) {
                                item.val /= self.repeat_penalty;
                            } else {
                                item.val *= self.repeat_penalty;
                            }
                        }
                        item.val -= self.presence_penalty + (@as(f32, @floatFromInt(count)) * self.frequency_penalty);
                    }
                }
            }

            // Re-sort items in descending order after applying repeat penalties
            std.mem.sort(TopKCandidate, items[0..K], {}, struct {
                fn cmp(_: void, a: TopKCandidate, b: TopKCandidate) bool {
                    return a.val > b.val;
                }
            }.cmp);
        }

        if (self.temp <= 0.0) return items[0].id;

        var probs: [64]f32 = undefined;
        var max_scaled: f32 = -1e9;
        const t = if (self.temp > 0.0) self.temp else 1.0;
        for (items[0..K], 0..) |cand, i| {
            const scaled = cand.val / t;
            probs[i] = scaled;
            if (scaled > max_scaled) max_scaled = scaled;
        }

        var sum: f32 = 0.0;
        for (0..K) |i| {
            probs[i] = @exp(probs[i] - max_scaled);
            sum += probs[i];
        }
        for (0..K) |i| probs[i] /= sum;

        const max_p = probs[0];
        const p_thresh = max_p * self.min_p;
        var valid_k: usize = 0;
        var valid_sum: f32 = 0.0;
        for (0..K) |i| {
            if (probs[i] >= p_thresh) {
                valid_k = i + 1;
                valid_sum += probs[i];
            } else break;
        }
        if (valid_k == 0) valid_k = 1;
        for (0..valid_k) |i| probs[i] /= valid_sum;

        var cum_p: f32 = 0.0;
        var cutoff: usize = 0;
        for (0..valid_k) |i| {
            cum_p += probs[i];
            cutoff = i + 1;
            if (cum_p >= self.top_p) break;
        }

        const r = self.prng.random().float(f32) * cum_p;
        var acc: f32 = 0.0;
        var chosen_id = items[0].id;
        for (0..cutoff) |i| {
            acc += probs[i];
            if (acc >= r) {
                chosen_id = items[i].id;
                break;
            }
        }

        return chosen_id;
    }

    pub const ConstrainedEvalResult = struct {
        winning_idx: usize,
        winning_token: u32,
        confidence: f32, // Normalized softmax probability (0.0 .. 1.0)
        entropy: f32,    // Shannon entropy across candidate distribution
    };

    pub fn evalConstrained(self: *Sampler, logits: []const f32, candidate_tokens: []const u32) ConstrainedEvalResult {
        _ = self;
        if (candidate_tokens.len == 0) return .{ .winning_idx = 0, .winning_token = 0, .confidence = 0.0, .entropy = 0.0 };
        var max_val: f32 = -1e9;
        var best_idx: usize = 0;
        var vals: [32]f32 = undefined;
        const count = @min(candidate_tokens.len, vals.len);
        for (0..count) |i| {
            const tok = candidate_tokens[i];
            const v = if (tok < logits.len) logits[tok] else -1e9;
            vals[i] = v;
            if (v > max_val) {
                max_val = v;
                best_idx = i;
            }
        }

        var sum_exp: f32 = 0.0;
        var probs: [32]f32 = undefined;
        for (0..count) |i| {
            const ep = @exp(vals[i] - max_val);
            probs[i] = ep;
            sum_exp += ep;
        }

        var entropy: f32 = 0.0;
        const inv_sum = if (sum_exp > 0.0) 1.0 / sum_exp else 1.0;
        for (0..count) |i| {
            const p = probs[i] * inv_sum;
            probs[i] = p;
            if (p > 1e-12) {
                entropy -= p * @log(p);
            }
        }

        return .{
            .winning_idx = best_idx,
            .winning_token = candidate_tokens[best_idx],
            .confidence = probs[best_idx],
            .entropy = entropy,
        };
    }

    pub fn evalConstrainedTopK(self: *Sampler, candidates: []const TopKCandidate, candidate_tokens: []const u32) ConstrainedEvalResult {
        _ = self;
        if (candidate_tokens.len == 0) return .{ .winning_idx = 0, .winning_token = 0, .confidence = 0.0, .entropy = 0.0 };
        var max_val: f32 = -1e9;
        var best_idx: usize = 0;
        var vals: [32]f32 = undefined;
        const count = @min(candidate_tokens.len, vals.len);
        const min_topk_val = if (candidates.len > 0) candidates[candidates.len - 1].val - 10.0 else -100.0;

        for (0..count) |i| {
            const tok = candidate_tokens[i];
            var v: f32 = min_topk_val;
            for (candidates) |c| {
                if (c.id == tok) {
                    v = c.val;
                    break;
                }
            }
            vals[i] = v;
            if (v > max_val) {
                max_val = v;
                best_idx = i;
            }
        }

        var sum_exp: f32 = 0.0;
        var probs: [32]f32 = undefined;
        for (0..count) |i| {
            const ep = @exp(vals[i] - max_val);
            probs[i] = ep;
            sum_exp += ep;
        }

        var entropy: f32 = 0.0;
        const inv_sum = if (sum_exp > 0.0) 1.0 / sum_exp else 1.0;
        for (0..count) |i| {
            const p = probs[i] * inv_sum;
            probs[i] = p;
            if (p > 1e-12) {
                entropy -= p * @log(p);
            }
        }

        return .{
            .winning_idx = best_idx,
            .winning_token = candidate_tokens[best_idx],
            .confidence = probs[best_idx],
            .entropy = entropy,
        };
    }

    pub fn sampleConstrained(self: *Sampler, logits: []const f32, candidate_tokens: []const u32) u32 {
        return self.evalConstrained(logits, candidate_tokens).winning_token;
    }

    pub fn sampleConstrainedTopK(self: *Sampler, candidates: []const TopKCandidate, candidate_tokens: []const u32) u32 {
        return self.evalConstrainedTopK(candidates, candidate_tokens).winning_token;
    }
};

test "sampler.sampleTopK re-sorts penalized candidates" {
    // Test deterministic greedy with penalty
    var greedy_sampler = Sampler.init(42, 0.0, 0.95);
    greedy_sampler.repeat_penalty = 1.2;

    const candidates = [_]TopKCandidate{
        .{ .id = 4373, .val = 18.0 }, // "pass"
        .{ .id = 705, .val = 17.5 },  // "return"
        .{ .id = 1980, .val = 16.0 }, // "continue"
    };

    // Before penalty, candidate 4373 would win.
    // If recent_tokens contains 4373, it should be penalized (18.0 / 1.2 = 15.0)
    // and candidate 705 (val 17.5) should become the top choice.
    const recent = [_]u32{ 4373, 4373 };
    const chosen = greedy_sampler.sampleTopK(&candidates, &recent);
    try std.testing.expectEqual(@as(u32, 705), chosen);

    // Test low temp sampling
    var low_temp_sampler = Sampler.init(42, 0.01, 0.95);
    low_temp_sampler.repeat_penalty = 1.2;
    const chosen_temp = low_temp_sampler.sampleTopK(&candidates, &recent);
    try std.testing.expectEqual(@as(u32, 705), chosen_temp);
}

test "sampler.sampleTopK frequency and presence penalties suppress repeated tokens" {
    var sampler = Sampler.init(42, 0.0, 0.95);
    sampler.repeat_penalty = 1.0;
    sampler.frequency_penalty = 0.5;
    sampler.presence_penalty = 0.5;

    const candidates = [_]TopKCandidate{
        .{ .id = 6639, .val = 18.0 }, // "call"
        .{ .id = 16132, .val = 17.0 }, // "will"
    };

    // If 6639 appears 3 times:
    // val becomes 18.0 - (0.5 + 3 * 0.5) = 18.0 - 2.0 = 16.0
    // Candidate 16132 (val 17.0) becomes the winner.
    const recent = [_]u32{ 6639, 6639, 6639 };
    const chosen = sampler.sampleTopK(&candidates, &recent);
    try std.testing.expectEqual(@as(u32, 16132), chosen);
}

test "sampler.sampleTopK suppresses token 100 when suppress_thinking is active" {
    var sampler = Sampler.init(42, 0.0, 0.95);
    sampler.suppress_thinking = true;

    const candidates = [_]TopKCandidate{
        .{ .id = 100, .val = 20.0 }, // <|channel>
        .{ .id = 236777, .val = 15.0 }, // "I"
    };

    const chosen = sampler.sampleTopK(&candidates, null);
    try std.testing.expectEqual(@as(u32, 236777), chosen);
}

test "sampler.sampleTopK suppresses token 101 when suppress_channel_close is active" {
    var sampler = Sampler.init(42, 0.0, 0.95);
    sampler.suppress_channel_close = true;

    const candidates = [_]TopKCandidate{
        .{ .id = 101, .val = 25.0 }, // <channel|>
        .{ .id = 120474, .val = 20.0 }, // "Thinking"
    };

    const chosen = sampler.sampleTopK(&candidates, null);
    try std.testing.expectEqual(@as(u32, 120474), chosen);
}

test "sampler.sampleTopK suppresses critique tokens when suppress_critique is active" {
    var sampler = Sampler.init(42, 0.0, 0.95);
    sampler.suppress_critique = true;

    const candidates = [_]TopKCandidate{
        .{ .id = 20470, .val = 22.0 }, // "Wait"
        .{ .id = 68755, .val = 21.0 }, // "Actually"
        .{ .id = 4573, .val = 20.0 },  // "But"
        .{ .id = 705, .val = 15.0 },   // "return"
    };

    const chosen = sampler.sampleTopK(&candidates, null);
    try std.testing.expectEqual(@as(u32, 705), chosen);
}

test "sampler.sampleConstrained selects highest candidate logit" {
    var sampler = Sampler.init(42, 0.0, 0.95);
    const candidates = [_]TopKCandidate{
        .{ .id = 101, .val = 15.0 },
        .{ .id = 202, .val = 25.0 },
        .{ .id = 303, .val = 18.0 },
    };
    const cands = [_]u32{ 101, 303 };
    const chosen = sampler.sampleConstrainedTopK(&candidates, &cands);
    try std.testing.expectEqual(@as(u32, 303), chosen);

    const eval_res = sampler.evalConstrainedTopK(&candidates, &cands);
    try std.testing.expectEqual(@as(usize, 1), eval_res.winning_idx);
    try std.testing.expectEqual(@as(u32, 303), eval_res.winning_token);
    try std.testing.expect(eval_res.confidence > 0.9);
    try std.testing.expect(eval_res.entropy >= 0.0);
}
