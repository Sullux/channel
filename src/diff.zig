const std = @import("std");

pub const StateDelta = struct {
    norm: f32,
    salience: f32,
    is_landmark: bool,
};

pub fn computeDelta(
    delta_out: []f32,
    current_state: []const f32,
    previous_state: []const f32,
    threshold: f32,
) StateDelta {
    var sum_sq: f32 = 0.0;
    var max_diff: f32 = 0.0;

    for (delta_out, current_state, previous_state) |*d, curr, prev| {
        const diff = curr - prev;
        d.* = diff;
        const abs_diff = @abs(diff);
        if (abs_diff > max_diff) max_diff = abs_diff;
        sum_sq += diff * diff;
    }

    const norm = @sqrt(sum_sq);
    const mean_norm = norm / @sqrt(@as(f32, @floatFromInt(current_state.len)));
    const is_landmark = (mean_norm >= threshold);

    return StateDelta{
        .norm = norm,
        .salience = mean_norm,
        .is_landmark = is_landmark,
    };
}

pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    var dot: f32 = 0.0;
    var norm_a_sq: f32 = 0.0;
    var norm_b_sq: f32 = 0.0;

    for (a, b) |va, vb| {
        dot += va * vb;
        norm_a_sq += va * va;
        norm_b_sq += vb * vb;
    }

    const denom = @sqrt(norm_a_sq) * @sqrt(norm_b_sq);
    if (denom < 1e-12) return 0.0;
    return dot / denom;
}

test "computeDelta calculation" {
    const s1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const s2 = [_]f32{ 1.1, 2.2, 3.0, 4.0 };
    var delta: [4]f32 = undefined;

    const res = computeDelta(&delta, &s2, &s1, 0.05);
    try std.testing.expect(res.norm > 0.2);
    try std.testing.expect(res.salience > 0.1);
    try std.testing.expect(res.is_landmark);
}
