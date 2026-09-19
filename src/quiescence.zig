const std = @import("std");

pub const QuiescenceConfig = struct {
    enabled: bool = false,
    split_ratio: f32 = 0.5,
    threshold: f32 = 0.02,
    forced_cadence: usize = 4,
};

pub const QuiescenceTracker = struct {
    config: QuiescenceConfig,
    num_layers: usize,
    total_evals: u64 = 0,
    skipped_evals: u64 = 0,

    pub fn init(config: QuiescenceConfig, num_layers: usize) QuiescenceTracker {
        return .{
            .config = config,
            .num_layers = num_layers,
        };
    }

    pub fn reset(self: *QuiescenceTracker) void {
        self.total_evals = 0;
        self.skipped_evals = 0;
    }

    /// Evaluates whether an upper-tier layer should execute its compute or stay quiescent.
    /// Fast lower layers (0 .. split_layer - 1) always execute.
    pub fn shouldExecute(self: *QuiescenceTracker, layer_idx: usize, clock: usize, x: []const f32, prev_x: []const f32) bool {
        if (!self.config.enabled) return true;

        const split_layer: usize = @intFromFloat(@as(f32, @floatFromInt(self.num_layers)) * self.config.split_ratio);
        if (layer_idx < split_layer) return true;

        // Force periodic refresh to prevent semantic drift
        if (clock % self.config.forced_cadence == 0) return true;

        self.total_evals += 1;

        var diff_sum_sq: f32 = 0.0;
        var norm_sum_sq: f32 = 0.0;
        for (x, prev_x) |curr, prev| {
            const d = curr - prev;
            diff_sum_sq += d * d;
            norm_sum_sq += curr * curr;
        }

        const relative_velocity = if (norm_sum_sq > 1e-12) @sqrt(diff_sum_sq / norm_sum_sq) else 0.0;
        if (relative_velocity < self.config.threshold) {
            self.skipped_evals += 1;
            return false;
        }
        return true;
    }

    pub fn getSkipRate(self: *const QuiescenceTracker) f32 {
        if (self.total_evals == 0) return 0.0;
        return @as(f32, @floatFromInt(self.skipped_evals)) / @as(f32, @floatFromInt(self.total_evals));
    }
};

test "quiescence tracker preserves lower layers unconditionally" {
    var tracker = QuiescenceTracker.init(.{ .enabled = true, .split_ratio = 0.5, .threshold = 0.5 }, 10);
    const x = [_]f32{ 1.0, 1.0 };
    const prev_x = [_]f32{ 1.0, 1.0 };

    // Layers 0..4 must always execute regardless of zero delta
    for (0..5) |l| {
        try std.testing.expect(tracker.shouldExecute(l, 1, &x, &prev_x));
    }

    // Upper layer 5 with zero delta should skip on non-forced cadence clock
    try std.testing.expect(!tracker.shouldExecute(5, 1, &x, &prev_x));
    // Upper layer 5 should execute on forced cadence clock (e.g. 4)
    try std.testing.expect(tracker.shouldExecute(5, 4, &x, &prev_x));
}
