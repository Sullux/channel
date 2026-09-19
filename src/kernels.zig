const std = @import("std");
const tensor = @import("tensor.zig");
const bf16 = tensor.bf16;

pub const QK: usize = 32;

pub fn geluTanh(x: f32) f32 {
    const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

pub fn swiGlu(out: []f32, gate: []const f32, up: []const f32) void {
    for (out, gate, up) |*o, g, u| {
        o.* = geluTanh(g) * u;
    }
}

pub fn rmsNorm(out: []f32, in: []const f32, weight: anytype, eps: f32) void {
    var sum_sq: f32 = 0.0;
    for (in) |v| sum_sq += v * v;
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(in.len)) + eps);
    for (out, in, weight) |*o, i, w| {
        const w_f: f32 = if (@TypeOf(w) == bf16) w.toF32() else w;
        o.* = i * inv_rms * w_f;
    }
}

pub fn unitRmsNorm(out: []f32, in: []const f32, eps: f32) void {
    var sum_sq: f32 = 0.0;
    for (in) |v| sum_sq += v * v;
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(in.len)) + eps);
    for (out, in) |*o, i| o.* = i * inv_rms;
}

pub fn dotF32(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

pub fn dotBf16(w: []const bf16, x: []const f32) f32 {
    var sum: f32 = 0.0;
    for (w, x) |w_val, x_val| sum += w_val.toF32() * x_val;
    return sum;
}

pub fn gemv(out: []f32, in: []const f32, w: []const bf16, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        out[r] = dotBf16(w[r * cols .. (r + 1) * cols], in);
    }
}

pub fn gemvParallel(out: []f32, in: []const f32, w: []const bf16, rows: usize, cols: usize, pool: *std.Thread.Pool) void {
    const chunk_size = @max(1, rows / (pool.threads.len + 1));
    var wg = std.Thread.WaitGroup{};
    var r_start: usize = 0;
    while (r_start < rows) {
        const r_end = @min(rows, r_start + chunk_size);
        wg.start();
        pool.spawnWg(&wg, struct {
            fn run(o: []f32, in_v: []const f32, w_m: []const bf16, s: usize, e: usize, c: usize) void {
                for (s..e) |r| o[r] = dotBf16(w_m[r * c .. (r + 1) * c], in_v);
            }
        }.run, .{ out, in, w, r_start, r_end, cols });
        r_start = r_end;
    }
    pool.waitAndWork(&wg);
}

pub fn gatedMlp(out: []f32, in: []const f32, gw: []const bf16, uw: []const bf16, dw: []const bf16, h: usize, inter: usize, act: []f32, tp: ?*std.Thread.Pool) void {
    if (tp) |pool| {
        const chunk_size = @max(1, inter / (pool.threads.len + 1));
        var wg = std.Thread.WaitGroup{};
        var r_start: usize = 0;
        while (r_start < inter) {
            const r_end = @min(inter, r_start + chunk_size);
            wg.start();
            pool.spawnWg(&wg, struct {
                fn run(a: []f32, in_v: []const f32, g_w: []const bf16, u_w: []const bf16, s: usize, e: usize, c: usize) void {
                    for (s..e) |r| {
                        const g_dot = dotBf16(g_w[r * c .. (r + 1) * c], in_v);
                        const u_dot = dotBf16(u_w[r * c .. (r + 1) * c], in_v);
                        a[r] = geluTanh(g_dot) * u_dot;
                    }
                }
            }.run, .{ act, in, gw, uw, r_start, r_end, h });
            r_start = r_end;
        }
        pool.waitAndWork(&wg);
        gemvParallel(out, act[0..inter], dw, h, inter, pool);
    } else {
        for (0..inter) |r| {
            const g_dot = dotBf16(gw[r * h .. (r + 1) * h], in);
            const u_dot = dotBf16(uw[r * h .. (r + 1) * h], in);
            act[r] = geluTanh(g_dot) * u_dot;
        }
        gemv(out, act[0..inter], dw, h, inter);
    }
}

pub fn applyRopePartial(vec: []f32, pos: usize, head_dim: usize, rotary_dim: usize, theta: f32) void {
    const num_heads = vec.len / head_dim;
    const half_rot = rotary_dim / 2;
    for (0..num_heads) |h| {
        const offset = h * head_dim;
        for (0..half_rot) |i| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rotary_dim)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            const cos_v = @cos(angle);
            const sin_v = @sin(angle);
            const idx0 = offset + i;
            const idx1 = offset + i + half_rot;
            const v0 = vec[idx0];
            const v1 = vec[idx1];
            vec[idx0] = v0 * cos_v - v1 * sin_v;
            vec[idx1] = v0 * sin_v + v1 * cos_v;
        }
    }
}

pub fn softmax(logits: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| max_val = @max(max_val, v);
    var sum_exp: f32 = 0.0;
    for (logits) |*v| {
        v.* = @exp(v.* - max_val);
        sum_exp += v.*;
    }
    const inv = 1.0 / sum_exp;
    for (logits) |*v| v.* *= inv;
}

pub fn sampleArgmax(logits: []const f32) u32 {
    var max_idx: u32 = 0;
    var max_val: f32 = logits[0];
    for (logits[1..], 1..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}
