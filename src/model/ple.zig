const std = @import("std");
pub const tensor = @import("../tensor.zig");
pub const bf16 = tensor.bf16;
pub const model = @import("../model.zig");
pub const Model = model.Model;
pub const LayerWeights = model.LayerWeights;
pub const ForwardScratch = model.ForwardScratch;
pub const kernels = @import("../kernels.zig");

pub fn preparePLE(self: *const Model, scratch: *ForwardScratch, token_id: u32, H: usize, ple_dim: usize, tp: ?*std.Thread.Pool) void {
    const ple_table = self.embed_tokens_per_layer orelse return;
    const total_ple_dim = self.layers.len * ple_dim;
    const ple_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
    const inv_sqrt_2: f32 = 0.70710678118;

    if (self.per_layer_model_projection) |plmp| {
        if (tp) |pool| kernels.gemvParallel(scratch.ple_context, scratch.x, plmp, total_ple_dim, H, pool) else kernels.gemv(scratch.ple_context, scratch.x, plmp, total_ple_dim, H);
        const proj_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(H)));
        for (scratch.ple_context) |*v| v.* *= proj_scale;

        if (self.per_layer_projection_norm) |plpn| {
            for (0..self.layers.len) |l| {
                const layer_ctx = scratch.ple_context[l * ple_dim .. (l + 1) * ple_dim];
                kernels.rmsNorm(layer_ctx, layer_ctx, plpn, self.config.rms_norm_eps);
            }
        }
        const tok_ple_offset = @as(usize, token_id) * total_ple_dim;
        const tok_ple_row = ple_table[tok_ple_offset .. tok_ple_offset + total_ple_dim];
        for (scratch.ple_context, tok_ple_row) |*ctx, tok_p| ctx.* = (ctx.* + tok_p.toF32() * ple_scale) * inv_sqrt_2;
    }
}

pub fn forwardPLE(self: *const Model, l: LayerWeights, scratch: *ForwardScratch, layer_idx: usize, ple_dim: usize, H: usize) void {
    if (self.embed_tokens_per_layer == null) return;
    if (l.per_layer_input_gate == null or l.per_layer_projection == null or l.post_per_layer_input_norm == null) return;

    kernels.gemv(scratch.ple_buf_1, scratch.x, l.per_layer_input_gate.?, ple_dim, H);
    const layer_ple_ctx = scratch.ple_context[layer_idx * ple_dim .. (layer_idx + 1) * ple_dim];
    for (scratch.ple_buf_1, layer_ple_ctx) |*g, ctx_val| g.* = kernels.geluTanh(g.*) * ctx_val;
    kernels.gemv(scratch.ple_buf_2, scratch.ple_buf_1, l.per_layer_projection.?, H, ple_dim);
    kernels.rmsNorm(scratch.ple_buf_2, scratch.ple_buf_2, l.post_per_layer_input_norm.?, self.config.rms_norm_eps);
    for (scratch.x, scratch.ple_buf_2) |*x_val, p_val| x_val.* += p_val;
}
