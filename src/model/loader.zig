const std = @import("std");
pub const safetensors = @import("../safetensors.zig");
pub const tensor = @import("../tensor.zig");
pub const types = @import("types.zig");
pub const ring_buffer = @import("../ring_buffer.zig");
pub const memory = @import("../memory.zig");
pub const quiescence = @import("../quiescence.zig");

const bf16 = tensor.bf16;
const ModelConfig = types.ModelConfig;
const LayerWeights = types.LayerWeights;
const DynamicRingBuffer = ring_buffer.DynamicRingBuffer;

pub const Model = struct {
    allocator: std.mem.Allocator,
    config: ModelConfig,
    embed_tokens: []const bf16,
    embed_tokens_per_layer: ?[]const bf16 = null,
    per_layer_model_projection: ?[]const bf16 = null,
    per_layer_projection_norm: ?[]const bf16 = null,
    layers: []LayerWeights,
    final_norm: []const bf16,

    pub fn loadFromSafeTensors(allocator: std.mem.Allocator, st: *const safetensors.SafeTensors, config: ModelConfig) !Model {
        const embed_tv = st.get("model.language_model.embed_tokens.weight") orelse return error.MissingEmbedTokens;
        const norm_tv = st.get("model.language_model.norm.weight") orelse return error.MissingFinalNorm;

        const ple_tv = st.get("model.language_model.embed_tokens_per_layer.weight");
        const plmp_tv = st.get("model.language_model.per_layer_model_projection.weight");
        const plpn_tv = st.get("model.language_model.per_layer_projection_norm.weight");

        var layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);

        var buf: [128]u8 = undefined;
        for (0..config.num_hidden_layers) |l| {
            const is_full = (config.layer_types[l] == .full_attention);
            const head_dim = if (is_full) config.global_head_dim else config.head_dim;
            const rotary_dim = head_dim;
            const num_kv_heads = if (is_full) config.num_global_key_value_heads else config.num_key_value_heads;
            const q_dim = config.num_attention_heads * head_dim;
            const kv_dim = num_kv_heads * head_dim;
            const k_eq_v = (is_full and config.attention_k_eq_v);

            const in_norm = try getSlice(st, &buf, "model.language_model.layers.{d}.input_layernorm.weight", .{l});
            const q = try getSlice(st, &buf, "model.language_model.layers.{d}.self_attn.q_proj.weight", .{l});
            const k = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.self_attn.k_proj.weight", .{l}) orelse &[_]bf16{};
            const v = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.self_attn.v_proj.weight", .{l}) orelse &[_]bf16{};
            const o = try getSlice(st, &buf, "model.language_model.layers.{d}.self_attn.o_proj.weight", .{l});
            const qn = try getSlice(st, &buf, "model.language_model.layers.{d}.self_attn.q_norm.weight", .{l});
            const kn = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.self_attn.k_norm.weight", .{l}) orelse &[_]bf16{};
            const pre_ffn = try getSlice(st, &buf, "model.language_model.layers.{d}.pre_feedforward_layernorm.weight", .{l});
            const gate = try getSlice(st, &buf, "model.language_model.layers.{d}.mlp.gate_proj.weight", .{l});
            const up = try getSlice(st, &buf, "model.language_model.layers.{d}.mlp.up_proj.weight", .{l});
            const down = try getSlice(st, &buf, "model.language_model.layers.{d}.mlp.down_proj.weight", .{l});
            const intermediate_dim = gate.len / config.hidden_size;

            const post_attn = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.post_attention_layernorm.weight", .{l});
            const post_ffn = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.post_feedforward_layernorm.weight", .{l});
            const scalar = getOptionalScalar(st, &buf, "model.language_model.layers.{d}.layer_scalar", .{l});

            const pl_gate = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.per_layer_input_gate.weight", .{l});
            const pl_proj = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.per_layer_projection.weight", .{l});
            const pl_norm = getOptionalSlice(st, &buf, "model.language_model.layers.{d}.post_per_layer_input_norm.weight", .{l});

            layers[l] = LayerWeights{
                .layer_type = if (is_full) .full_attention else .sliding_attention,
                .head_dim = head_dim,
                .rotary_dim = rotary_dim,
                .q_dim = q_dim,
                .kv_dim = kv_dim,
                .num_kv_heads = num_kv_heads,
                .k_eq_v = k_eq_v,
                .intermediate_dim = intermediate_dim,
                .input_layernorm = in_norm,
                .q_proj = q,
                .k_proj = k,
                .v_proj = v,
                .o_proj = o,
                .q_norm = qn,
                .k_norm = kn,
                .post_attention_layernorm = post_attn,
                .pre_feedforward_layernorm = pre_ffn,
                .gate_proj = gate,
                .up_proj = up,
                .down_proj = down,
                .post_feedforward_layernorm = post_ffn,
                .layer_scalar = scalar,
                .per_layer_input_gate = pl_gate,
                .per_layer_projection = pl_proj,
                .post_per_layer_input_norm = pl_norm,
            };
        }

        return Model{
            .allocator = allocator,
            .config = config,
            .embed_tokens = embed_tv.asSlice(bf16),
            .embed_tokens_per_layer = if (ple_tv) |p| p.asSlice(bf16) else null,
            .per_layer_model_projection = if (plmp_tv) |p| p.asSlice(bf16) else null,
            .per_layer_projection_norm = if (plpn_tv) |p| p.asSlice(bf16) else null,
            .layers = layers,
            .final_norm = norm_tv.asSlice(bf16),
        };
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.layers);
    }

    pub fn forwardToken(
        self: *const Model,
        ring: *DynamicRingBuffer,
        scratch: *types.ForwardScratch,
        token_id: u32,
        clock: usize,
        thread_pool: ?*std.Thread.Pool,
        memory_opt: ?*memory.DiffArchive,
        quiescence_opt: ?*quiescence.QuiescenceTracker,
        gpu_opt: ?*@import("../gpu.zig").model_gpu.GpuModelContext,
        compute_logits: bool,
    ) u32 {
        const fwd = @import("forward.zig");
        return fwd.forwardToken(self, ring, scratch, token_id, clock, thread_pool, memory_opt, quiescence_opt, gpu_opt, compute_logits);
    }
};

fn getSlice(st: *const safetensors.SafeTensors, buf: []u8, comptime fmt: []const u8, args: anytype) ![]const bf16 {
    const name = try std.fmt.bufPrint(buf, fmt, args);
    const tv = st.get(name) orelse return error.MissingLayerWeight;
    return tv.asSlice(bf16);
}

fn getOptionalSlice(st: *const safetensors.SafeTensors, buf: []u8, comptime fmt: []const u8, args: anytype) ?[]const bf16 {
    const name = std.fmt.bufPrint(buf, fmt, args) catch return null;
    const tv = st.get(name) orelse return null;
    return tv.asSlice(bf16);
}

fn getOptionalScalar(st: *const safetensors.SafeTensors, buf: []u8, comptime fmt: []const u8, args: anytype) ?f32 {
    const name = std.fmt.bufPrint(buf, fmt, args) catch return null;
    const tv = st.get(name) orelse return null;
    return tv.asSlice(bf16)[0].toF32();
}
