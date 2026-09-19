const std = @import("std");
pub const tensor = @import("../tensor.zig");
pub const ring_buffer = @import("../ring_buffer.zig");

const bf16 = tensor.bf16;
pub const DynamicRingBuffer = ring_buffer.DynamicRingBuffer;
pub const LayerType = enum { sliding_attention, full_attention };

/// Upper bound on associative recall slots injected per cycle. The runtime
/// ring buffer may request fewer via the `--recall` CLI flag.
pub const MAX_RECALL_SLOTS: usize = 512;

pub const LayerWeights = struct {
    layer_type: LayerType,
    head_dim: usize,
    rotary_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    num_kv_heads: usize,
    k_eq_v: bool,
    intermediate_dim: usize,

    input_layernorm: []const bf16,
    q_proj: []const bf16,
    k_proj: []const bf16,
    v_proj: []const bf16,
    o_proj: []const bf16,
    q_norm: []const bf16,
    k_norm: []const bf16,
    post_attention_layernorm: ?[]const bf16 = null,
    pre_feedforward_layernorm: []const bf16,
    gate_proj: []const bf16,
    up_proj: []const bf16,
    down_proj: []const bf16,
    post_feedforward_layernorm: ?[]const bf16 = null,
    layer_scalar: ?f32 = null,

    per_layer_input_gate: ?[]const bf16 = null,
    per_layer_projection: ?[]const bf16 = null,
    post_per_layer_input_norm: ?[]const bf16 = null,
};

pub const ModelConfig = struct {
    vocab_size: usize = 262144,
    hidden_size: usize = 1536,
    intermediate_size: usize = 12288,
    hidden_size_per_layer_input: usize = 0,
    num_hidden_layers: usize = 35,
    num_attention_heads: usize = 8,
    num_key_value_heads: usize = 1,
    num_global_key_value_heads: usize = 1,
    head_dim: usize = 256,
    global_head_dim: usize = 256,
    num_kv_shared_layers: usize = 0,
    attention_k_eq_v: bool = false,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,
    rope_theta_full: f32 = 1000000.0,
    sliding_window: usize = 512,
    max_seq_len: usize = 4096,
    layer_types: [64]LayerType = [_]LayerType{.sliding_attention} ** 64,

    pub fn loadFromJson(allocator: std.mem.Allocator, path: []const u8) !ModelConfig {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const file_size = (try file.stat()).size;
        const raw_json = try allocator.alloc(u8, file_size);
        defer allocator.free(raw_json);
        _ = try file.readAll(raw_json);

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_json, .{});
        defer parsed.deinit();

        var root = parsed.value;
        if (root.object.get("text_config")) |tc| if (tc == .object) { root = tc; };

        var cfg = ModelConfig{};
        if (root.object.get("vocab_size")) |v| if (v == .integer) { cfg.vocab_size = @intCast(v.integer); };
        if (root.object.get("hidden_size")) |v| if (v == .integer) { cfg.hidden_size = @intCast(v.integer); };
        if (root.object.get("intermediate_size")) |v| if (v == .integer) { cfg.intermediate_size = @intCast(v.integer); };
        if (root.object.get("hidden_size_per_layer_input")) |v| if (v == .integer) { cfg.hidden_size_per_layer_input = @intCast(v.integer); };
        if (root.object.get("num_hidden_layers")) |v| if (v == .integer) { cfg.num_hidden_layers = @intCast(v.integer); };
        if (root.object.get("num_attention_heads")) |v| if (v == .integer) { cfg.num_attention_heads = @intCast(v.integer); };
        if (root.object.get("num_key_value_heads")) |v| if (v == .integer) { cfg.num_key_value_heads = @intCast(v.integer); };
        if (root.object.get("num_global_key_value_heads")) |v| {
            if (v == .integer) cfg.num_global_key_value_heads = @intCast(v.integer) else cfg.num_global_key_value_heads = cfg.num_key_value_heads;
        } else {
            cfg.num_global_key_value_heads = cfg.num_key_value_heads;
        }
        if (root.object.get("head_dim")) |v| if (v == .integer) { cfg.head_dim = @intCast(v.integer); };
        if (root.object.get("global_head_dim")) |v| {
            if (v == .integer) cfg.global_head_dim = @intCast(v.integer) else cfg.global_head_dim = cfg.head_dim;
        } else {
            cfg.global_head_dim = cfg.head_dim;
        }
        if (root.object.get("num_kv_shared_layers")) |v| if (v == .integer) { cfg.num_kv_shared_layers = @intCast(v.integer); };
        if (root.object.get("attention_k_eq_v")) |v| if (v == .bool) { cfg.attention_k_eq_v = v.bool; };
        if (root.object.get("sliding_window")) |v| if (v == .integer) { cfg.sliding_window = @intCast(v.integer); };
        if (root.object.get("rms_norm_eps")) |v| cfg.rms_norm_eps = if (v == .float) @floatCast(v.float) else 1e-6;

        if (root.object.get("layer_types")) |lt| {
            if (lt == .array) {
                for (lt.array.items, 0..) |item, i| {
                    if (i >= 64) break;
                    cfg.layer_types[i] = if (item == .string and std.mem.eql(u8, item.string, "full_attention")) .full_attention else .sliding_attention;
                }
            }
        } else {
            for (0..cfg.num_hidden_layers) |l| cfg.layer_types[l] = if ((l + 1) % 5 == 0) .full_attention else .sliding_attention;
        }
        return cfg;
    }
};

pub const TopKCandidate = struct {
    id: u32 = 0,
    val: f32 = 0.0,
};

pub const ForwardScratch = struct {
    x: []f32,
    prev_x: []f32,
    delta_x: []f32,
    normed_x: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_scores: []f32,
    attn_out: []f32,
    mlp_gate_up: []f32,
    mlp_out: []f32,
    ple_context: []f32,
    ple_buf_1: []f32,
    ple_buf_2: []f32,
    logits: []f32,
    topk_candidates: [64]TopKCandidate = [_]TopKCandidate{.{}} ** 64,
    active_slots: []usize,
    sliding_slots: []usize,
    prev_normed_x: []f32,
    recall_indices: []usize,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !ForwardScratch {
        const max_head = @max(config.head_dim, config.global_head_dim);
        const max_q = config.num_attention_heads * max_head;
        const max_kv = @max(config.num_key_value_heads, config.num_global_key_value_heads) * max_head;
        const total_ple = config.num_hidden_layers * config.hidden_size_per_layer_input;
        const max_slots = config.max_seq_len;

        const scratch = ForwardScratch{
            .x = try allocator.alloc(f32, config.hidden_size),
            .prev_x = try allocator.alloc(f32, config.hidden_size),
            .delta_x = try allocator.alloc(f32, config.hidden_size),
            .normed_x = try allocator.alloc(f32, config.hidden_size),
            .q = try allocator.alloc(f32, max_q),
            .k = try allocator.alloc(f32, max_kv),
            .v = try allocator.alloc(f32, max_kv),
            .attn_scores = try allocator.alloc(f32, max_slots),
            .attn_out = try allocator.alloc(f32, @max(max_q, config.hidden_size)),
            .mlp_gate_up = try allocator.alloc(f32, config.intermediate_size * 2),
            .mlp_out = try allocator.alloc(f32, config.hidden_size),
            .ple_context = try allocator.alloc(f32, @max(total_ple, 1)),
            .ple_buf_1 = try allocator.alloc(f32, @max(config.hidden_size_per_layer_input, 1)),
            .ple_buf_2 = try allocator.alloc(f32, config.hidden_size),
            .logits = try allocator.alloc(f32, config.vocab_size),
            .active_slots = try allocator.alloc(usize, max_slots),
            .sliding_slots = try allocator.alloc(usize, @min(max_slots, 1024)),
            .prev_normed_x = try allocator.alloc(f32, config.hidden_size),
            .recall_indices = try allocator.alloc(usize, MAX_RECALL_SLOTS),
        };
        @memset(scratch.prev_normed_x, 0);
        return scratch;
    }

    pub fn deinit(self: *ForwardScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.sliding_slots);
        allocator.free(self.prev_normed_x);
        allocator.free(self.recall_indices);
        allocator.free(self.x);
        allocator.free(self.prev_x);
        allocator.free(self.delta_x);
        allocator.free(self.normed_x);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.attn_scores);
        allocator.free(self.attn_out);
        allocator.free(self.mlp_gate_up);
        allocator.free(self.mlp_out);
        allocator.free(self.ple_context);
        allocator.free(self.ple_buf_1);
        allocator.free(self.ple_buf_2);
        allocator.free(self.logits);
        allocator.free(self.active_slots);
    }
};
