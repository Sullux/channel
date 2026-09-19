const std = @import("std");
pub const safetensors = @import("safetensors.zig");
pub const tensor = @import("tensor.zig");
pub const kernels = @import("kernels.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const model = @import("model.zig");
pub const ring_buffer = @import("ring_buffer.zig");
pub const diff = @import("diff.zig");
pub const memory = @import("memory.zig");
pub const storage = @import("storage.zig");
pub const quiescence = @import("quiescence.zig");
pub const vq = @import("vq.zig");
pub const gpu = @import("gpu.zig");
pub const quant = @import("quant.zig");
pub const bench = @import("model/bench.zig");
pub const server = @import("server.zig");
pub const interactive = @import("interactive.zig");


pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var thread_pool: std.Thread.Pool = undefined;
    try thread_pool.init(.{ .allocator = allocator });
    defer thread_pool.deinit();

    const stdout, const stdin = .{ std.io.getStdOut().writer(), std.io.getStdIn().reader() };
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_dir: []const u8 = "../gemma-4-E2B";
    var max_tokens: usize, var num_anchors: usize, var window_size: usize, var num_recall: usize = .{ 128, 32, 512, 96 };
    var memory_enabled = true;
    var quiescence_enabled = false;
    var quiescence_threshold: f32 = 0.0;
    var gpu_enabled = false;
    var bench_mode = false;
    var serve_mode = false;
    var quant_mode: quant.QuantMode = .none;
    var storage_path: ?[]const u8 = null;
    var prompt_buf = std.ArrayList(u8).init(allocator);
    defer prompt_buf.deinit();

    var memory_capacity: usize = 64;
    var arg_idx: usize = 1;
    while (arg_idx < args.len) : (arg_idx += 1) {
        const arg = args[arg_idx];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try stdout.print(
                \\Usage: infer [OPTIONS] [PROMPT...]
                \\Options:
                \\  -m, --model <path>             Model directory (default: ../gemma-4-E2B)
                \\  -n, --max-tokens <N>           Max tokens to generate (default: 128)
                \\      --gpu                      Enable Vulkan GPU compute (BF16)
                \\      --q4, --mixed              Enable GPU Q4_0 acceleration
                \\      --q8                       Enable GPU Q8_0 acceleration
                \\      --quant <q4|q8|none>       Select quantization mode
                \\      --serve                    Run wire-protocol server on STDIN/STDOUT
                \\      --bench                    Run GPU throughput benchmark
                \\      --anchors <N>              Immutable anchor slots (default: 32)
                \\      --window <N>               Sliding window slots (default: 512)
                \\      --recall <N>               Dynamic recall slots (default: 96)
                \\      --memory [<path>]          Enable episodic store (default: .episodic.mem)
                \\      --mem-capacity <N>         Episodic memory capacity (default: 64)
                \\      --no-memory                Disable episodic memory
                \\      --quiescence               Enable quiescence gating
                \\      --quiescence-threshold <F> Set quiescence threshold (default: 0.001)
                \\  -h, --help                     Show this help and exit
                \\
            , .{});
            return;
        } else if ((std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) and arg_idx + 1 < args.len) { model_dir = args[arg_idx + 1]; arg_idx += 1;
        } else if ((std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) and arg_idx + 1 < args.len) { max_tokens = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 128; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--anchors") and arg_idx + 1 < args.len) { num_anchors = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 32; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--window") and arg_idx + 1 < args.len) { window_size = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 512; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--recall") and arg_idx + 1 < args.len) { num_recall = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 96; arg_idx += 1;
        } else if ((std.mem.eql(u8, arg, "--mem-capacity") or std.mem.eql(u8, arg, "--memory-capacity")) and arg_idx + 1 < args.len) { memory_capacity = std.fmt.parseInt(usize, args[arg_idx + 1], 10) catch 64; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--memory") or std.mem.eql(u8, arg, "--storage")) {
            if (arg_idx + 1 < args.len and !std.mem.startsWith(u8, args[arg_idx + 1], "-")) {
                storage_path = args[arg_idx + 1];
                arg_idx += 1;
            } else {
                storage_path = ".episodic.mem";
            }
        } else if (std.mem.eql(u8, arg, "--quiescence-threshold") and arg_idx + 1 < args.len) { quiescence_threshold = std.fmt.parseFloat(f32, args[arg_idx + 1]) catch 0.001; quiescence_enabled = true; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--quiescence")) { quiescence_enabled = true; quiescence_threshold = 0.001;
        } else if (std.mem.eql(u8, arg, "--serve")) { serve_mode = true;
        } else if (std.mem.eql(u8, arg, "--bench")) { bench_mode = true; gpu_enabled = true;
        } else if (std.mem.eql(u8, arg, "--gpu")) { gpu_enabled = true;
        } else if (std.mem.eql(u8, arg, "--q8")) { quant_mode = .q8; gpu_enabled = true;
        } else if (std.mem.eql(u8, arg, "--q4") or std.mem.eql(u8, arg, "--mixed") or std.mem.eql(u8, arg, "--q4-mixed")) { quant_mode = .q4; gpu_enabled = true;
        } else if (std.mem.eql(u8, arg, "--quant") and arg_idx + 1 < args.len) { quant_mode = quant.QuantMode.fromString(args[arg_idx + 1]); gpu_enabled = true; arg_idx += 1;
        } else if (std.mem.eql(u8, arg, "--no-memory")) { memory_enabled = false;
        } else {
            if (prompt_buf.items.len > 0) try prompt_buf.append(' ');
            try prompt_buf.appendSlice(arg);
        }
    }

    num_recall = @min(num_recall, model.types.MAX_RECALL_SLOTS);
    if (!serve_mode) try stdout.print("Loading model from: {s}...\n", .{model_dir});

    var config_path_buf: [512]u8 = undefined;
    const config = try model.ModelConfig.loadFromJson(allocator, try std.fmt.bufPrint(&config_path_buf, "{s}/config.json", .{model_dir}));
    var tok_path_buf: [512]u8 = undefined;
    var tok = try tokenizer.Tokenizer.loadFromJson(allocator, try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{model_dir}));
    defer tok.deinit();
    var st = try safetensors.SafeTensors.openDir(allocator, model_dir);
    defer st.deinit();
    var m = try model.Model.loadFromSafeTensors(allocator, &st, config);
    defer m.deinit();

    var gpu_ctx: ?gpu.context.GpuContext = if (gpu_enabled) gpu.context.GpuContext.init(allocator) catch |err| {
        if (!serve_mode) try stdout.print("GpuContext.init error: {}\n", .{err});
        return err;
    } else null;
    defer if (gpu_ctx) |*gc| gc.deinit();
    var gpu_model_ctx: ?gpu.model_gpu.GpuModelContext = if (gpu_ctx) |*gc| gpu.model_gpu.GpuModelContext.init(allocator, gc, &m, config, quant_mode, quiescence_threshold) catch |err| {
        if (!serve_mode) std.debug.print("GPU init error: {any}\n", .{err});
        return err;
    } else null;
    defer if (gpu_model_ctx) |*gmc| gmc.deinit();
    if (gpu_model_ctx != null) st.adviseDontNeed();
    const gpu_ptr: ?*gpu.model_gpu.GpuModelContext = if (gpu_model_ctx) |*gmc| gmc else null;
    if (gpu_ctx) |gc| {
        const mode_tag = switch (quant_mode) { .none => "BF16", .q8 => "Q8_0", .q4, .mixed => "Q4_0 (Attn:Q8_0/MLP:Q4_0)" };
        if (!serve_mode) try stdout.print("GPU: {s} (UMA Compute, {s})\n", .{ gc.device_name[0..(std.mem.indexOfScalar(u8, &gc.device_name, 0) orelse gc.device_name.len)], mode_tag });
    }
    if (bench_mode and gpu_ptr != null) {
        try bench.runGpuBenchmark(&m, config, gpu_ptr.?, stdout);
        return;
    }

    const max_kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
    var ring = try ring_buffer.DynamicRingBuffer.init(allocator, config.num_hidden_layers, max_kv_dim, num_anchors, window_size, num_recall);
    defer ring.deinit();

    var archive: ?memory.DiffArchive = null;
    var store: ?storage.PersistentDiffStore = null;
    defer if (store) |*s| s.close();
    if (memory_enabled) {
        var arch = try memory.DiffArchive.initWithKV(allocator, config.hidden_size, memory_capacity, config.num_hidden_layers, max_kv_dim, .{});
        if (storage_path) |sp| {
            var s = try storage.PersistentDiffStore.open(sp, memory_capacity, config.num_hidden_layers, config.hidden_size, max_kv_dim, 64);
            _ = s.loadIntoArchive(&arch);
            store = s;
        }
        archive = arch;
    }
    defer if (archive) |*a| {
        if (store) |*s| s.saveFromArchive(a);
        a.deinit();
    };

    var scratch = try model.ForwardScratch.init(allocator, config);
    defer scratch.deinit(allocator);

    if (serve_mode) {
        var srv = try server.Server.init(allocator, &m, &ring, &tok, if (archive) |*a| a else null, if (store) |*s| s else null, &scratch, null, config, max_tokens, quiescence_threshold, gpu_ptr);
        defer srv.deinit();
        try srv.run(stdin, stdout);
        return;
    }

    var q_tracker = quiescence.QuiescenceTracker.init(.{ .enabled = quiescence_enabled }, config.num_hidden_layers);
    const q_ptr: ?*quiescence.QuiescenceTracker = if (quiescence_enabled) &q_tracker else null;
    const memory_ptr: ?*memory.DiffArchive = if (archive) |*a| a else null;

    if (prompt_buf.items.len > 0) {
        var clock: usize = 0;
        try interactive.runInference(&m, &tok, &ring, &scratch, prompt_buf.items, max_tokens, &thread_pool, stdout, allocator, memory_ptr, q_ptr, gpu_ptr, &clock, true);
        try stdout.print("\n", .{});
        return;
    }

    try stdout.print("\n=== Gemma 4 Dynamic Streaming REPL ({s}) ===\nAnchors: {d}, Window: {d}, Recall: {d}, Total Slots: {d}\n\n", .{ model_dir, num_anchors, window_size, num_recall, ring.total_slots });
    var global_clock: usize = 0;
    var line_buf: [4096]u8 = undefined;
    while (true) {
        try stdout.print(">>> ", .{});
        const maybe_line = try stdin.readUntilDelimiterOrEof(&line_buf, '\n');
        const line = maybe_line orelse break;
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;
        try interactive.runInference(&m, &tok, &ring, &scratch, trimmed, max_tokens, &thread_pool, stdout, allocator, memory_ptr, q_ptr, gpu_ptr, &global_clock, false);
        try stdout.print("\n\n", .{});
    }
}

test {
    _ = @import("safetensors.zig"); _ = @import("tensor.zig"); _ = @import("kernels.zig");
    _ = @import("tokenizer.zig"); _ = @import("ring_buffer.zig"); _ = @import("diff.zig");
    _ = @import("memory.zig"); _ = @import("storage.zig"); _ = @import("quiescence.zig");
    _ = @import("vq.zig"); _ = @import("gpu.zig"); _ = @import("quant.zig");
    _ = @import("protocol.zig"); _ = @import("hippocampus.zig");
    _ = @import("sampler.zig");
    _ = @import("model/memory_inject.zig");
    _ = @import("test_memory_lifecycle.zig");
}
