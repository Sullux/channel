const std = @import("std");
const protocol = @import("protocol.zig");
const model = @import("model.zig");
const ring_buffer = @import("ring_buffer.zig");
const tokenizer = @import("tokenizer.zig");
const memory = @import("memory.zig");
const storage = @import("storage.zig");
const quiescence = @import("quiescence.zig");
const hippocampus = @import("hippocampus.zig");
const server_queue = @import("server_queue.zig");
const gpu = @import("gpu.zig");
const sampler = @import("sampler.zig");

pub const template_state = @import("server/template_state.zig");
pub const TemplateState = template_state.TemplateState;
pub const syntax_tracker = @import("server/syntax_tracker.zig");
pub const SyntaxTracker = syntax_tracker.SyntaxTracker;
pub const tool_format = @import("server/tool_format.zig");
pub const snapshot_handler = @import("server/snapshot_handler.zig");
pub const autonomic = @import("server/autonomic.zig");
pub const AutonomicProbeResult = autonomic.AutonomicProbeResult;
pub const prefill = @import("server/prefill.zig");
pub const decode = @import("server/decode.zig");
pub const stream_handler = @import("server/stream_handler.zig");

pub const Server = struct {
    allocator: std.mem.Allocator,
    m: *const model.Model,
    ring: *ring_buffer.DynamicRingBuffer,
    tok: *const tokenizer.Tokenizer,
    archive: ?*memory.DiffArchive,
    store: ?*storage.PersistentDiffStore,
    scratch: *model.ForwardScratch,
    thread_pool: ?*std.Thread.Pool,
    config: model.ModelConfig,
    max_tokens: usize,
    thinking_budget: usize = 512,
    top_p: f32 = 0.95,
    temp: f32 = 0.7,
    repeat_last_n: usize = 64,
    sampler: sampler.Sampler,
    q_tracker: quiescence.QuiescenceTracker,
    hippo: ?hippocampus.Hippocampus = null,
    clock: usize = 0,
    is_aborted: std.atomic.Value(bool),
    in_thinking_channel: bool = false,
    gpu_opt: ?*gpu.model_gpu.GpuModelContext,
    last_snapshot_clock: ?usize = null,
    last_yield_token: ?u32 = null,
    turn_open: bool = false,
    template_state: TemplateState = .idle_between_turns,
    out_queue: ?*server_queue.OutboundQueue = null,
    is_saving_snapshot: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thinking_gate: bool = false,
    force_reasoning: bool = false,

    pub const prefillTokens = prefill.prefillTokens;
    pub const handleSetSystem = prefill.handleSetSystem;
    pub const decodeResponse = decode.decodeResponse;
    pub const advanceToken = decode.advanceToken;
    pub const probeAutonomic = autonomic.probeAutonomic;
    pub const shouldBypassThinking = autonomic.shouldBypassThinking;
    pub const handleProbeAutonomic = autonomic.handleProbeAutonomic;
    pub const handleSnapshotSave = snapshot_handler.handleSnapshotSave;
    pub const handleSnapshotLoad = snapshot_handler.handleSnapshotLoad;
    pub const handleStreamInput = stream_handler.handleStreamInput;
    pub const handleResume = stream_handler.handleResume;
    pub const handleToolReturn = stream_handler.handleToolReturn;
    pub const parseTokens = stream_handler.parseTokens;

    pub fn init(allocator: std.mem.Allocator, m: *const model.Model, ring: *ring_buffer.DynamicRingBuffer, tok: *const tokenizer.Tokenizer, archive: ?*memory.DiffArchive, store: ?*storage.PersistentDiffStore, scratch: *model.ForwardScratch, tp: ?*std.Thread.Pool, config: model.ModelConfig, max_tokens: usize, q_thresh: f32, gpu_opt: ?*gpu.model_gpu.GpuModelContext) !Server {
        @memset(scratch.x, 0.0);
        @memset(scratch.logits, 0.0);
        @memset(ring.k, 0.0);
        @memset(ring.v, 0.0);
        if (gpu_opt) |g| {
            @memset(g.buf_logits.asSlice(f32), 0.0);
            @memset(g.buf_x.asSlice(f32), 0.0);
            if (g.batch_prefill_ctx) |bp| {
                @memset(bp.buf_x.asSlice(f32), 0.0);
                @memset(bp.buf_normed_x.asSlice(f32), 0.0);
            }
        }
        var hippo_inst: ?hippocampus.Hippocampus = null;
        if (archive != null or store != null) {
            const kv_dim = @max(config.head_dim, config.global_head_dim) * @max(config.num_key_value_heads, config.num_global_key_value_heads);
            hippo_inst = try hippocampus.Hippocampus.init(allocator, config.hidden_size, 64, 6000, config.num_hidden_layers, kv_dim);
        }
        return .{
            .allocator = allocator,
            .m = m,
            .ring = ring,
            .tok = tok,
            .archive = archive,
            .store = store,
            .scratch = scratch,
            .thread_pool = tp,
            .config = config,
            .max_tokens = max_tokens,
            .thinking_budget = 512,
            .temp = 0.7,
            .top_p = 0.95,
            .repeat_last_n = 64,
            .sampler = sampler.Sampler.init(@intCast(@max(1, std.time.nanoTimestamp())), 0.7, 0.95),
            .q_tracker = quiescence.QuiescenceTracker.init(.{ .enabled = q_thresh > 0.0, .threshold = q_thresh }, config.num_hidden_layers),
            .hippo = hippo_inst,
            .is_aborted = std.atomic.Value(bool).init(false),
            .gpu_opt = gpu_opt,
            .last_snapshot_clock = null,
            .last_yield_token = null,
            .turn_open = false,
            .thinking_gate = false,
            .force_reasoning = false,
            .template_state = .idle_between_turns,
        };
    }

    pub fn deinit(self: *Server) void {
        if (self.hippo) |*h| h.deinit();
    }

    pub inline fn slots(self: *Server) u16 {
        return @intCast(self.ring.getActiveSlots(0, self.clock, self.scratch.active_slots));
    }

    pub inline fn statusFlags(self: *Server) u16 {
        var flags: u16 = 0;
        if (self.ring.isWorkingSetSaturated(0.85, 0.35)) {
            flags |= protocol.STATUS_FLAG_SATURATED;
        }
        return flags;
    }

    pub fn formatGemmaToolResponse(self: *Server, tool_name: []const u8, result_json: []const u8) ![]u8 {
        return tool_format.formatGemmaToolResponse(self.allocator, tool_name, result_json);
    }

    pub fn handleSetConfig(self: *Server, p: []const u8) void {
        if (p.len < 20) return;
        self.thinking_budget = std.mem.readInt(u32, p[0..4], .little);
        self.temp = @bitCast(std.mem.readInt(u32, p[4..8], .little));
        self.top_p = @bitCast(std.mem.readInt(u32, p[8..12], .little));
        self.sampler.temp = self.temp;
        self.sampler.top_p = self.top_p;
        self.q_tracker.config.threshold = @bitCast(std.mem.readInt(u32, p[12..16], .little));
        self.max_tokens = std.mem.readInt(u32, p[16..20], .little);
        if (p.len >= 28) {
            self.sampler.min_p = @bitCast(std.mem.readInt(u32, p[20..24], .little));
            self.sampler.repeat_penalty = @bitCast(std.mem.readInt(u32, p[24..28], .little));
        }
        if (p.len >= 32) {
            self.repeat_last_n = std.mem.readInt(u32, p[28..32], .little);
        }
        if (p.len >= 40) {
            self.sampler.frequency_penalty = @bitCast(std.mem.readInt(u32, p[32..36], .little));
            self.sampler.presence_penalty = @bitCast(std.mem.readInt(u32, p[36..40], .little));
        }
        if (p.len >= 41) {
            self.thinking_gate = (p[40] != 0);
        }
    }

    pub fn handleMemQuery(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
        if (self.archive == null or p.len < 4) {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{});
            writer.flush();
            return;
        }
        const top_k = std.mem.readInt(u16, p[0..2], .little);
        const q_tokens = try self.tok.encode(self.allocator, p[4..], false);
        defer self.allocator.free(q_tokens);
        const q_vec = try self.allocator.alloc(f32, self.config.hidden_size);
        defer self.allocator.free(q_vec);
        if (!model.memory_inject.computeKeywordQueryVector(self.m, q_tokens, q_vec)) {
            try protocol.writeMemResponse(writer, msg_id, 0, 0x01, 0, 0, &.{});
            writer.flush();
            return;
        }
        var indices: [16]usize = undefined;
        const count = self.archive.?.scan(q_vec, @intCast(self.clock), &indices, @min(top_k, 16));
        const rot_dim: usize = if (self.m.layers.len > 0) self.m.layers[0].rotary_dim else self.config.head_dim;
        _ = model.memory_inject.primeSubconsciousMemory(
            self.archive.?, self.ring, self.scratch, q_vec, @intCast(self.clock),
            self.gpu_opt, self.clock, self.config.rope_theta, self.config.head_dim,
            rot_dim, self.config.num_key_value_heads,
        );
        var timestamps: [16]u64 = undefined;
        for (0..count) |i| timestamps[i] = self.archive.?.metas[indices[i]].timestamp;
        try protocol.writeMemResponse(writer, msg_id, @intCast(count), 0x00, 0, 0, timestamps[0..count]);
        writer.flush();
    }

    pub fn handleMemCommit(self: *Server) void {
        if (self.hippo) |*h| {
            const start_clock = if (self.clock >= h.count) self.clock - h.count else 0;
            _ = h.commit(self.archive, self.ring, self.store, start_clock, self.gpu_opt);
        }
    }

    pub fn run(self: *Server, reader: anytype, writer: anytype) !void {
        var in_queue = server_queue.MessageQueue.init(self.allocator);
        defer in_queue.deinit();
        var out_queue = try server_queue.OutboundQueue.init(self.allocator);
        defer out_queue.deinit();

        const WriterThread = struct {
            fn run(q: *server_queue.OutboundQueue, w: @TypeOf(writer)) void {
                var chunk_buf: [16384]u8 = undefined;
                while (true) {
                    const n = q.readChunk(&chunk_buf);
                    if (n == 0) break;
                    w.writeAll(chunk_buf[0..n]) catch break;
                }
            }
        };
        var w_thread = try std.Thread.spawn(.{}, WriterThread.run, .{ &out_queue, writer });
        defer {
            out_queue.close();
            w_thread.join();
        }

        var async_writer = server_queue.AsyncWriter.init(&out_queue, self.allocator);
        defer async_writer.deinit();

        self.out_queue = &out_queue;
        defer {
            while (self.is_saving_snapshot.load(.monotonic)) std.time.sleep(10 * std.time.ns_per_ms);
            self.out_queue = null;
        }

        try protocol.writeStatus(&async_writer, 0, protocol.STATUS_IDLE, 0.0, 0, 0, 0, 0, if (self.gpu_opt != null) 1 else 0, 0);
        async_writer.flush();

        const ReaderThread = struct {
            fn run(q: *server_queue.MessageQueue, r: @TypeOf(reader), aborted: *std.atomic.Value(bool)) void {
                defer q.close();
                while (true) {
                    const hdr = protocol.readHeader(r) catch break;
                    if (hdr.opcode == protocol.OP_ABORT) {
                        aborted.store(true, .seq_cst);
                        continue;
                    }
                    var p: []u8 = &.{};
                    if (hdr.payload_len > 0) {
                        p = q.allocator.alloc(u8, hdr.payload_len) catch break;
                        r.readNoEof(p) catch {
                            q.allocator.free(p);
                            break;
                        };
                    }
                    q.push(.{ .hdr = hdr, .payload = p });
                    if (hdr.opcode == protocol.OP_SHUTDOWN) break;
                }
            }
        };
        var r_thread = try std.Thread.spawn(.{}, ReaderThread.run, .{ &in_queue, reader, &self.is_aborted });
        defer r_thread.join();

        while (true) {
            const frame = in_queue.pop() orelse break;
            const hdr, const p = .{ frame.hdr, frame.payload };
            defer if (p.len > 0) self.allocator.free(p);
            switch (hdr.opcode) {
                protocol.OP_STREAM_INPUT => try self.handleStreamInput(hdr.msg_id, p, &async_writer),
                protocol.OP_RESUME => try self.handleResume(hdr.msg_id, p, &async_writer),
                protocol.OP_TOOL_RETURN => try self.handleToolReturn(hdr.msg_id, p, &async_writer),
                protocol.OP_SET_SYSTEM => try self.handleSetSystem(hdr.msg_id, p, &async_writer),
                protocol.OP_SNAPSHOT_SAVE => try self.handleSnapshotSave(hdr.msg_id, p, &async_writer),
                protocol.OP_SNAPSHOT_LOAD => try self.handleSnapshotLoad(hdr.msg_id, p, &async_writer),
                protocol.OP_PROBE_AUTONOMIC => try self.handleProbeAutonomic(hdr.msg_id, p, &async_writer),
                protocol.OP_SET_CONFIG => self.handleSetConfig(p),
                protocol.OP_MEM_QUERY => try self.handleMemQuery(hdr.msg_id, p, &async_writer),
                protocol.OP_MEM_COMMIT => self.handleMemCommit(),
                protocol.OP_PING => {
                    try protocol.writePong(&async_writer, hdr.msg_id);
                    async_writer.flush();
                },
                protocol.OP_SHUTDOWN => break,
                else => {},
            }
        }
    }
};
