const std = @import("std");
const protocol = @import("../protocol.zig");
const model = @import("../model.zig");
const gpu = @import("../gpu.zig");
const server_queue = @import("../server_queue.zig");
const template_state = @import("template_state.zig");
const Server = @import("../server.zig").Server;

pub const PrefillProgress = struct {
    w: *server_queue.AsyncWriter,
    msg_id: u16,
    slots: u16,
    diff_count: u16,
    base_tok: u32,
    chunk_tok: u32,
    total_tok: u32,
    is_gpu: u8,
    flags: u16,
    start_time: i64,

    pub fn cb(layer_idx: usize, total_layers: usize, ctx_ptr: ?*anyopaque) void {
        const ptr: *@This() = @ptrCast(@alignCast(ctx_ptr.?));
        const el = @max(1, std.time.milliTimestamp() - ptr.start_time);
        const chunk_prog: u32 = @intCast((layer_idx * ptr.chunk_tok) / total_layers);
        const tok_prog: u32 = @min(ptr.total_tok, ptr.base_tok + chunk_prog);
        const tok_sec = (@as(f32, @floatFromInt(tok_prog)) / @as(f32, @floatFromInt(el))) * 1000.0;
        protocol.writeStatus(ptr.w, ptr.msg_id, protocol.STATUS_ENCODING, tok_sec, ptr.slots, ptr.diff_count, tok_prog, ptr.total_tok, ptr.is_gpu, ptr.flags) catch {};
        ptr.w.flush();
    }
};

pub fn prefillTokens(self: *Server, msg_id: u16, tokens: []const u32, writer: anytype, is_system_only: bool) anyerror!u32 {
    const total_prefill: u32 = @intCast(tokens.len);
    const prefill_start = std.time.milliTimestamp();
    const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
    const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;

    if (self.archive) |a| {
        if (model.memory_inject.computeKeywordQueryVector(self.m, tokens, self.scratch.normed_x)) {
            const rot_dim: usize = if (self.m.layers.len > 0) self.m.layers[0].rotary_dim else self.config.head_dim;
            _ = model.memory_inject.primeSubconsciousMemory(
                a,
                self.ring,
                self.scratch,
                self.scratch.normed_x,
                @intCast(self.clock),
                self.gpu_opt,
                self.clock,
                self.config.rope_theta,
                self.config.head_dim,
                rot_dim,
                self.config.num_key_value_heads,
            );
        }
    }

    try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, 0.0, self.slots(), diff_count, 0, total_prefill, is_gpu, self.statusFlags());
    writer.flush();

    var cur: u32 = 0;
    if (tokens.len > 1 and self.gpu_opt != null and self.gpu_opt.?.batch_prefill_ctx != null) {
        const gmc, const bp = .{ self.gpu_opt.?, self.gpu_opt.?.batch_prefill_ctx.? };
        var off: usize = 0;
        while (off < tokens.len) {
            if (self.is_aborted.load(.monotonic)) break;
            const chunk = tokens[off..@min(tokens.len, off + bp.max_tokens)];
            const is_last = (off + chunk.len == tokens.len);
            const prev_count = self.ring.getPrefillPrevSlots(0, self.clock, chunk.len, self.scratch.active_slots);
            var c_slots = try self.allocator.alloc(u32, prev_count + chunk.len);
            defer self.allocator.free(c_slots);
            for (self.scratch.active_slots[0..prev_count], 0..) |s, j| c_slots[j] = @intCast(s);
            for (chunk, 0..) |_, i| {
                const c = self.clock + i;
                c_slots[prev_count + i] = @intCast(self.ring.getSlotIndex(c));
                for (0..self.config.num_hidden_layers) |l| _ = self.ring.activateSlot(l, c);
            }
            const l_dst = if (is_last and !is_system_only) self.scratch.logits else self.scratch.logits[0..0];
            var p_prog = PrefillProgress{
                .w = writer,
                .msg_id = msg_id,
                .slots = self.slots(),
                .diff_count = diff_count,
                .base_tok = @intCast(off),
                .chunk_tok = @intCast(chunk.len),
                .total_tok = total_prefill,
                .is_gpu = is_gpu,
                .flags = self.statusFlags(),
                .start_time = prefill_start,
            };
            try gpu.batch_dispatch.gpuDispatchPrefillBatch(bp, gmc, &self.config, self.m.layers, chunk, self.m.embed_tokens, c_slots, self.clock, prev_count, l_dst, PrefillProgress.cb, &p_prog);
            if (is_last and !is_system_only) {
                const ids = gmc.buf_topk_ids.asSlice(u32)[0..64];
                const vals = gmc.buf_topk_vals.asSlice(f32)[0..64];
                for (&self.scratch.topk_candidates, ids, vals) |*dst, id, v| {
                    dst.* = .{ .id = id, .val = v };
                }
            }
            self.clock += chunk.len;
            off += chunk.len;
        }
    } else {
        for (tokens, 0..) |t, i| {
            if (self.is_aborted.load(.monotonic)) break;
            cur = self.m.forwardToken(self.ring, self.scratch, t, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, !is_system_only and (i == tokens.len - 1));
            self.clock += 1;
            if ((i + 1) % 16 == 0 or i == tokens.len - 1) {
                const el = @max(1, std.time.milliTimestamp() - prefill_start);
                try protocol.writeStatus(writer, msg_id, protocol.STATUS_ENCODING, (@as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(el))) * 1000.0, self.slots(), diff_count, @intCast(i + 1), total_prefill, is_gpu, self.statusFlags());
                writer.flush();
            }
        }
    }

    if (self.is_aborted.load(.monotonic)) {
        try protocol.writeTurnComplete(writer, msg_id, 0, 0, 0.0, protocol.STOP_ABORTED);
        try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, self.slots(), diff_count, 0, 0, is_gpu, self.statusFlags());
        writer.flush();
        return 0;
    }

    if (!is_system_only and tokens.len > 0) {
        var in_thought = false;
        var has_closed_thought = false;
        var t_idx: usize = tokens.len;
        while (t_idx > 0) {
            t_idx -= 1;
            const tok_id = tokens[t_idx];
            if (tok_id == template_state.TOK_CHANNEL_CLOSE) {
                has_closed_thought = true;
                break;
            }
            if (tok_id == template_state.TOK_TURN_CLOSE) {
                break;
            }
            if (tok_id == template_state.TOK_CHANNEL_OPEN) {
                in_thought = true;
                break;
            }
        }

        if (has_closed_thought) {
            self.in_thinking_channel = false;
            self.sampler.suppress_thinking = true;
            self.template_state = .in_response_channel;
        } else {
            self.in_thinking_channel = in_thought;
            self.sampler.suppress_thinking = false;
            self.template_state = if (in_thought) .in_thinking_channel else .turn_open_model;
        }

        cur = if (self.gpu_opt != null)
            self.sampler.sampleTopK(&self.scratch.topk_candidates, null)
        else
            self.sampler.sample(self.scratch.logits, null);
    }
    return cur;
}

pub fn handleSetSystem(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
    if (payload.len == 0) return;
    self.is_aborted.store(false, .seq_cst);
    self.ring.markBoundary(self.clock, .system, 1.0);

    var formatted_system: []const u8 = payload;
    var parsed_json: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_json) |*p| p.deinit();

    parsed_json = std.json.parseFromSlice(std.json.Value, self.allocator, payload, .{}) catch null;
    var dyn_buf = std.ArrayList(u8).init(self.allocator);
    defer dyn_buf.deinit();

    if (parsed_json) |p| {
        if (p.value == .object) {
            const root = p.value.object;
            try dyn_buf.appendSlice("<|turn>system\n<|think|>\n");
            if (root.get("instructions")) |inst| {
                if (inst == .string) {
                    try dyn_buf.appendSlice(inst.string);
                    try dyn_buf.appendSlice("\n");
                }
            }
            if (root.get("tools")) |tools_val| {
                if (tools_val == .array) {
                    for (tools_val.array.items) |tool_item| {
                        if (tool_item != .object) continue;
                        const t_obj = tool_item.object;
                        const t_name = if (t_obj.get("name")) |n| (if (n == .string) n.string else "") else "";
                        const t_desc = if (t_obj.get("description")) |d| (if (d == .string) d.string else "") else "";
                        try dyn_buf.appendSlice("<|tool>declaration:");
                        try dyn_buf.appendSlice(t_name);
                        try dyn_buf.appendSlice("{description:<|\"|>");
                        try dyn_buf.appendSlice(t_desc);
                        try dyn_buf.appendSlice("<|\"|>");
                        if (t_obj.get("parameters")) |param_val| {
                            if (param_val == .object) {
                                const p_obj = param_val.object;
                                try dyn_buf.appendSlice(",parameters:{");
                                if (p_obj.get("properties")) |props_val| {
                                    if (props_val == .object) {
                                        try dyn_buf.appendSlice("properties:{");
                                        var f_first = false;
                                        var key_list = std.ArrayList([]const u8).init(self.allocator);
                                        defer key_list.deinit();
                                        var it = props_val.object.iterator();
                                        while (it.next()) |entry| try key_list.append(entry.key_ptr.*);
                                        std.mem.sort([]const u8, key_list.items, {}, struct {
                                            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                                                return std.mem.order(u8, a, b) == .lt;
                                            }
                                        }.lessThan);

                                        for (key_list.items) |k| {
                                            const v = props_val.object.get(k).?;
                                            if (f_first) try dyn_buf.appendSlice(",");
                                            f_first = true;
                                            try dyn_buf.appendSlice(k);
                                            try dyn_buf.appendSlice(":{");
                                            if (v == .object) {
                                                const sub = v.object;
                                                if (sub.get("description")) |sd| {
                                                    if (sd == .string) {
                                                        try dyn_buf.appendSlice("description:<|\"|>");
                                                        try dyn_buf.appendSlice(sd.string);
                                                        try dyn_buf.appendSlice("<|\"|>,");
                                                    }
                                                }
                                                const st = if (sub.get("type")) |st_val| (if (st_val == .string) st_val.string else "STRING") else "STRING";
                                                if (std.ascii.eqlIgnoreCase(st, "array")) {
                                                    try dyn_buf.appendSlice("items:{type:<|\"|>STRING<|\"|>},");
                                                }
                                                try dyn_buf.appendSlice("type:<|\"|>");
                                                var st_up: [32]u8 = undefined;
                                                const up_len = @min(st.len, st_up.len);
                                                for (0..up_len) |ui| st_up[ui] = std.ascii.toUpper(st[ui]);
                                                try dyn_buf.appendSlice(st_up[0..up_len]);
                                                try dyn_buf.appendSlice("<|\"|>}");
                                            } else {
                                                try dyn_buf.appendSlice("type:<|\"|>STRING<|\"|>}");
                                            }
                                        }
                                        try dyn_buf.appendSlice("},");
                                    }
                                }
                                if (p_obj.get("required")) |req_val| {
                                    if (req_val == .array and req_val.array.items.len > 0) {
                                        try dyn_buf.appendSlice("required:[");
                                        for (req_val.array.items, 0..) |ri, r_idx| {
                                            if (r_idx > 0) try dyn_buf.appendSlice(",");
                                            try dyn_buf.appendSlice("<|\"|>");
                                            if (ri == .string) try dyn_buf.appendSlice(ri.string);
                                            try dyn_buf.appendSlice("<|\"|>");
                                        }
                                        try dyn_buf.appendSlice("],");
                                    }
                                }
                                try dyn_buf.appendSlice("type:<|\"|>OBJECT<|\"|>}");
                            }
                        }
                        try dyn_buf.appendSlice("}<tool|>");
                    }
                }
            }
            try dyn_buf.appendSlice("<turn|>\n");
            formatted_system = dyn_buf.items;
        }
    }

    const tokens = try self.tok.encode(self.allocator, formatted_system, self.clock == 0);
    defer self.allocator.free(tokens);

    if (self.clock == 0 and tokens.len > 0) {
        self.ring.setNumAnchors(tokens.len);
    }

    _ = try self.prefillTokens(msg_id, tokens, writer, true);
    self.template_state = .idle_between_turns;

    const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
    const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
    try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, self.slots(), diff_count, 0, 0, is_gpu, self.statusFlags());
    writer.flush();
}
