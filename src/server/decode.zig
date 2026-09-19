const std = @import("std");
const protocol = @import("../protocol.zig");
const syntax_tracker = @import("syntax_tracker.zig");
const template_state = @import("template_state.zig");
const Server = @import("../server.zig").Server;

pub fn advanceToken(self: *Server, cur: u32, recent_tokens: []const u32) u32 {
    const has_penalties = (self.sampler.repeat_penalty != 1.0 or self.sampler.frequency_penalty != 0.0 or self.sampler.presence_penalty != 0.0);
    const needs_logits = (self.sampler.temp > 0.0 or has_penalties or self.gpu_opt == null);
    const next_tok = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, needs_logits);
    self.clock += 1;
    if (self.hippo) |*h| {
        const x_vec = if (self.gpu_opt) |g| g.buf_x.asSlice(f32)[0..self.config.hidden_size] else self.scratch.x[0..self.config.hidden_size];
        const now_ms = std.time.milliTimestamp();
        const slot_idx = self.ring.getSlotIndex(self.clock - 1);
        h.stage(x_vec, @intCast(@max(0, now_ms)), 1.0, @intCast(self.config.num_hidden_layers - 1), cur, slot_idx, now_ms);
        if (h.shouldFlush(now_ms, false)) {
            const start_clock = if (self.clock >= h.count) self.clock - h.count else 0;
            _ = h.commit(self.archive, self.ring, self.store, start_clock, self.gpu_opt);
        }
    }
    if (!needs_logits) return next_tok;
    if (self.gpu_opt != null) {
        return self.sampler.sampleTopK(&self.scratch.topk_candidates, recent_tokens);
    }
    return self.sampler.sample(self.scratch.logits, recent_tokens);
}

pub fn decodeResponse(self: *Server, msg_id: u16, first_token: u32, writer: anytype, diff_count: u16, is_gpu: u8) !void {
    var cur = first_token;
    var start: i64 = 0;
    var thinking_count: u32 = 0;
    var response_count: u32 = 0;
    var reason: u8 = protocol.STOP_END_OF_TURN;
    const max_recent: usize = @max(64, self.max_tokens + self.thinking_budget);
    const recent_buf = try self.allocator.alloc(u32, max_recent);
    defer self.allocator.free(recent_buf);
    var recent_count: usize = 0;
    var syntax = syntax_tracker.SyntaxTracker{};
    self.turn_open = true;

    while (true) {
        if (self.is_aborted.load(.monotonic)) {
            reason = protocol.STOP_ABORTED;
            if (self.hippo) |*h| h.markInterrupted();
            break;
        }
        if (cur == self.tok.eos_token_id or cur == template_state.TOK_TURN_CLOSE) {
            self.turn_open = false;
            self.template_state = .idle_between_turns;
            _ = self.m.forwardToken(self.ring, self.scratch, cur, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
            self.clock += 1;
            break;
        }

        if (start == 0) start = std.time.milliTimestamp();

        if (recent_count < max_recent) {
            recent_buf[recent_count] = cur;
            recent_count += 1;
        } else {
            std.mem.copyForwards(u32, recent_buf[0 .. max_recent - 1], recent_buf[1..max_recent]);
            recent_buf[max_recent - 1] = cur;
        }

        const w_start = if (recent_count > self.repeat_last_n) recent_count - self.repeat_last_n else 0;
        const window_tokens = recent_buf[w_start..recent_count];

        if (cur == template_state.TOK_CHANNEL_OPEN) {
            if (self.sampler.suppress_thinking or self.in_thinking_channel) {
                self.in_thinking_channel = false;
                self.sampler.suppress_thinking = true;
                self.sampler.suppress_critique = false;
                self.template_state = .in_response_channel;
                self.ring.markBoundary(self.clock, .response_sentence, 1.0);
                cur = self.advanceToken(template_state.TOK_CHANNEL_CLOSE, window_tokens);
                continue;
            }
            self.in_thinking_channel = true;
            self.sampler.suppress_thinking = false;
            self.template_state = .in_thinking_channel;
            thinking_count = 0;
            self.ring.markBoundary(self.clock, .thought, 0.8);
            var chan_tok = self.advanceToken(cur, window_tokens);
            while (chan_tok != template_state.TOK_CHANNEL_CLOSE and chan_tok != self.tok.eos_token_id) {
                if (chan_tok == template_state.TOK_NEWLINE or chan_tok == template_state.TOK_TOOL_CALL) {
                    self.sampler.suppress_channel_close = true;
                    cur = self.advanceToken(chan_tok, window_tokens);
                    self.sampler.suppress_channel_close = false;
                    break;
                }
                chan_tok = self.advanceToken(chan_tok, window_tokens);
            }
            if (chan_tok == template_state.TOK_CHANNEL_CLOSE or chan_tok == self.tok.eos_token_id) {
                cur = chan_tok;
            }
            continue;
        }
        if (cur == template_state.TOK_CHANNEL_CLOSE) {
            self.in_thinking_channel = false;
            self.sampler.suppress_thinking = true;
            self.sampler.suppress_critique = false;
            self.template_state = .in_response_channel;
            self.ring.markBoundary(self.clock, .response_sentence, 1.0);
            cur = self.advanceToken(cur, window_tokens);
            continue;
        }
        if (cur == 236779 and !self.sampler.suppress_thinking) {
            const peek_tok = self.advanceToken(cur, window_tokens);
            if (peek_tok == template_state.TOK_THOUGHT) {
                self.in_thinking_channel = true;
                self.sampler.suppress_thinking = false;
                self.template_state = .in_thinking_channel;
                thinking_count = 0;
                var next_c = self.advanceToken(peek_tok, window_tokens);
                while (next_c == template_state.TOK_NEWLINE or next_c == template_state.TOK_TOOL_CALL) {
                    self.sampler.suppress_channel_close = true;
                    next_c = self.advanceToken(next_c, window_tokens);
                    self.sampler.suppress_channel_close = false;
                }
                cur = next_c;
                continue;
            } else {
                cur = peek_tok;
            }
        }
        if (cur == 48) {
            var call_buf: [4096]u8 = undefined;
            var call_len: usize = 0;
            var next_tok = self.advanceToken(cur, window_tokens);
            while (next_tok != template_state.TOK_TOOL_CLOSE and next_tok != template_state.TOK_TURN_CLOSE and next_tok != self.tok.eos_token_id) {
                const dec_str = self.tok.decode(next_tok);
                if (call_len + dec_str.len < call_buf.len) {
                    @memcpy(call_buf[call_len .. call_len + dec_str.len], dec_str);
                    call_len += dec_str.len;
                }
                next_tok = self.advanceToken(next_tok, window_tokens);
            }
            const raw_call = call_buf[0..call_len];
            var tool_name: []const u8 = "";
            var args_json: []const u8 = "{}";
            if (std.mem.indexOf(u8, raw_call, ":")) |c_idx| {
                const rest = raw_call[c_idx + 1 ..];
                if (std.mem.indexOf(u8, rest, "{")) |b_idx| {
                    tool_name = std.mem.trim(u8, rest[0..b_idx], " \t\r\n");
                    args_json = std.mem.trim(u8, rest[b_idx..], " \t\r\n");
                } else {
                    tool_name = std.mem.trim(u8, rest, " \t\r\n");
                }
            }
            try protocol.writeToolCall(writer, msg_id, 1, tool_name, args_json);
            writer.flush();
            reason = protocol.STOP_TOOL_CALL;
            self.ring.markBoundary(self.clock, .tool_call, 0.9);
            if (next_tok == template_state.TOK_TOOL_CLOSE) {
                _ = self.m.forwardToken(self.ring, self.scratch, template_state.TOK_TOOL_CLOSE, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
                self.clock += 1;
            }
            break;
        }
        if (cur == template_state.TOK_TURN_OPEN or cur == 98) {
            cur = self.advanceToken(cur, window_tokens);
            continue;
        }

        if (self.in_thinking_channel) {
            thinking_count += 1;
        } else {
            if (response_count >= self.max_tokens) {
                reason = protocol.STOP_MAX_TOKENS;
                break;
            }
            response_count += 1;
        }

        const str = self.tok.decode(cur);
        const opcode = if (self.in_thinking_channel) protocol.OP_STREAM_THOUGHT else protocol.OP_STREAM_CONTENT;
        try protocol.writeToken(writer, msg_id, opcode, cur, @intCast(self.clock), 0xFFFFFFFFFFFF, protocol.TOKEN_TYPE_TEXT, str);
        writer.flush();

        syntax.ingestChunk(str);

        if (syntax.isAtRest()) {
            const is_para_break = (cur == template_state.TOK_TOOL_CALL or (str.len > 0 and std.mem.endsWith(u8, str, "\n\n")));
            const is_sentence_newline = (cur == template_state.TOK_NEWLINE and (
                (recent_count >= 2 and (
                    recent_buf[recent_count - 2] == template_state.TOK_TOOL_CALL or
                    recent_buf[recent_count - 2] == 236761 or
                    recent_buf[recent_count - 2] == 236881 or
                    recent_buf[recent_count - 2] == 236888
                )) or
                (recent_count >= 3 and (
                    recent_buf[recent_count - 3] == 236761 or
                    recent_buf[recent_count - 3] == 236881 or
                    recent_buf[recent_count - 3] == 236888
                ))
            )) or (str.len >= 2 and (std.mem.endsWith(u8, str, ".\n") or std.mem.endsWith(u8, str, "?\n") or std.mem.endsWith(u8, str, "!\n")));

            if (self.in_thinking_channel) {
                const should_yield_thinking = (thinking_count >= 24 and is_para_break) or
                    (thinking_count >= 36 and is_sentence_newline) or
                    (thinking_count >= 64 and (is_para_break or is_sentence_newline));

                if (should_yield_thinking) {
                    reason = protocol.STOP_ELASTIC_YIELD;
                    self.last_yield_token = cur;
                    self.template_state = .soft_yielded;
                    break;
                }
            } else if (response_count >= 10) {
                const top1_val = self.scratch.topk_candidates[0].val;
                const top2_val = self.scratch.topk_candidates[1].val;
                const logit_margin = top1_val - top2_val;
                const high_confidence = logit_margin >= 1.5;

                const should_yield = (response_count >= 16 and is_para_break) or
                    (response_count >= 12 and is_sentence_newline and high_confidence) or
                    (response_count >= 32 and is_sentence_newline) or
                    (response_count >= 48 and (is_para_break or is_sentence_newline));

                if (should_yield) {
                    reason = protocol.STOP_ELASTIC_YIELD;
                    self.last_yield_token = cur;
                    self.template_state = .soft_yielded;
                    self.ring.markBoundary(self.clock, .response_sentence, 1.0);
                    break;
                }
            }
        }

        if (self.in_thinking_channel) {
            self.sampler.suppress_critique = (cur == template_state.TOK_NEWLINE or cur == template_state.TOK_TOOL_CALL or (str.len > 0 and str[str.len - 1] == '\n'));
            const grace_ceiling = self.thinking_budget + 32;
            const is_boundary = (str.len > 0 and (str[str.len - 1] == '\n' or str[str.len - 1] == '.' or str[str.len - 1] == '!' or str[str.len - 1] == '?' or str[str.len - 1] == ':'));

            if ((thinking_count >= self.thinking_budget and is_boundary) or thinking_count >= grace_ceiling) {
                const bridge_text = "\n\nIdentified steps complete. To execute across multiple stages, invoke `plan()`. Otherwise, deliver the final answer.\n";
                const bridge_tokens = try self.tok.encode(self.allocator, bridge_text, false);
                defer self.allocator.free(bridge_tokens);
                for (bridge_tokens) |bt| {
                    _ = self.m.forwardToken(self.ring, self.scratch, bt, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, false);
                    self.clock += 1;
                    const b_str = self.tok.decode(bt);
                    try protocol.writeToken(writer, msg_id, protocol.OP_STREAM_THOUGHT, bt, @intCast(self.clock), 0xFFFFFFFFFFFF, protocol.TOKEN_TYPE_TEXT, b_str);
                }
                writer.flush();
                self.in_thinking_channel = false;
                self.sampler.suppress_thinking = true;
                self.sampler.suppress_critique = false;
                self.template_state = .in_response_channel;
                thinking_count = 0;
                cur = self.advanceToken(template_state.TOK_CHANNEL_CLOSE, window_tokens);
                continue;
            }
        } else {
            self.sampler.suppress_critique = false;
        }

        const total_gen = thinking_count + response_count;
        if (total_gen % 8 == 0) {
            const el = @max(1, std.time.milliTimestamp() - start);
            const dividend = if (total_gen > 1) total_gen - 1 else total_gen;
            const total_budget: u32 = @intCast(self.max_tokens + self.thinking_budget);
            try protocol.writeStatus(writer, msg_id, protocol.STATUS_GENERATING, (@as(f32, @floatFromInt(dividend)) / @as(f32, @floatFromInt(el))) * 1000.0, self.slots(), diff_count, total_gen, total_budget, is_gpu, self.statusFlags());
            writer.flush();
        }
        cur = self.advanceToken(cur, window_tokens);
    }

    if (reason != protocol.STOP_ELASTIC_YIELD) {
        self.ring.markBoundary(self.clock, .turn_end, 1.0);
    }
    const total_gen = thinking_count + response_count;
    const now = std.time.milliTimestamp();
    const elapsed: u32 = @intCast(@max(1, now - (if (start > 0) start else now)));
    const dividend = if (total_gen > 1) total_gen - 1 else total_gen;
    const tok_sec = (@as(f32, @floatFromInt(dividend)) / @as(f32, @floatFromInt(elapsed))) * 1000.0;
    try protocol.writeTurnComplete(writer, msg_id, total_gen, elapsed, tok_sec, reason);
    try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, tok_sec, self.slots(), diff_count, total_gen, total_gen, is_gpu, self.statusFlags());
    writer.flush();
    if (self.hippo) |*h| {
        const start_clock = if (self.clock >= h.count) self.clock - h.count else 0;
        _ = h.commit(self.archive, self.ring, self.store, start_clock, self.gpu_opt);
    }
}
