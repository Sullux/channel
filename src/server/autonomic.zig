const std = @import("std");
const protocol = @import("../protocol.zig");
const template_state = @import("template_state.zig");
const sampler = @import("../sampler.zig");
const Server = @import("../server.zig").Server;

pub const AutonomicProbeResult = struct {
    winning_idx: usize,
    winning_token: u32,
    confidence: f32,
    entropy: f32,
    cost_ms: f32,
    decoded_buf: [128]u8 = undefined,
    decoded_len: usize = 0,

    pub fn decoded(self: *const AutonomicProbeResult) []const u8 {
        return self.decoded_buf[0..self.decoded_len];
    }
};

pub fn probeAutonomic(
    self: *Server,
    msg_id: u16,
    prompt: []const u8,
    candidates: []const u32,
    max_decode_tokens: usize,
    writer: anytype,
) anyerror!AutonomicProbeResult {
    const t_start = std.time.milliTimestamp();

    const framed_prompt = try template_state.formatProbeFrame(self.allocator, self.template_state, prompt);
    defer self.allocator.free(framed_prompt);

    const tokens = try self.tok.encode(self.allocator, framed_prompt, false);
    defer self.allocator.free(tokens);

    if (tokens.len == 0) {
        return .{
            .winning_idx = 0,
            .winning_token = if (candidates.len > 0) candidates[0] else 0,
            .confidence = 0.0,
            .entropy = 0.0,
            .cost_ms = 0.0,
        };
    }

    const saved_clock = self.clock;
    const saved_state = self.template_state;
    defer {
        self.ring.rollbackClock(saved_clock);
        self.clock = saved_clock;
        self.template_state = saved_state;
    }

    const last_tok: u32 = tokens[tokens.len - 1];
    if (tokens.len > 1) {
        _ = try self.prefillTokens(msg_id, tokens[0 .. tokens.len - 1], writer, true);
    }
    _ = self.m.forwardToken(self.ring, self.scratch, last_tok, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, true);
    self.clock += 1;

    var result = AutonomicProbeResult{
        .winning_idx = 0,
        .winning_token = 0,
        .confidence = 0.0,
        .entropy = 0.0,
        .cost_ms = 0.0,
    };

    if (max_decode_tokens <= 1 and candidates.len > 0) {
        const eval_res = if (self.gpu_opt != null)
            self.sampler.evalConstrainedTopK(&self.scratch.topk_candidates, candidates)
        else
            self.sampler.evalConstrained(self.scratch.logits, candidates);

        result.winning_idx = eval_res.winning_idx;
        result.winning_token = eval_res.winning_token;
        result.confidence = eval_res.confidence;
        result.entropy = eval_res.entropy;
    } else {
        var decoded_tokens: [16]u32 = undefined;
        var decoded_count: usize = 0;
        const max_count = @min(max_decode_tokens, decoded_tokens.len);

        var next_tok = if (self.gpu_opt != null)
            self.sampler.sampleTopK(&self.scratch.topk_candidates, null)
        else
            self.sampler.sample(self.scratch.logits, null);

        while (decoded_count < max_count) {
            if (next_tok == 106 or next_tok == 100 or next_tok == 101 or next_tok == 107 or next_tok == 108) break;
            decoded_tokens[decoded_count] = next_tok;
            decoded_count += 1;

            _ = self.m.forwardToken(self.ring, self.scratch, next_tok, self.clock, self.thread_pool, self.archive, &self.q_tracker, self.gpu_opt, true);
            self.clock += 1;

            next_tok = if (self.gpu_opt != null)
                self.sampler.sampleTopK(&self.scratch.topk_candidates, null)
            else
                self.sampler.sample(self.scratch.logits, null);
        }

        var title_len: usize = 0;
        for (decoded_tokens[0..decoded_count]) |dt| {
            const piece = self.tok.decode(dt);
            if (title_len + piece.len > result.decoded_buf.len) break;
            @memcpy(result.decoded_buf[title_len .. title_len + piece.len], piece);
            title_len += piece.len;
        }

        if (title_len > 0) {
            const full_text = result.decoded_buf[0..title_len];
            const trimmed = std.mem.trim(u8, full_text, " \t\r\n\"'");
            var clean_len = trimmed.len;
            if (std.mem.indexOf(u8, trimmed, "\n")) |nl| {
                clean_len = nl;
            }
            const clean_slice = std.mem.trim(u8, trimmed[0..clean_len], " \t\r\n\"'");
            @memcpy(result.decoded_buf[0..clean_slice.len], clean_slice);
            result.decoded_len = clean_slice.len;
        }
    }

    const t_end = std.time.milliTimestamp();
    result.cost_ms = @floatFromInt(t_end - t_start);
    return result;
}

pub fn handleProbeAutonomic(self: *Server, msg_id: u16, p: []const u8, writer: anytype) !void {
    if (p.len < 10) return;
    const max_decode_tokens = std.mem.readInt(u16, p[0..2][0..2], .little);
    const candidate_count = std.mem.readInt(u16, p[2..4][0..2], .little);
    const prompt_len = std.mem.readInt(u16, p[8..10][0..2], .little);
    if (p.len < 10 + prompt_len) return;
    const prompt_str = p[10 .. 10 + prompt_len];

    var offset: usize = 10 + prompt_len;
    var cand_toks: [32]u32 = undefined;
    var cand_count: usize = 0;

    for (0..candidate_count) |_| {
        if (offset + 2 > p.len) break;
        const c_len = std.mem.readInt(u16, p[offset .. offset + 2][0..2], .little);
        offset += 2;
        if (offset + c_len > p.len) break;
        const c_str = p[offset .. offset + c_len];
        offset += c_len;
        if (cand_count < cand_toks.len) {
            const encoded = try self.tok.encode(self.allocator, c_str, false);
            defer self.allocator.free(encoded);
            if (encoded.len > 0) {
                cand_toks[cand_count] = encoded[0];
                cand_count += 1;
            }
        }
    }

    const res = try self.probeAutonomic(msg_id, prompt_str, cand_toks[0..cand_count], max_decode_tokens, writer);
    try protocol.writeAutonomicResult(writer, msg_id, @intCast(res.winning_idx), res.confidence, res.entropy, res.cost_ms, res.decoded());
    writer.flush();
}
