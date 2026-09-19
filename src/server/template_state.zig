const std = @import("std");

pub const TemplateState = enum(u8) {
    idle_between_turns = 0,
    in_user_turn = 1,
    turn_open_model = 2,
    in_thinking_channel = 3,
    in_response_channel = 4,
    soft_yielded = 5,
};

pub const TOK_CHANNEL_OPEN: u32 = 100;
pub const TOK_CHANNEL_CLOSE: u32 = 101;
pub const TOK_TURN_OPEN: u32 = 105;
pub const TOK_TURN_CLOSE: u32 = 106;
pub const TOK_NEWLINE: u32 = 107;
pub const TOK_TOOL_CALL: u32 = 108;
pub const TOK_TOOL_CLOSE: u32 = 49;
pub const TOK_THOUGHT: u32 = 45518;

pub const BYPASS_THOUGHT_TOKENS = [_]u32{
    TOK_CHANNEL_OPEN,
    TOK_THOUGHT,
    TOK_NEWLINE,
    TOK_CHANNEL_CLOSE,
};

/// Enforces Gemma 4 canonical turn boundaries around reflex probes.
/// When the server is idle between turns, wraps the prompt in an ephemeral user/model exchange
/// with thought channel bypassed. If already within an open turn or if prompt carries raw tags,
/// returns a copy of the prompt unmodified.
pub fn formatProbeFrame(
    allocator: std.mem.Allocator,
    state: TemplateState,
    prompt: []const u8,
) ![]u8 {
    if (std.mem.indexOf(u8, prompt, "<|turn>") != null) {
        return allocator.dupe(u8, prompt);
    }

    if (state == .idle_between_turns) {
        var buf = std.ArrayList(u8).init(allocator);
        errdefer buf.deinit();
        const w = buf.writer();

        try w.writeAll("<|turn>user\n");
        try w.writeAll(prompt);
        try w.writeAll("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
        return buf.toOwnedSlice();
    }

    if (state == .soft_yielded or state == .in_thinking_channel) {
        var buf = std.ArrayList(u8).init(allocator);
        errdefer buf.deinit();
        const w = buf.writer();

        try w.writeAll("<channel|>\n<turn|>\n<|turn>user\n");
        try w.writeAll(prompt);
        try w.writeAll("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
        return buf.toOwnedSlice();
    }

    return allocator.dupe(u8, prompt);
}
