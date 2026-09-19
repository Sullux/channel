const std = @import("std");

pub const MAGIC: u32 = 0x53554C58; // 'SULX'
pub const VERSION: u16 = 1;

pub const OP_STREAM_INPUT: u16 = 0x0001;
pub const OP_ABORT: u16 = 0x0002;
pub const OP_MEM_QUERY: u16 = 0x0003;
pub const OP_SET_CONFIG: u16 = 0x0004;
pub const OP_TOOL_RETURN: u16 = 0x0005;
pub const OP_MEM_COMMIT: u16 = 0x0006;
pub const OP_SET_SYSTEM: u16 = 0x0007;
pub const OP_SNAPSHOT_SAVE: u16 = 0x0008;
pub const OP_SNAPSHOT_LOAD: u16 = 0x0009;
pub const OP_RESUME: u16 = 0x000A;
pub const OP_PING: u16 = 0x000E;
pub const OP_SHUTDOWN: u16 = 0x000F;
pub const OP_PROBE_AUTONOMIC: u16 = 0x0012;

pub const OP_STREAM_CONTENT: u16 = 0x0101;
pub const OP_STREAM_THOUGHT: u16 = 0x0102;
pub const OP_TURN_COMPLETE: u16 = 0x0103;
pub const OP_TOOL_CALL: u16 = 0x0104;
pub const OP_MEM_RESPONSE: u16 = 0x0105;
pub const OP_STATUS: u16 = 0x0106;
pub const OP_SNAPSHOT_STATUS: u16 = 0x0107;
pub const OP_AUTONOMIC_RESULT: u16 = 0x010C;
pub const OP_PONG: u16 = 0x010E;
pub const OP_ERROR: u16 = 0x01FF;

pub const MODE_TEXT: u8 = 0x00;
pub const MODE_TOKENS: u8 = 0x01;
pub const MODE_SOFT_VECTORS: u8 = 0x02;
pub const MODE_AUDIO_PCM: u8 = 0x03;
pub const MODE_RAW_IMAGE: u8 = 0x04;
pub const MODE_ENCODED_IMAGE: u8 = 0x05;
pub const MODE_VIDEO_FRAME: u8 = 0x06;

pub const INPUT_FLAG_NONE: u8 = 0x00;
pub const INPUT_FLAG_DIRECT: u8 = 0x01; // Direct response (bypass thinking)
pub const INPUT_FLAG_REASON: u8 = 0x02; // Explicit thinking pass requested
pub const INPUT_FLAG_RAW: u8 = 0x80;    // Raw stream (bypass server-side turn templating)

pub const RESUME_ACTION_CONTINUE: u8 = 0x00;
pub const RESUME_ACTION_CLOSE_THOUGHT: u8 = 0x01;

pub const TOKEN_TYPE_TEXT: u8 = 0x00;
pub const TOKEN_TYPE_AUDIO: u8 = 0x01;
pub const TOKEN_TYPE_IMAGE: u8 = 0x02;
pub const TOKEN_TYPE_CONTROL: u8 = 0x03;

pub const STOP_END_OF_TURN: u8 = 0x00;
pub const STOP_MAX_TOKENS: u8 = 0x01;
pub const STOP_ABORTED: u8 = 0x02;
pub const STOP_TOOL_CALL: u8 = 0x03;
pub const STOP_ELASTIC_YIELD: u8 = 0x04;

pub const QUERY_KEYWORDS: u8 = 0x00;
pub const QUERY_FULLTEXT: u8 = 0x01;
pub const QUERY_TEMPORAL: u8 = 0x02;
pub const QUERY_PINNED: u8 = 0x03;

pub const SNAPSHOT_STATUS_SAVED: u8 = 0;
pub const SNAPSHOT_STATUS_LOADED: u8 = 1;
pub const SNAPSHOT_STATUS_EXISTS: u8 = 2;

pub const Header = extern struct {
    magic: u32 = MAGIC,
    version: u16 = VERSION,
    msg_id: u16 = 0,
    opcode: u16,
    reserved: u16 = 0,
    payload_len: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(Header) == 16);
}

pub fn readHeader(reader: anytype) !Header {
    var hdr: Header = undefined;
    const bytes = try reader.readBytesNoEof(16);
    hdr = @bitCast(bytes);
    if (hdr.magic != MAGIC) return error.InvalidMagic;
    if (hdr.version != VERSION) return error.UnsupportedVersion;
    return hdr;
}

pub fn writeHeader(writer: anytype, hdr: Header) !void {
    const bytes: [16]u8 = @bitCast(hdr);
    try writer.writeAll(&bytes);
}

pub fn writeToken(writer: anytype, msg_id: u16, opcode: u16, token_id: u32, clock: u64, active_mask: u64, token_type: u8, text: []const u8) !void {
    const text_len: u16 = @intCast(@min(text.len, std.math.maxInt(u16)));
    const payload_len = 4 + 8 + 8 + 1 + 1 + 2 + @as(u32, text_len);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = opcode, .payload_len = payload_len });
    try writer.writeInt(u32, token_id, .little);
    try writer.writeInt(u64, clock, .little);
    try writer.writeInt(u64, active_mask, .little);
    try writer.writeByte(token_type);
    try writer.writeByte(0); // reserved
    try writer.writeInt(u16, text_len, .little);
    if (text_len > 0) try writer.writeAll(text[0..text_len]);
}

pub fn writeTurnComplete(writer: anytype, msg_id: u16, tokens: u32, elapsed_ms: u32, tok_sec: f32, reason: u8) !void {
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_TURN_COMPLETE, .payload_len = 16 });
    try writer.writeInt(u32, tokens, .little);
    try writer.writeInt(u32, elapsed_ms, .little);
    try writer.writeInt(u32, @bitCast(tok_sec), .little);
    try writer.writeByte(reason);
    try writer.writeAll(&[_]u8{ 0, 0, 0 });
}

pub fn writeToolCall(writer: anytype, msg_id: u16, call_id: u16, name: []const u8, args_json: []const u8) !void {
    const name_len: u16 = @intCast(@min(name.len, std.math.maxInt(u16)));
    const payload_len = 2 + 2 + @as(u32, name_len) + @as(u32, @intCast(args_json.len));
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_TOOL_CALL, .payload_len = payload_len });
    try writer.writeInt(u16, call_id, .little);
    try writer.writeInt(u16, name_len, .little);
    if (name_len > 0) try writer.writeAll(name[0..name_len]);
    if (args_json.len > 0) try writer.writeAll(args_json);
}

pub fn writeToolReturn(writer: anytype, msg_id: u16, call_id: u16, status: u16, tool_name: []const u8, result_json: []const u8) !void {
    const name_len: u16 = @intCast(@min(tool_name.len, std.math.maxInt(u16)));
    const payload_len = 2 + 2 + 2 + @as(u32, name_len) + @as(u32, @intCast(result_json.len));
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_TOOL_RETURN, .payload_len = payload_len });
    try writer.writeInt(u16, call_id, .little);
    try writer.writeInt(u16, status, .little);
    try writer.writeInt(u16, name_len, .little);
    if (name_len > 0) try writer.writeAll(tool_name[0..name_len]);
    if (result_json.len > 0) try writer.writeAll(result_json);
}

pub fn writeMemResponse(writer: anytype, msg_id: u16, count: u8, status: u8, cursor: u16, total_tokens: u32, timestamps: []const u64) !void {
    const ts_len: u32 = @intCast(timestamps.len * 8);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_MEM_RESPONSE, .payload_len = 8 + ts_len });
    try writer.writeByte(count);
    try writer.writeByte(status);
    try writer.writeInt(u16, cursor, .little);
    try writer.writeInt(u32, total_tokens, .little);
    for (timestamps) |ts| try writer.writeInt(u64, ts, .little);
}

pub const STATUS_IDLE: u8 = 0;
pub const STATUS_ENCODING: u8 = 1;
pub const STATUS_GENERATING: u8 = 2;
pub const STATUS_MEMORY: u8 = 3;
pub const STATUS_CONSOLIDATING: u8 = 4;

pub const FLAG_GPU: u16 = 1 << 0;
pub const FLAG_THINKING_GATE: u16 = 1 << 1;
pub const STATUS_FLAG_SATURATED: u16 = 0x0001;

pub fn writeStatus(writer: anytype, msg_id: u16, status: u8, tok_sec: f32, active_slots: u16, archived_diffs: u16, current_tok: u32, total_tok: u32, is_gpu: u8, flags: u16) !void {
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_STATUS, .payload_len = 20 });
    try writer.writeByte(status);
    try writer.writeByte(is_gpu);
    try writer.writeInt(u16, active_slots, .little);
    try writer.writeInt(u16, archived_diffs, .little);
    try writer.writeInt(u16, flags, .little);
    try writer.writeInt(u32, @bitCast(tok_sec), .little);
    try writer.writeInt(u32, current_tok, .little);
    try writer.writeInt(u32, total_tok, .little);
}

pub fn writeSnapshotStatus(writer: anytype, msg_id: u16, status: u8, clock: u64, active_slots: u16, stream_id: []const u8) !void {
    const s_len: u16 = @intCast(@min(stream_id.len, 64));
    const payload_len: u32 = 1 + 1 + 2 + 8 + 2 + @as(u32, s_len);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_SNAPSHOT_STATUS, .payload_len = payload_len });
    try writer.writeByte(status);
    try writer.writeByte(0); // reserved
    try writer.writeInt(u16, active_slots, .little);
    try writer.writeInt(u64, clock, .little);
    try writer.writeInt(u16, s_len, .little);
    if (s_len > 0) try writer.writeAll(stream_id[0..s_len]);
}

pub fn writeAutonomicResult(writer: anytype, msg_id: u16, winning_idx: u16, confidence: f32, entropy: f32, cost_ms: f32, text: []const u8) !void {
    const payload_len: u32 = @intCast(2 + 4 + 4 + 4 + 2 + text.len);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_AUTONOMIC_RESULT, .payload_len = payload_len });
    try writer.writeInt(u16, winning_idx, .little);
    try writer.writeInt(u32, @bitCast(confidence), .little);
    try writer.writeInt(u32, @bitCast(entropy), .little);
    try writer.writeInt(u32, @bitCast(cost_ms), .little);
    try writer.writeInt(u16, @intCast(text.len), .little);
    try writer.writeAll(text);
}

pub fn writeError(writer: anytype, msg_id: u16, msg: []const u8) !void {
    const len: u32 = @intCast(msg.len);
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_ERROR, .payload_len = len });
    if (len > 0) try writer.writeAll(msg);
}

pub fn writePong(writer: anytype, msg_id: u16) !void {
    try writeHeader(writer, .{ .msg_id = msg_id, .opcode = OP_PONG, .payload_len = 0 });
}

test "protocol header roundtrip serialization" {
    var buf: [128]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    const hdr = Header{ .msg_id = 42, .opcode = OP_STREAM_INPUT, .payload_len = 100 };
    try writeHeader(stream.writer(), hdr);
    stream.pos = 0;
    const parsed = try readHeader(stream.reader());
    try std.testing.expectEqual(hdr.magic, parsed.magic);
    try std.testing.expectEqual(hdr.version, parsed.version);
    try std.testing.expectEqual(hdr.msg_id, parsed.msg_id);
    try std.testing.expectEqual(hdr.opcode, parsed.opcode);
    try std.testing.expectEqual(hdr.payload_len, parsed.payload_len);
}

test "protocol write token frame" {
    var buf: [128]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try writeToken(stream.writer(), 1, OP_STREAM_CONTENT, 12345, 999, 0xFFFFFFFFFFFF, TOKEN_TYPE_TEXT, "hello");
    stream.pos = 0;
    const parsed_hdr = try readHeader(stream.reader());
    try std.testing.expectEqual(OP_STREAM_CONTENT, parsed_hdr.opcode);
    try std.testing.expectEqual(@as(u32, 24 + 5), parsed_hdr.payload_len);
}

test "protocol write autonomic result frame" {
    var buf: [128]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try writeAutonomicResult(stream.writer(), 1, 3, 0.95, 0.12, 1.45, "hello");
    stream.pos = 0;
    const parsed_hdr = try readHeader(stream.reader());
    try std.testing.expectEqual(OP_AUTONOMIC_RESULT, parsed_hdr.opcode);
    try std.testing.expectEqual(@as(u32, 2 + 4 + 4 + 4 + 2 + 5), parsed_hdr.payload_len);
    const idx = try stream.reader().readInt(u16, .little);
    const conf_u = try stream.reader().readInt(u32, .little);
    const conf: f32 = @bitCast(conf_u);
    try std.testing.expectEqual(@as(u16, 3), idx);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), conf, 0.001);
}
