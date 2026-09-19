const std = @import("std");
const trie = @import("trie.zig");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    id_to_token: [][]const u8,
    trie: trie.Trie,
    raw_json: []u8,
    parsed: std.json.Parsed(std.json.Value),
    bos_token_id: u32 = 2,
    eos_token_id: u32 = 1,

    pub fn loadFromJson(allocator: std.mem.Allocator, path: []const u8) !Tokenizer {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const file_size = (try file.stat()).size;
        const raw_json = try allocator.alloc(u8, file_size);
        errdefer allocator.free(raw_json);
        _ = try file.readAll(raw_json);

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_json, .{});
        errdefer parsed.deinit();

        const root = parsed.value;
        const model_obj = root.object.get("model") orelse return error.InvalidTokenizerJson;
        const vocab_obj = model_obj.object.get("vocab") orelse return error.InvalidTokenizerJson;

        var tr = try trie.Trie.init(allocator, vocab_obj.object.count() * 3);
        errdefer tr.deinit();

        const id_to_token = try allocator.alloc([]const u8, vocab_obj.object.count());
        @memset(id_to_token, "");

        var iter = vocab_obj.object.iterator();
        while (iter.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const id: u32 = @intCast(entry.value_ptr.*.integer);
            if (id < id_to_token.len) id_to_token[id] = token_str;
            try tr.insert(token_str, id);
        }

        return Tokenizer{
            .allocator = allocator,
            .id_to_token = id_to_token,
            .trie = tr,
            .raw_json = raw_json,
            .parsed = parsed,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.trie.deinit();
        self.allocator.free(self.id_to_token);
        self.parsed.deinit();
        self.allocator.free(self.raw_json);
    }

    pub fn decode(self: *const Tokenizer, token_id: u32) []const u8 {
        if (token_id >= self.id_to_token.len) return "";
        return self.id_to_token[token_id];
    }

    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, add_bos: bool) ![]u32 {
        var token_ids = std.ArrayList(u32).init(allocator);
        errdefer token_ids.deinit();
        if (add_bos) try token_ids.append(self.bos_token_id);
        if (text.len == 0) return token_ids.toOwnedSlice();

        const specials = [_][]const u8{
            "<|turn>", "<turn|>", "<|channel>", "<channel|>",
            "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
            "<|tool>", "<tool|>", "<|think|>",
            "<|\"|>",
            "<|start_recalled_memory|>", "<|end_recalled_memory|>",
            "<|notification>", "<notification|>", "<bos>", "<eos>", "<pad>",
        };

        var pos: usize = 0;
        while (pos < text.len) {
            var next_pos: usize = text.len;
            var next_sp_idx: ?usize = null;

            for (specials, 0..) |sp, idx| {
                if (std.mem.indexOfPos(u8, text, pos, sp)) |found| {
                    if (found < next_pos) {
                        next_pos = found;
                        next_sp_idx = idx;
                    }
                }
            }

            if (next_pos > pos) {
                const chunk = text[pos..next_pos];
                var norm = std.ArrayList(u8).init(allocator);
                defer norm.deinit();
                for (chunk) |byte| {
                    if (byte == ' ') try norm.appendSlice("\xe2\x96\x81") else try norm.append(byte);
                }

                const raw = norm.items;
                var cursor: usize = 0;
                while (cursor < raw.len) {
                    const match = self.trie.findLongestPrefix(raw[cursor..]);
                    if (match.id != null and match.len > 0) {
                        try token_ids.append(match.id.?);
                        cursor += match.len;
                    } else {
                        var hex_buf: [6]u8 = undefined;
                        const hex_str = std.fmt.bufPrint(&hex_buf, "<0x{X:0>2}>", .{raw[cursor]}) catch "<0x00>";
                        const hex_match = self.trie.findLongestPrefix(hex_str);
                        if (hex_match.id) |id| try token_ids.append(id);
                        cursor += 1;
                    }
                }
            }

            if (next_sp_idx) |sp_idx| {
                const sp_str = specials[sp_idx];
                const match = self.trie.findLongestPrefix(sp_str);
                if (match.id) |id| try token_ids.append(id);
                pos = next_pos + sp_str.len;
            } else {
                break;
            }
        }
        return token_ids.toOwnedSlice();
    }
};

test "tokenizer encode and decode" {
    const testing = std.testing;
    var tok = Tokenizer.loadFromJson(testing.allocator, "../gemma-4-E2B/tokenizer.json") catch return;
    defer tok.deinit();
    const tokens = try tok.encode(testing.allocator, "The capital of France", true);
    defer testing.allocator.free(tokens);
    try testing.expect(tokens.len >= 4);
    try testing.expectEqual(tokens[0], 2);
}
