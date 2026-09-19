const std = @import("std");
const trie = @import("trie.zig");
const tokenizer = @import("tokenizer.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const t0 = std.time.milliTimestamp();
    const tok = try tokenizer.Tokenizer.loadFromJson(allocator, "../gemma-4-12B-it/tokenizer.json");
    const t1 = std.time.milliTimestamp();
    std.debug.print("Loaded JSON in {}ms. Vocabulary size = {}\n", .{ t1 - t0, tok.id_to_token.len });

    var tr = try trie.Trie.init(allocator, 500_000);
    defer tr.deinit();

    const t2 = std.time.milliTimestamp();
    for (tok.id_to_token, 0..) |s, id| {
        if (s.len > 0) try tr.insert(s, @intCast(id));
    }
    const t3 = std.time.milliTimestamp();
    std.debug.print("Built Trie in {}ms. Total nodes = {}\n", .{ t3 - t2, tr.nodes.items.len });

    var kernel_buf: [4096]u8 = undefined;
    const kernel_bytes = try std.fs.cwd().readFile("tui/PROMPT_KERNEL.md", &kernel_buf);
    var prompt_buf: [8192]u8 = undefined;
    const prompt = try std.fmt.bufPrint(&prompt_buf, "<|turn>system\n<|think|>\n{s}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n", .{kernel_bytes});

    // Benchmark Trie encoding
    const t4 = std.time.microTimestamp();
    const tokens = try tok.encode(allocator, prompt, true);
    const t5 = std.time.microTimestamp();
    std.debug.print("Current tokenizer encoded {} tokens in {}us\n", .{ tokens.len, t5 - t4 });

    // Test trie prefix lookups
    const t6 = std.time.microTimestamp();
    var norm = std.ArrayList(u8).init(allocator);
    defer norm.deinit();
    for (prompt) |byte| {
        if (byte == ' ') try norm.appendSlice("\xe2\x96\x81") else try norm.append(byte);
    }
    var pos: usize = 0;
    var count: usize = 0;
    while (pos < norm.items.len) {
        const match = tr.findLongestPrefix(norm.items[pos..]);
        if (match.id != null and match.len > 0) {
            count += 1;
            pos += match.len;
        } else {
            pos += 1;
        }
    }
    const t7 = std.time.microTimestamp();
    std.debug.print("Trie prefix search found {} tokens in {}us ({}ns per token)\n", .{ count, t7 - t6, if (count > 0) @as(f64, @floatFromInt(t7 - t6)) * 1000.0 / @as(f64, @floatFromInt(count)) else 0 });
}
