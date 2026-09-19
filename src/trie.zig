const std = @import("std");

pub const Trie = struct {
    pub const Node = struct {
        token_id: u32 = 0xFFFFFFFF,
        child_head: u32 = 0,
        next_sibling: u32 = 0,
        byte: u8 = 0,
    };

    nodes: std.ArrayList(Node),
    root_children: [256]u32,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Trie {
        var nodes = std.ArrayList(Node).init(allocator);
        try nodes.ensureTotalCapacity(capacity);
        try nodes.append(.{}); // Root node at index 0
        return Trie{
            .nodes = nodes,
            .root_children = std.mem.zeroes([256]u32),
        };
    }

    pub fn deinit(self: *Trie) void {
        self.nodes.deinit();
    }

    pub fn insert(self: *Trie, key: []const u8, token_id: u32) !void {
        if (key.len == 0) return;
        var curr: u32 = 0;
        const first_byte = key[0];
        if (self.root_children[first_byte] != 0) {
            curr = self.root_children[first_byte];
        } else {
            const new_idx: u32 = @intCast(self.nodes.items.len);
            try self.nodes.append(.{ .byte = first_byte });
            self.root_children[first_byte] = new_idx;
            curr = new_idx;
        }

        for (key[1..]) |b| {
            var child = self.nodes.items[curr].child_head;
            var found_idx: u32 = 0;
            while (child != 0) {
                if (self.nodes.items[child].byte == b) {
                    found_idx = child;
                    break;
                }
                child = self.nodes.items[child].next_sibling;
            }

            if (found_idx != 0) {
                curr = found_idx;
            } else {
                const new_idx: u32 = @intCast(self.nodes.items.len);
                const prev_head = self.nodes.items[curr].child_head;
                try self.nodes.append(.{
                    .byte = b,
                    .next_sibling = prev_head,
                });
                self.nodes.items[curr].child_head = new_idx;
                curr = new_idx;
            }
        }
        self.nodes.items[curr].token_id = token_id;
    }

    pub inline fn findLongestPrefix(self: *const Trie, slice: []const u8) struct { id: ?u32, len: usize } {
        if (slice.len == 0) return .{ .id = null, .len = 0 };
        const first_byte = slice[0];
        var curr = self.root_children[first_byte];
        if (curr == 0) return .{ .id = null, .len = 0 };

        var best_id: ?u32 = if (self.nodes.items[curr].token_id != 0xFFFFFFFF) self.nodes.items[curr].token_id else null;
        var best_len: usize = if (best_id != null) 1 else 0;

        for (slice[1..], 1..) |b, i| {
            var child = self.nodes.items[curr].child_head;
            var found = false;
            while (child != 0) {
                const n = self.nodes.items[child];
                if (n.byte == b) {
                    curr = child;
                    if (n.token_id != 0xFFFFFFFF) {
                        best_id = n.token_id;
                        best_len = i + 1;
                    }
                    found = true;
                    break;
                }
                child = n.next_sibling;
            }
            if (!found) break;
        }

        return .{ .id = best_id, .len = best_len };
    }
};
