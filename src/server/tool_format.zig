const std = @import("std");

fn sortKeys(a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

pub fn formatGemmaToolArg(allocator: std.mem.Allocator, writer: anytype, val: std.json.Value) anyerror!void {
    switch (val) {
        .null => try writer.writeAll("null"),
        .bool => |b| try writer.writeAll(if (b) "true" else "false"),
        .integer => |i| try writer.print("{d}", .{i}),
        .float => |f| try writer.print("{d}", .{f}),
        .number_string => |s| try writer.writeAll(s),
        .string => |s| try writer.print("<|\"|>{s}<|\"|>", .{s}),
        .array => |arr| {
            try writer.writeByte('[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try writer.writeByte(',');
                try formatGemmaToolArg(allocator, writer, item);
            }
            try writer.writeByte(']');
        },
        .object => |obj| {
            try writer.writeByte('{');
            var key_list = std.ArrayList([]const u8).init(allocator);
            defer key_list.deinit();
            var it = obj.iterator();
            while (it.next()) |entry| try key_list.append(entry.key_ptr.*);
            std.mem.sort([]const u8, key_list.items, {}, struct {
                fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                    return sortKeys(a, b);
                }
            }.lessThan);
            for (key_list.items, 0..) |k, i| {
                if (i > 0) try writer.writeByte(',');
                try writer.print("{s}:", .{k});
                if (obj.get(k)) |sub_val| try formatGemmaToolArg(allocator, writer, sub_val);
            }
            try writer.writeByte('}');
        },
    }
}

pub fn formatGemmaToolResponse(allocator: std.mem.Allocator, tool_name: []const u8, result_json: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    const w = out.writer();

    try w.writeAll("<|tool_response>response:");
    try w.writeAll(tool_name);

    var parsed_json: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_json) |*p| p.deinit();
    parsed_json = std.json.parseFromSlice(std.json.Value, allocator, result_json, .{}) catch null;

    var has_error = false;
    if (parsed_json) |p| {
        if (p.value == .object) {
            if (p.value.object.get("error") != null) has_error = true;
            try w.writeByte('{');
            const obj = p.value.object;
            var key_list = std.ArrayList([]const u8).init(allocator);
            defer key_list.deinit();
            var it = obj.iterator();
            while (it.next()) |entry| try key_list.append(entry.key_ptr.*);
            std.mem.sort([]const u8, key_list.items, {}, struct {
                fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                    return sortKeys(a, b);
                }
            }.lessThan);
            for (key_list.items, 0..) |k, i| {
                if (i > 0) try w.writeByte(',');
                try w.print("{s}:", .{k});
                if (obj.get(k)) |sub_val| try formatGemmaToolArg(allocator, w, sub_val);
            }
            try w.writeByte('}');
        } else {
            try w.writeAll("{value:");
            try formatGemmaToolArg(allocator, w, p.value);
            try w.writeByte('}');
        }
    } else {
        if (std.mem.indexOf(u8, result_json, "error") != null) has_error = true;
        try w.print("{{value:<|\"|>{s}<|\"|>}}", .{result_json});
    }
    try w.writeAll("<tool_response|><|channel>thought\n");
    if (has_error) {
        try w.writeAll("Notice: Tool execution failed. Analyze error and correct tool call:\n");
    }
    return out.toOwnedSlice();
}
