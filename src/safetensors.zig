const std = @import("std");

pub const DType = enum {
    BF16,
    F16,
    F32,
    I32,
    I16,
    I8,
    U8,
    Unknown,

    pub fn fromString(str: []const u8) DType {
        if (std.mem.eql(u8, str, "BF16")) return .BF16;
        if (std.mem.eql(u8, str, "F16")) return .F16;
        if (std.mem.eql(u8, str, "F32")) return .F32;
        if (std.mem.eql(u8, str, "I32")) return .I32;
        if (std.mem.eql(u8, str, "I16")) return .I16;
        if (std.mem.eql(u8, str, "I8")) return .I8;
        if (std.mem.eql(u8, str, "U8")) return .U8;
        return .Unknown;
    }
};

pub const TensorView = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    data: []const u8,

    pub fn asSlice(self: TensorView, comptime T: type) []const T {
        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0 .. self.data.len / @sizeOf(T)];
    }
};

pub const SafeTensors = struct {
    allocator: std.mem.Allocator,
    mappings: std.ArrayList([]align(std.heap.page_size_min) const u8),
    file_handles: std.ArrayList(std.fs.File),
    tensors: std.StringHashMap(TensorView),

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !SafeTensors {
        var st = SafeTensors{
            .allocator = allocator,
            .mappings = std.ArrayList([]align(std.heap.page_size_min) const u8).init(allocator),
            .file_handles = std.ArrayList(std.fs.File).init(allocator),
            .tensors = std.StringHashMap(TensorView).init(allocator),
        };
        errdefer st.deinit();

        try st.mapAndParseFile(path);
        return st;
    }

    pub fn openDir(allocator: std.mem.Allocator, dir_path: []const u8) !SafeTensors {
        var st = SafeTensors{
            .allocator = allocator,
            .mappings = std.ArrayList([]align(std.heap.page_size_min) const u8).init(allocator),
            .file_handles = std.ArrayList(std.fs.File).init(allocator),
            .tensors = std.StringHashMap(TensorView).init(allocator),
        };
        errdefer st.deinit();

        // 1. Check for single model.safetensors
        var single_path_buf: [512]u8 = undefined;
        const single_path = std.fmt.bufPrint(&single_path_buf, "{s}/model.safetensors", .{dir_path}) catch return error.PathTooLong;
        if (std.fs.cwd().openFile(single_path, .{ .mode = .read_only })) |f| {
            f.close();
            try st.mapAndParseFile(single_path);
            return st;
        } else |_| {}

        // 2. Check for sharded model.safetensors.index.json
        var index_path_buf: [512]u8 = undefined;
        const index_path = std.fmt.bufPrint(&index_path_buf, "{s}/model.safetensors.index.json", .{dir_path}) catch return error.PathTooLong;
        const index_file = try std.fs.cwd().openFile(index_path, .{ .mode = .read_only });
        defer index_file.close();

        const file_size = try index_file.getEndPos();
        const raw_json = try allocator.alloc(u8, file_size);
        defer allocator.free(raw_json);
        _ = try index_file.readAll(raw_json);

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_json, .{});
        defer parsed.deinit();

        const weight_map = parsed.value.object.get("weight_map") orelse return error.InvalidIndexJson;
        var shards = std.StringHashMap(void).init(allocator);
        defer shards.deinit();

        var it = weight_map.object.iterator();
        while (it.next()) |entry| {
            try shards.put(entry.value_ptr.*.string, {});
        }

        var shard_it = shards.iterator();
        while (shard_it.next()) |entry| {
            var shard_path_buf: [512]u8 = undefined;
            const shard_path = try std.fmt.bufPrint(&shard_path_buf, "{s}/{s}", .{ dir_path, entry.key_ptr.* });
            try st.mapAndParseFile(shard_path);
        }

        return st;
    }

    pub fn deinit(self: *SafeTensors) void {
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.shape);
        }
        self.tensors.deinit();

        for (self.mappings.items) |mapped| {
            std.posix.munmap(mapped);
        }
        self.mappings.deinit();

        for (self.file_handles.items) |file| {
            file.close();
        }
        self.file_handles.deinit();
    }

    pub fn adviseDontNeed(self: *SafeTensors) void {
        for (self.mappings.items) |mapped| {
            _ = std.posix.madvise(@constCast(mapped.ptr), mapped.len, std.posix.MADV.DONTNEED) catch {};
        }
    }

    fn mapAndParseFile(self: *SafeTensors, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        errdefer file.close();

        const file_size = try file.getEndPos();
        if (file_size < 8) return error.InvalidSafeTensorsFile;

        const mapped = try std.posix.mmap(null, file_size, std.posix.PROT.READ, .{ .TYPE = .SHARED }, file.handle, 0);
        try self.mappings.append(mapped);
        try self.file_handles.append(file);

        const header_len = std.mem.readInt(u64, mapped[0..8], .little);
        const data_start = 8 + header_len;
        if (mapped.len < data_start) return error.InvalidHeaderLength;

        const header_json = mapped[8..data_start];
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, header_json, .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJsonObject;

        var obj_it = root.object.iterator();
        while (obj_it.next()) |entry| {
            const raw_name = entry.key_ptr.*;
            if (std.mem.eql(u8, raw_name, "__metadata__")) continue;

            const t_obj = entry.value_ptr.*;
            if (t_obj != .object) continue;

            const dtype_val = t_obj.object.get("dtype") orelse continue;
            const shape_val = t_obj.object.get("shape") orelse continue;
            const offsets_val = t_obj.object.get("data_offsets") orelse continue;

            if (offsets_val != .array or offsets_val.array.items.len != 2) continue;

            const offset_begin: usize = @intCast(offsets_val.array.items[0].integer);
            const offset_end: usize = @intCast(offsets_val.array.items[1].integer);
            const byte_start = data_start + offset_begin;
            const byte_end = data_start + offset_end;
            if (byte_end > mapped.len) return error.TensorOffsetOutOfBounds;

            var shape_list = try self.allocator.alloc(usize, shape_val.array.items.len);
            for (shape_val.array.items, 0..) |s, i| shape_list[i] = @intCast(s.integer);

            const owned_name = try self.allocator.dupe(u8, raw_name);
            try self.tensors.put(owned_name, TensorView{
                .name = owned_name,
                .dtype = DType.fromString(dtype_val.string),
                .shape = shape_list,
                .data = mapped[byte_start..byte_end],
            });
        }
    }

    pub fn get(self: *const SafeTensors, name: []const u8) ?TensorView {
        return self.tensors.get(name);
    }
};
