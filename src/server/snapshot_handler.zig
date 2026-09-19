const std = @import("std");
const protocol = @import("../protocol.zig");
const snapshot = @import("../snapshot.zig");
const server_queue = @import("../server_queue.zig");
const Server = @import("../server.zig").Server;

const SaveTask = struct {
    server: *Server,
    msg_id: u16,
    path: []u8,
    stream_id: []u8,
    clock: usize,
    slots: u16,
    out_queue: *server_queue.OutboundQueue,

    fn run(t: @This()) void {
        defer {
            t.server.is_saving_snapshot.store(false, .seq_cst);
            t.server.allocator.free(t.path);
            t.server.allocator.free(t.stream_id);
        }
        snapshot.saveSnapshot(t.path, t.stream_id, t.clock, t.server.ring, &t.server.config, t.server.gpu_opt) catch |err| {
            std.log.err("Background snapshot failed: {any}", .{err});
            var err_writer = server_queue.AsyncWriter.init(t.out_queue, t.server.allocator);
            defer err_writer.deinit();
            protocol.writeError(&err_writer, t.msg_id, "Failed to save snapshot") catch {};
            err_writer.flush();
            return;
        };

        t.server.last_snapshot_clock = t.clock;
        var stat_writer = server_queue.AsyncWriter.init(t.out_queue, t.server.allocator);
        defer stat_writer.deinit();
        protocol.writeSnapshotStatus(&stat_writer, t.msg_id, protocol.SNAPSHOT_STATUS_SAVED, @intCast(t.clock), t.slots, t.stream_id) catch {};
        stat_writer.flush();
    }
};

pub fn handleSnapshotSave(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
    if (payload.len < 4) return;
    const path_len = std.mem.readInt(u16, payload[0..2], .little);
    if (payload.len < 2 + path_len + 2) return;
    const snap_path = payload[2 .. 2 + path_len];

    const stream_id_off = 2 + path_len;
    const stream_id_len = std.mem.readInt(u16, payload[stream_id_off .. stream_id_off + 2][0..2], .little);
    const stream_id = if (payload.len >= stream_id_off + 2 + stream_id_len) payload[stream_id_off + 2 .. stream_id_off + 2 + stream_id_len] else "";

    if (self.last_snapshot_clock != null and self.last_snapshot_clock.? == self.clock) {
        try protocol.writeSnapshotStatus(writer, msg_id, protocol.SNAPSHOT_STATUS_EXISTS, @intCast(self.clock), self.slots(), stream_id);
        writer.flush();
        return;
    }

    if (self.is_saving_snapshot.swap(true, .seq_cst)) {
        try protocol.writeSnapshotStatus(writer, msg_id, protocol.SNAPSHOT_STATUS_EXISTS, @intCast(self.clock), self.slots(), stream_id);
        writer.flush();
        return;
    }

    const out_q = self.out_queue orelse {
        self.is_saving_snapshot.store(false, .seq_cst);
        return;
    };

    const owned_path = try self.allocator.dupe(u8, snap_path);
    errdefer self.allocator.free(owned_path);
    const owned_stream_id = try self.allocator.dupe(u8, stream_id);
    errdefer self.allocator.free(owned_stream_id);

    const task = SaveTask{
        .server = self,
        .msg_id = msg_id,
        .path = owned_path,
        .stream_id = owned_stream_id,
        .clock = self.clock,
        .slots = self.slots(),
        .out_queue = out_q,
    };

    const thread = std.Thread.spawn(.{}, SaveTask.run, .{task}) catch |err| {
        self.is_saving_snapshot.store(false, .seq_cst);
        self.allocator.free(owned_path);
        self.allocator.free(owned_stream_id);
        return err;
    };
    thread.detach();
}

pub fn handleSnapshotLoad(self: *Server, msg_id: u16, payload: []const u8, writer: anytype) !void {
    if (payload.len == 0) return;
    const snap_path = payload;

    var restored_id: [64]u8 = undefined;
    var restored_id_len: usize = 0;
    const restored_clock = snapshot.loadSnapshot(snap_path, self.ring, &self.config, self.gpu_opt, &restored_id, &restored_id_len) catch |err| {
        try protocol.writeError(writer, msg_id, "Failed to load snapshot");
        writer.flush();
        return err;
    };

    self.clock = restored_clock;
    self.last_snapshot_clock = restored_clock;
    const stream_id = restored_id[0..restored_id_len];
    try protocol.writeSnapshotStatus(writer, msg_id, protocol.SNAPSHOT_STATUS_LOADED, @intCast(self.clock), self.slots(), stream_id);

    const is_gpu: u8 = if (self.gpu_opt != null) 1 else 0;
    const diff_count: u16 = if (self.archive) |a| @intCast(a.count) else 0;
    try protocol.writeStatus(writer, msg_id, protocol.STATUS_IDLE, 0.0, self.slots(), diff_count, 0, 0, is_gpu, self.statusFlags());
    writer.flush();
}
