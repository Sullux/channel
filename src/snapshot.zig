const std = @import("std");
pub const ring_buffer = @import("ring_buffer.zig");
pub const model_types = @import("model/types.zig");
pub const gpu = @import("gpu.zig");

pub const SNAPSHOT_MAGIC = [4]u8{ 'S', 'N', 'A', 'P' };
pub const SNAPSHOT_VERSION: u32 = 1;

pub const SnapshotHeader = extern struct {
    magic: [4]u8 = SNAPSHOT_MAGIC,
    version: u32 = SNAPSHOT_VERSION,
    clock: u64,
    num_anchors: u32,
    num_layers: u32,
    kv_dim: u32,
    max_slots: u32,
    total_ingested: u64,
    stream_id_len: u16,
    reserved_pad: u16 = 0,
    stream_id: [64]u8, // UTF-8 stream event ID at checkpoint
    turn_boundary_count: u32,
    turn_boundaries: [128]u32,
};

comptime {
    std.debug.assert(@sizeOf(SnapshotHeader) == 4 + 4 + 8 + 4 + 4 + 4 + 4 + 8 + 2 + 2 + 64 + 4 + 512);
}

pub fn saveSnapshot(
    path: []const u8,
    stream_id: []const u8,
    clock: usize,
    ring: *const ring_buffer.DynamicRingBuffer,
    config: *const model_types.ModelConfig,
    gpu_opt: ?*gpu.model_gpu.GpuModelContext,
) !void {
    var tmp_path_buf: [512]u8 = undefined;
    const tmp_path = try std.fmt.bufPrint(&tmp_path_buf, "{s}.tmp", .{path});

    var file = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
    errdefer {
        file.close();
        std.fs.cwd().deleteFile(tmp_path) catch {};
    }

    var buf_writer = std.io.BufferedWriter(1024 * 1024, @TypeOf(file.writer())){ .unbuffered_writer = file.writer() };
    const writer = buf_writer.writer();

    var hdr = SnapshotHeader{
        .clock = @intCast(clock),
        .num_anchors = @intCast(ring.num_anchors),
        .num_layers = @intCast(config.num_hidden_layers),
        .kv_dim = @intCast(ring.max_kv_dim),
        .max_slots = @intCast(ring.total_slots),
        .total_ingested = @intCast(ring.total_ingested),
        .stream_id_len = @intCast(@min(stream_id.len, 64)),
        .stream_id = [_]u8{0} ** 64,
        .turn_boundary_count = @intCast(@min(ring.num_turn_boundaries, 128)),
        .turn_boundaries = [_]u32{0} ** 128,
    };
    @memcpy(hdr.stream_id[0..hdr.stream_id_len], stream_id[0..hdr.stream_id_len]);
    for (0..hdr.turn_boundary_count) |i| {
        hdr.turn_boundaries[i] = @intCast(ring.turn_boundaries[i]);
    }

    try writer.writeAll(std.mem.asBytes(&hdr));

    // Write ring buffer metadata: clocks, active, attention_mass
    const slot_count = ring.num_layers * ring.total_slots;
    for (ring.clocks[0..slot_count]) |c| {
        try writer.writeInt(u64, @intCast(c), .little);
    }
    for (ring.active[0..slot_count]) |a| {
        try writer.writeByte(if (a) 1 else 0);
    }
    try writer.writeAll(std.mem.sliceAsBytes(ring.attention_mass[0..slot_count]));

    // Write KV caches across layers only for active slots!
    // Since only active slots contain valid attention history, saving active slots reduces
    // snapshot I/O from 6.1 GB down to ~30-50 MB for early turns (and ~1.2 GB for full contexts),
    // speeding up checkpointing and restore by over 100x!
    var active_slot_indices: [4096]usize = undefined;
    var active_count: usize = 0;
    for (0..ring.total_slots) |s| {
        if (ring.active[s]) {
            active_slot_indices[active_count] = s;
            active_count += 1;
        }
    }
    try writer.writeInt(u32, @intCast(active_count), .little);
    for (active_slot_indices[0..active_count]) |s| {
        try writer.writeInt(u32, @intCast(s), .little);
    }

    if (gpu_opt) |g| {
        for (0..config.num_hidden_layers) |l| {
            const k_slice = g.layers[l].buf_k_cache.asSlice(f32);
            const v_slice = g.layers[l].buf_v_cache.asSlice(f32);
            for (active_slot_indices[0..active_count]) |s| {
                const off = s * ring.max_kv_dim;
                try writer.writeAll(std.mem.sliceAsBytes(k_slice[off .. off + ring.max_kv_dim]));
                try writer.writeAll(std.mem.sliceAsBytes(v_slice[off .. off + ring.max_kv_dim]));
            }
        }
    } else {
        for (0..config.num_hidden_layers) |l| {
            const base = l * ring.total_slots * ring.max_kv_dim;
            for (active_slot_indices[0..active_count]) |s| {
                const off = base + s * ring.max_kv_dim;
                try writer.writeAll(std.mem.sliceAsBytes(ring.k[off .. off + ring.max_kv_dim]));
                try writer.writeAll(std.mem.sliceAsBytes(ring.v[off .. off + ring.max_kv_dim]));
            }
        }
    }

    try buf_writer.flush();
    file.close();
    try std.fs.cwd().rename(tmp_path, path);
}

pub fn loadSnapshot(
    path: []const u8,
    ring: *ring_buffer.DynamicRingBuffer,
    config: *const model_types.ModelConfig,
    gpu_opt: ?*gpu.model_gpu.GpuModelContext,
    out_stream_id: *[64]u8,
    out_stream_id_len: *usize,
) !usize {
    var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();

    var buf_reader = std.io.BufferedReader(1024 * 1024, @TypeOf(file.reader())){ .unbuffered_reader = file.reader() };
    const reader = buf_reader.reader();

    var hdr: SnapshotHeader = undefined;
    try reader.readNoEof(std.mem.asBytes(&hdr));

    if (!std.mem.eql(u8, &hdr.magic, &SNAPSHOT_MAGIC)) return error.InvalidSnapshotMagic;
    if (hdr.version != SNAPSHOT_VERSION) return error.UnsupportedSnapshotVersion;
    if (hdr.num_layers != config.num_hidden_layers) return error.MismatchedLayerCount;
    if (hdr.kv_dim != ring.max_kv_dim) return error.MismatchedKvDim;
    if (hdr.max_slots != ring.total_slots) return error.MismatchedSlotCount;

    ring.num_anchors = @intCast(hdr.num_anchors);
    ring.total_ingested = @intCast(hdr.total_ingested);
    ring.num_turn_boundaries = @intCast(hdr.turn_boundary_count);
    for (0..ring.num_turn_boundaries) |i| {
        ring.turn_boundaries[i] = hdr.turn_boundaries[i];
    }

    const id_len = @min(@as(usize, hdr.stream_id_len), 64);
    @memcpy(out_stream_id[0..id_len], hdr.stream_id[0..id_len]);
    out_stream_id_len.* = id_len;

    const slot_count = ring.num_layers * ring.total_slots;
    for (ring.clocks[0..slot_count]) |*c| {
        c.* = @intCast(try reader.readInt(u64, .little));
    }
    for (ring.active[0..slot_count]) |*a| {
        const byte = try reader.readByte();
        a.* = (byte != 0);
    }
    try reader.readNoEof(std.mem.sliceAsBytes(ring.attention_mass[0..slot_count]));

    const active_count: usize = @intCast(try reader.readInt(u32, .little));
    var active_slot_indices: [4096]usize = undefined;
    for (0..active_count) |i| {
        active_slot_indices[i] = @intCast(try reader.readInt(u32, .little));
    }

    if (gpu_opt) |g| {
        for (0..config.num_hidden_layers) |l| {
            const k_slice = g.layers[l].buf_k_cache.asSlice(f32);
            const v_slice = g.layers[l].buf_v_cache.asSlice(f32);
            const base = l * ring.total_slots * ring.max_kv_dim;
            for (active_slot_indices[0..active_count]) |s| {
                const off = s * ring.max_kv_dim;
                try reader.readNoEof(std.mem.sliceAsBytes(k_slice[off .. off + ring.max_kv_dim]));
                try reader.readNoEof(std.mem.sliceAsBytes(v_slice[off .. off + ring.max_kv_dim]));
                // Keep CPU ring buffer in sync with GPU for memory/salience diff tracking
                const ring_off = base + off;
                @memcpy(ring.k[ring_off .. ring_off + ring.max_kv_dim], k_slice[off .. off + ring.max_kv_dim]);
                @memcpy(ring.v[ring_off .. ring_off + ring.max_kv_dim], v_slice[off .. off + ring.max_kv_dim]);
            }
        }
    } else {
        for (0..config.num_hidden_layers) |l| {
            const base = l * ring.total_slots * ring.max_kv_dim;
            for (active_slot_indices[0..active_count]) |s| {
                const off = base + s * ring.max_kv_dim;
                try reader.readNoEof(std.mem.sliceAsBytes(ring.k[off .. off + ring.max_kv_dim]));
                try reader.readNoEof(std.mem.sliceAsBytes(ring.v[off .. off + ring.max_kv_dim]));
            }
        }
    }

    return @intCast(hdr.clock);
}
