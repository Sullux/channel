const std = @import("std");
pub const protocol = @import("protocol.zig");

pub const InboundMsg = struct {
    hdr: protocol.Header,
    payload: []u8,
};

pub const MessageQueue = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    list: std.ArrayList(InboundMsg),
    head: usize = 0,
    closed: bool = false,

    pub fn init(allocator: std.mem.Allocator) MessageQueue {
        return .{
            .allocator = allocator,
            .list = std.ArrayList(InboundMsg).init(allocator),
        };
    }

    pub fn deinit(self: *MessageQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.list.items[self.head..]) |msg| {
            if (msg.payload.len > 0) self.allocator.free(msg.payload);
        }
        self.list.deinit();
    }

    pub fn push(self: *MessageQueue, msg: InboundMsg) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.list.append(msg) catch return;
        self.cond.signal();
    }

    pub fn pop(self: *MessageQueue) ?InboundMsg {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.head >= self.list.items.len and !self.closed) {
            self.cond.wait(&self.mutex);
        }
        if (self.head >= self.list.items.len) return null;
        const msg = self.list.items[self.head];
        self.head += 1;
        if (self.head > 64 and self.head == self.list.items.len) {
            self.list.clearRetainingCapacity();
            self.head = 0;
        }
        return msg;
    }

    pub fn close(self: *MessageQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.closed = true;
        self.cond.broadcast();
    }
};

pub const OutboundQueue = struct {
    allocator: std.mem.Allocator,
    buffer: []u8,
    head: usize = 0,
    tail: usize = 0,
    mutex: std.Thread.Mutex = .{},
    cond_not_empty: std.Thread.Condition = .{},
    cond_not_full: std.Thread.Condition = .{},
    closed: bool = false,

    const CAPACITY: usize = 256 * 1024; // 256 KB preallocated ring buffer

    pub fn init(allocator: std.mem.Allocator) !OutboundQueue {
        const buf = try allocator.alloc(u8, CAPACITY);
        return .{
            .allocator = allocator,
            .buffer = buf,
        };
    }

    pub fn deinit(self: *OutboundQueue) void {
        self.allocator.free(self.buffer);
    }

    pub fn write(self: *OutboundQueue, data: []const u8) void {
        if (data.len == 0) return;
        self.mutex.lock();
        defer self.mutex.unlock();

        var src_off: usize = 0;
        while (src_off < data.len and !self.closed) {
            const used = self.head - self.tail;
            const free_space = self.buffer.len - used;
            if (free_space == 0) {
                self.cond_not_full.wait(&self.mutex);
                continue;
            }
            const chunk_len = @min(data.len - src_off, free_space);
            const head_idx = self.head % self.buffer.len;
            const first_write = @min(chunk_len, self.buffer.len - head_idx);
            @memcpy(self.buffer[head_idx .. head_idx + first_write], data[src_off .. src_off + first_write]);
            if (chunk_len > first_write) {
                const second_write = chunk_len - first_write;
                @memcpy(self.buffer[0..second_write], data[src_off + first_write .. src_off + chunk_len]);
            }
            self.head += chunk_len;
            src_off += chunk_len;
            self.cond_not_empty.signal();
        }
    }

    pub fn readChunk(self: *OutboundQueue, dst: []u8) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.head == self.tail and !self.closed) {
            self.cond_not_empty.wait(&self.mutex);
        }
        if (self.head == self.tail) return 0;

        const available = self.head - self.tail;
        const to_read = @min(dst.len, available);
        const tail_idx = self.tail % self.buffer.len;
        const first_read = @min(to_read, self.buffer.len - tail_idx);
        @memcpy(dst[0..first_read], self.buffer[tail_idx .. tail_idx + first_read]);
        if (to_read > first_read) {
            const second_read = to_read - first_read;
            @memcpy(dst[first_read..to_read], self.buffer[0..second_read]);
        }
        self.tail += to_read;
        self.cond_not_full.signal();
        return to_read;
    }

    pub fn close(self: *OutboundQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.closed = true;
        self.cond_not_empty.broadcast();
        self.cond_not_full.broadcast();
    }
};

pub const AsyncWriter = struct {
    queue: *OutboundQueue,
    buf: [4096]u8 = undefined,
    len: usize = 0,

    pub fn init(queue: *OutboundQueue, allocator: std.mem.Allocator) AsyncWriter {
        _ = allocator;
        return .{ .queue = queue, .len = 0 };
    }

    pub fn deinit(self: *AsyncWriter) void {
        self.flush();
    }

    pub fn writeAll(self: *AsyncWriter, bytes: []const u8) !void {
        if (self.len + bytes.len <= self.buf.len) {
            @memcpy(self.buf[self.len .. self.len + bytes.len], bytes);
            self.len += bytes.len;
        } else {
            self.flush();
            if (bytes.len > self.buf.len) {
                self.queue.write(bytes);
            } else {
                @memcpy(self.buf[0..bytes.len], bytes);
                self.len = bytes.len;
            }
        }
    }

    pub fn writeByte(self: *AsyncWriter, b: u8) !void {
        if (self.len >= self.buf.len) self.flush();
        self.buf[self.len] = b;
        self.len += 1;
    }

    pub fn writeInt(self: *AsyncWriter, comptime T: type, val: T, endian: std.builtin.Endian) !void {
        var bytes: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &bytes, val, endian);
        try self.writeAll(&bytes);
    }

    pub fn flush(self: *AsyncWriter) void {
        if (self.len > 0) {
            self.queue.write(self.buf[0..self.len]);
            self.len = 0;
        }
    }
};

test "server_queue outbound ring buffer" {
    const allocator = std.testing.allocator;
    var queue = try OutboundQueue.init(allocator);
    defer queue.deinit();

    var writer = AsyncWriter.init(&queue, allocator);
    try writer.writeAll("hello world 12345");
    writer.flush();

    var read_buf: [32]u8 = undefined;
    const n = queue.readChunk(&read_buf);
    try std.testing.expectEqual(@as(usize, 17), n);
    try std.testing.expectEqualStrings("hello world 12345", read_buf[0..n]);
}
