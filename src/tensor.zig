const std = @import("std");

/// Brain Floating Point (bfloat16) representation
pub const bf16 = struct {
    bits: u16,

    pub inline fn fromF32(val: f32) bf16 {
        const u32_bits: u32 = @bitCast(val);
        // Round to nearest even:
        const rounding_bias = 0x7FFF + ((u32_bits >> 16) & 1);
        const rounded = u32_bits + rounding_bias;
        return bf16{ .bits = @truncate(rounded >> 16) };
    }

    pub inline fn toF32(self: bf16) f32 {
        const u32_bits: u32 = @as(u32, self.bits) << 16;
        return @bitCast(u32_bits);
    }
};

/// Half precision (f16) helper conversions
pub inline fn f16ToF32(val: f16) f32 {
    return @floatCast(val);
}

pub inline fn f32ToF16(val: f32) f16 {
    return @floatCast(val);
}

test "bfloat16 roundtrip" {
    const val: f32 = 3.1415926;
    const bf = bf16.fromF32(val);
    const back = bf.toF32();
    try std.testing.expect(std.math.approxEqAbs(f32, val, back, 0.02));

    const zero: f32 = 0.0;
    try std.testing.expectEqual(@as(f32, 0.0), bf16.fromF32(zero).toF32());

    const neg: f32 = -42.5;
    try std.testing.expect(std.math.approxEqAbs(f32, neg, bf16.fromF32(neg).toF32(), 0.1));
}
