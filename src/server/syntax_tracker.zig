const std = @import("std");

pub const SyntaxTracker = struct {
    in_code_fence: bool = false,
    in_inline_code: bool = false,
    in_double_quote: bool = false,
    paren_depth: u32 = 0,
    brace_depth: u32 = 0,
    bracket_depth: u32 = 0,

    pub fn reset(self: *SyntaxTracker) void {
        self.in_code_fence = false;
        self.in_inline_code = false;
        self.in_double_quote = false;
        self.paren_depth = 0;
        self.brace_depth = 0;
        self.bracket_depth = 0;
    }

    pub fn ingestChunk(self: *SyntaxTracker, chunk: []const u8) void {
        var i: usize = 0;
        while (i < chunk.len) {
            const ch = chunk[i];
            if (i + 3 <= chunk.len and std.mem.eql(u8, chunk[i .. i + 3], "```")) {
                self.in_code_fence = !self.in_code_fence;
                i += 3;
                continue;
            }
            if (ch == '`' and !self.in_code_fence) {
                self.in_inline_code = !self.in_inline_code;
                i += 1;
                continue;
            }
            if (self.in_code_fence or self.in_inline_code) {
                i += 1;
                continue;
            }

            if (ch == '"') {
                self.in_double_quote = !self.in_double_quote;
            } else if (ch == '(') {
                self.paren_depth += 1;
            } else if (ch == ')' and self.paren_depth > 0) {
                self.paren_depth -= 1;
            } else if (ch == '{') {
                self.brace_depth += 1;
            } else if (ch == '}' and self.brace_depth > 0) {
                self.brace_depth -= 1;
            } else if (ch == '[') {
                self.bracket_depth += 1;
            } else if (ch == ']' and self.bracket_depth > 0) {
                self.bracket_depth -= 1;
            }
            i += 1;
        }
    }

    pub fn isAtRest(self: *const SyntaxTracker) bool {
        return !self.in_code_fence and
            !self.in_inline_code and
            !self.in_double_quote and
            self.paren_depth == 0 and
            self.brace_depth == 0 and
            self.bracket_depth == 0;
    }
};
