#!/usr/bin/env python3
import subprocess, os

def compile_wgsl(wgsl_path, spv_path):
    cmd = ["npx", "--yes", "naga-wasi-cli", wgsl_path, spv_path]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print(f"Error compiling {wgsl_path}: {res.stderr.decode('utf-8')}")
        return False
    return True

def spv_to_u32_array(spv_path):
    with open(spv_path, "rb") as f:
        data = f.read()
    words = []
    for i in range(0, len(data), 4):
        words.append(int.from_bytes(data[i:i+4], 'little'))
    return words

def write_shader_zig(var_name, words, zig_path):
    with open(zig_path, "w") as f:
        f.write(f"pub const {var_name} = [_]u32{{\n")
        for i, word in enumerate(words):
            f.write(f" 0x{word:08x},")
            if (i + 1) % 16 == 0:
                f.write("\n")
        if len(words) % 16 != 0:
            f.write("\n")
        f.write("};\n")

if __name__ == '__main__':
    os.makedirs("shaders", exist_ok=True)
    if compile_wgsl("shaders/gemv_q4.wgsl", "shaders/gemv_q4.spv"):
        words = spv_to_u32_array("shaders/gemv_q4.spv")
        write_shader_zig("GEMV_Q4_SPIRV", words, "src/gpu/shaders_q4.zig")
        print(f"Successfully compiled shaders/gemv_q4.wgsl -> src/gpu/shaders_q4.zig ({len(words)} u32 words)")
