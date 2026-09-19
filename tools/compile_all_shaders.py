#!/usr/bin/env python3
import subprocess, os

shaders = [
    ("gemv_bf16", "GEMV_BF16_SPIRV", "src/gpu/shaders_bf16.zig"),
    ("gemv_q4", "GEMV_Q4_SPIRV", "src/gpu/shaders_q4.zig"),
    ("gemv_q8", "GEMV_Q8_SPIRV", "src/gpu/shaders_q8.zig"),
    ("rmsnorm", "RMSNORM_SPIRV", "src/gpu/shaders_rmsnorm.zig"),
    ("add_rmsnorm", "ADD_RMSNORM_SPIRV", "src/gpu/shaders_add_rmsnorm.zig"),
    ("fused_mlp_bf16", "FUSED_GATE_UP_SWIGLU_BF16_SPIRV", "src/gpu/shaders_fused_bf16.zig"),
    ("fused_mlp_q4", "FUSED_GATE_UP_SWIGLU_SPIRV", "src/gpu/shaders_fused_q4.zig"),
    ("fused_mlp_q8", "FUSED_GATE_UP_SWIGLU_Q8_SPIRV", "src/gpu/shaders_fused_q8.zig"),
    ("fused_swiglu", "FUSED_SWIGLU_SPIRV", "src/gpu/shaders_fused.zig"),
    ("decode_attn", "DECODE_ATTENTION_SPIRV", "src/gpu/shaders_attn.zig"),
    ("argmax", "ARGMAX_SPIRV", "src/gpu/shaders_argmax.zig"),
    ("quiescence_gate", "QUIESCENCE_GATE_SPIRV", "src/gpu/shaders_quiescence.zig"),
    ("qkv_prep", "QKV_PREP_SPIRV", "src/gpu/shaders_qkv_prep.zig"),
    ("qkv_rope", "QKV_ROPE_SPIRV", "src/gpu/shaders_qkv_rope.zig"),
    ("batch_rmsnorm", "BATCH_RMSNORM_SPIRV", "src/gpu/shaders_batch_rmsnorm.zig"),
    ("batch_gemm_q4", "BATCH_GEMM_Q4_SPIRV", "src/gpu/shaders_batch_gemm_q4.zig"),
    ("batch_gemm_q8", "BATCH_GEMM_Q8_SPIRV", "src/gpu/shaders_batch_gemm_q8.zig"),
    ("batch_add_rmsnorm", "BATCH_ADD_RMSNORM_SPIRV", "src/gpu/shaders_batch_add_rmsnorm.zig"),
    ("batch_fused_mlp_q4", "BATCH_FUSED_MLP_Q4_SPIRV", "src/gpu/shaders_batch_fused_mlp_q4.zig"),
    ("batch_fused_mlp_q8", "BATCH_FUSED_MLP_Q8_SPIRV", "src/gpu/shaders_batch_fused_mlp_q8.zig"),
    ("batch_qkv_rope", "BATCH_QKV_ROPE_SPIRV", "src/gpu/shaders_batch_qkv_rope.zig"),
    ("batch_causal_attn", "BATCH_CAUSAL_ATTN_SPIRV", "src/gpu/shaders_batch_causal_attn.zig"),
    ("fused_qkv_q4", "FUSED_QKV_Q4_SPIRV", "src/gpu/shaders_fused_qkv_q4.zig"),
    ("topk_pass1", "TOPK_PASS1_SPIRV", "src/gpu/shaders_topk_pass1.zig"),
    ("topk_pass2", "TOPK_PASS2_SPIRV", "src/gpu/shaders_topk_pass2.zig"),
]

def compile_wgsl(src, dst):
    res = subprocess.run(["npx", "--yes", "naga-wasi-cli", src, dst], capture_output=True)
    if res.returncode != 0:
        print(f"Error compiling {src}: {res.stderr.decode('utf-8')}")
        return False
    return True

def spv_to_zig(spv_path, var_name, zig_path):
    with open(spv_path, "rb") as f:
        data = f.read()
    words = []
    for i in range(0, len(data), 4):
        words.append(int.from_bytes(data[i:i+4], 'little'))
    with open(zig_path, "w") as f:
        f.write(f"pub const {var_name} = [_]u32{{\n")
        for i, word in enumerate(words):
            f.write(f" 0x{word:08x},")
            if (i + 1) % 16 == 0:
                f.write("\n")
        if len(words) % 16 != 0:
            f.write("\n")
        f.write("};\n")
    return len(words)

for name, var_name, zig_path in shaders:
    wgsl_path = f"shaders/{name}.wgsl"
    spv_path = f"shaders/{name}.spv"
    if os.path.exists(wgsl_path):
        if compile_wgsl(wgsl_path, spv_path):
            count = spv_to_zig(spv_path, var_name, zig_path)
            print(f"Compiled {name:15s} -> {zig_path:30s} ({count} words)")
