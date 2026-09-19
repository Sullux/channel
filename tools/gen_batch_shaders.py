#!/usr/bin/env python3
import os

batch_gemm_q4_wgsl = """
struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    let t = wgid.y;
    if (row >= pc.M || t >= pc.N) {
        return;
    }
    let lane = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let x_offset = t * pc.K;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    var acc: f32 = 0.0;
    var b = 0u;
    while (b < num_blocks) {
        let blk_off0 = row_word_offset + b * 5u;
        let s0 = bitcast<f32>(W[blk_off0]);
        let packed_word0 = W[blk_off0 + lane_word_idx];
        let nibble0 = (packed_word0 >> nib_shift) & 0x0Fu;
        let w_val0 = (f32(nibble0) - 8.0) * s0;
        let x_val0 = X[x_offset + b * 32u + lane];
        acc = acc + w_val0 * x_val0;

        let blk_off1 = blk_off0 + 5u;
        let s1 = bitcast<f32>(W[blk_off1]);
        let packed_word1 = W[blk_off1 + lane_word_idx];
        let nibble1 = (packed_word1 >> nib_shift) & 0x0Fu;
        let w_val1 = (f32(nibble1) - 8.0) * s1;
        let x_val1 = X[x_offset + (b + 1u) * 32u + lane];
        acc = acc + w_val1 * x_val1;

        let blk_off2 = blk_off0 + 10u;
        let s2 = bitcast<f32>(W[blk_off2]);
        let packed_word2 = W[blk_off2 + lane_word_idx];
        let nibble2 = (packed_word2 >> nib_shift) & 0x0Fu;
        let w_val2 = (f32(nibble2) - 8.0) * s2;
        let x_val2 = X[x_offset + (b + 2u) * 32u + lane];
        acc = acc + w_val2 * x_val2;

        let blk_off3 = blk_off0 + 15u;
        let s3 = bitcast<f32>(W[blk_off3]);
        let packed_word3 = W[blk_off3 + lane_word_idx];
        let nibble3 = (packed_word3 >> nib_shift) & 0x0Fu;
        let w_val3 = (f32(nibble3) - 8.0) * s3;
        let x_val3 = X[x_offset + (b + 3u) * 32u + lane];
        acc = acc + w_val3 * x_val3;

        b = b + 4u;
    }

    sdata[lane] = acc;
    workgroupBarrier();

    if (lane < 16u) { sdata[lane] = sdata[lane] + sdata[lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { sdata[lane] = sdata[lane] + sdata[lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { sdata[lane] = sdata[lane] + sdata[lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { sdata[lane] = sdata[lane] + sdata[lane + 2u]; }
    workgroupBarrier();

    if (lane == 0u) {
        Y[t * pc.M + row] = sdata[0] + sdata[1];
    }
}
"""

batch_gemm_q8_wgsl = """
struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    let t = wgid.y;
    if (row >= pc.M || t >= pc.N) {
        return;
    }
    let lane = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;
    let x_offset = t * pc.K;
    let lane_word_idx = 1u + (lane >> 2u);
    let byte_shift = (lane & 3u) * 8u;

    var acc: f32 = 0.0;
    var b = 0u;
    while (b < num_blocks) {
        let blk_off0 = row_word_offset + b * 9u;
        let s0 = bitcast<f32>(W[blk_off0]);
        let packed_word0 = W[blk_off0 + lane_word_idx];
        let byte_val0 = (packed_word0 >> byte_shift) & 0xFFu;
        let signed_val0 = f32(bitcast<i32>(byte_val0 << 24u) >> 24);
        let w_val0 = signed_val0 * s0;
        let x_val0 = X[x_offset + b * 32u + lane];
        acc = acc + w_val0 * x_val0;

        let blk_off1 = blk_off0 + 9u;
        let s1 = bitcast<f32>(W[blk_off1]);
        let packed_word1 = W[blk_off1 + lane_word_idx];
        let byte_val1 = (packed_word1 >> byte_shift) & 0xFFu;
        let signed_val1 = f32(bitcast<i32>(byte_val1 << 24u) >> 24);
        let w_val1 = signed_val1 * s1;
        let x_val1 = X[x_offset + (b + 1u) * 32u + lane];
        acc = acc + w_val1 * x_val1;

        b = b + 2u;
    }

    sdata[lane] = acc;
    workgroupBarrier();

    if (lane < 16u) { sdata[lane] = sdata[lane] + sdata[lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { sdata[lane] = sdata[lane] + sdata[lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { sdata[lane] = sdata[lane] + sdata[lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { sdata[lane] = sdata[lane] + sdata[lane + 2u]; }
    workgroupBarrier();

    if (lane == 0u) {
        Y[t * pc.M + row] = sdata[0] + sdata[1];
    }
}
"""

batch_add_rmsnorm_wgsl = """
struct PushConstants {
    H: u32,
    eps: f32,
    scalar: f32,
    N: u32,
};

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> R: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Normed_X: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_sum_sq: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    if (row >= pc.N) {
        return;
    }
    let lane = lid.x;
    let H = pc.H;
    let offset = row * H;

    var sum_sq: f32 = 0.0;
    var i = lane;
    while (i < H) {
        let sum = (X[offset + i] + R[offset + i]) * pc.scalar;
        X[offset + i] = sum;
        sum_sq = sum_sq + sum * sum;
        i = i + 64u;
    }
    s_sum_sq[lane] = sum_sq;
    workgroupBarrier();

    if (lane < 32u) { s_sum_sq[lane] = s_sum_sq[lane] + s_sum_sq[lane + 32u]; }
    workgroupBarrier();
    if (lane < 16u) { s_sum_sq[lane] = s_sum_sq[lane] + s_sum_sq[lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { s_sum_sq[lane] = s_sum_sq[lane] + s_sum_sq[lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { s_sum_sq[lane] = s_sum_sq[lane] + s_sum_sq[lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { s_sum_sq[lane] = s_sum_sq[lane] + s_sum_sq[lane + 2u]; }
    workgroupBarrier();

    if (lane == 0u) {
        let mean_sq = (s_sum_sq[0] + s_sum_sq[1]) / f32(H);
        s_sum_sq[0] = 1.0 / sqrt(mean_sq + pc.eps);
    }
    workgroupBarrier();

    let inv_rms = s_sum_sq[0];
    i = lane;
    while (i < H) {
        Normed_X[offset + i] = X[offset + i] * inv_rms * W[i];
        i = i + 64u;
    }
}
"""

batch_fused_mlp_q4_wgsl = """
struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_gate: array<f32, 32>;
var<workgroup> sdata_up: array<f32, 32>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    let t = wgid.y;
    if (row >= pc.M || t >= pc.N) {
        return;
    }
    let lane = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let x_offset = t * pc.K;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;
    var b = 0u;
    while (b < num_blocks) {
        let blk_off0 = row_word_offset + b * 5u;
        let gs0 = bitcast<f32>(W_gate[blk_off0]);
        let gp0 = W_gate[blk_off0 + lane_word_idx];
        let gn0 = (gp0 >> nib_shift) & 0x0Fu;
        let x0 = X[x_offset + b * 32u + lane];
        gate_acc = gate_acc + (f32(gn0) - 8.0) * gs0 * x0;
        let us0 = bitcast<f32>(W_up[blk_off0]);
        let up0 = W_up[blk_off0 + lane_word_idx];
        let un0 = (up0 >> nib_shift) & 0x0Fu;
        up_acc = up_acc + (f32(un0) - 8.0) * us0 * x0;

        let blk_off1 = blk_off0 + 5u;
        let gs1 = bitcast<f32>(W_gate[blk_off1]);
        let gp1 = W_gate[blk_off1 + lane_word_idx];
        let gn1 = (gp1 >> nib_shift) & 0x0Fu;
        let x1 = X[x_offset + (b + 1u) * 32u + lane];
        gate_acc = gate_acc + (f32(gn1) - 8.0) * gs1 * x1;
        let us1 = bitcast<f32>(W_up[blk_off1]);
        let up1 = W_up[blk_off1 + lane_word_idx];
        let un1 = (up1 >> nib_shift) & 0x0Fu;
        up_acc = up_acc + (f32(un1) - 8.0) * us1 * x1;

        b = b + 2u;
    }

    sdata_gate[lane] = gate_acc;
    sdata_up[lane] = up_acc;
    workgroupBarrier();

    if (lane < 16u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 16u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 8u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 4u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 2u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 2u];
    }
    workgroupBarrier();

    if (lane == 0u) {
        let g_final = sdata_gate[0] + sdata_gate[1];
        let u_final = sdata_up[0] + sdata_up[1];
        Y[t * pc.M + row] = gelu_tanh(g_final) * u_final;
    }
}
"""

batch_qkv_rope_wgsl = """
struct PushConstants {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    k_eq_v: u32,
    rope_theta: f32,
    eps: f32,
    N: u32,
};

@group(0) @binding(0) var<storage, read> Q_in: array<f32>;
@group(0) @binding(1) var<storage, read> K_in: array<f32>;
@group(0) @binding(2) var<storage, read> V_in: array<f32>;
@group(0) @binding(3) var<storage, read> Q_norm_w: array<f32>;
@group(0) @binding(4) var<storage, read> K_norm_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> Q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> K_cache: array<f32>;
@group(0) @binding(7) var<storage, read_write> V_cache: array<f32>;
@group(0) @binding(8) var<storage, read> Slots: array<u32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_sum_sq: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let head_idx = wgid.x;
    let t = wgid.y;
    if (t >= pc.N) {
        return;
    }
    let slot_idx = Slots[t];
    let clock = t;
    let lane = lid.x;
    let D = pc.head_dim;
    let rot_D = pc.rotary_dim;
    let q_dim = pc.num_q_heads * D;
    let kv_dim = pc.num_kv_heads * D;
    let is_q = (head_idx < pc.num_q_heads);

    if (is_q) {
        let q_head = head_idx;
        let head_offset = t * q_dim + q_head * D;

        var sum_sq: f32 = 0.0;
        var d = lane;
        while (d < D) {
            let v = Q_in[head_offset + d];
            sum_sq = sum_sq + v * v;
            d = d + 32u;
        }
        s_sum_sq[lane] = sum_sq;
        workgroupBarrier();

        if (lane == 0u) {
            var total: f32 = 0.0;
            for (var i = 0u; i < 32u; i = i + 1u) { total = total + s_sum_sq[i]; }
            s_sum_sq[0] = 1.0 / sqrt(total / f32(D) + pc.eps);
        }
        workgroupBarrier();
        let inv_rms = s_sum_sq[0];

        let half_rot = rot_D / 2u;
        d = lane;
        while (d < half_rot) {
            let idx0 = d;
            let idx1 = d + half_rot;
            let w0 = Q_norm_w[idx0];
            let w1 = Q_norm_w[idx1];
            let v0 = Q_in[head_offset + idx0] * inv_rms * w0;
            let v1 = Q_in[head_offset + idx1] * inv_rms * w1;

            let freq_exp = (2.0 * f32(d)) / f32(rot_D);
            let freq = 1.0 / pow(pc.rope_theta, freq_exp);
            let angle = f32(clock) * freq;
            let cos_a = cos(angle);
            let sin_a = sin(angle);

            Q_out[head_offset + idx0] = v0 * cos_a - v1 * sin_a;
            Q_out[head_offset + idx1] = v0 * sin_a + v1 * cos_a;
            d = d + 32u;
        }

        d = rot_D + lane;
        while (d < D) {
            let w = Q_norm_w[d];
            Q_out[head_offset + d] = Q_in[head_offset + d] * inv_rms * w;
            d = d + 32u;
        }
    } else {
        let kv_head = head_idx - pc.num_q_heads;
        if (kv_head >= pc.num_kv_heads) {
            return;
        }
        let in_head_offset = t * kv_dim + kv_head * D;
        let cache_head_offset = slot_idx * kv_dim + kv_head * D;

        var sum_sq: f32 = 0.0;
        var d = lane;
        while (d < D) {
            let v = K_in[in_head_offset + d];
            sum_sq = sum_sq + v * v;
            d = d + 32u;
        }
        s_sum_sq[lane] = sum_sq;
        workgroupBarrier();

        if (lane == 0u) {
            var total: f32 = 0.0;
            for (var i = 0u; i < 32u; i = i + 1u) { total = total + s_sum_sq[i]; }
            s_sum_sq[0] = 1.0 / sqrt(total / f32(D) + pc.eps);
        }
        workgroupBarrier();
        let inv_rms = s_sum_sq[0];

        let half_rot = rot_D / 2u;
        d = lane;
        while (d < half_rot) {
            let idx0 = d;
            let idx1 = d + half_rot;
            let w0 = K_norm_w[idx0];
            let w1 = K_norm_w[idx1];
            let v0 = K_in[in_head_offset + idx0] * inv_rms * w0;
            let v1 = K_in[in_head_offset + idx1] * inv_rms * w1;

            let freq_exp = (2.0 * f32(d)) / f32(rot_D);
            let freq = 1.0 / pow(pc.rope_theta, freq_exp);
            let angle = f32(clock) * freq;
            let cos_a = cos(angle);
            let sin_a = sin(angle);

            K_cache[cache_head_offset + idx0] = v0 * cos_a - v1 * sin_a;
            K_cache[cache_head_offset + idx1] = v0 * sin_a + v1 * cos_a;
            d = d + 32u;
        }

        d = rot_D + lane;
        while (d < D) {
            let w = K_norm_w[d];
            K_cache[cache_head_offset + d] = K_in[in_head_offset + d] * inv_rms * w;
            d = d + 32u;
        }

        var sum_sq_v: f32 = 0.0;
        d = lane;
        while (d < D) {
            let v_raw = select(V_in[in_head_offset + d], K_in[in_head_offset + d], pc.k_eq_v == 1u);
            sum_sq_v = sum_sq_v + v_raw * v_raw;
            d = d + 32u;
        }
        s_sum_sq[lane] = sum_sq_v;
        workgroupBarrier();

        if (lane == 0u) {
            var total_v: f32 = 0.0;
            for (var i = 0u; i < 32u; i = i + 1u) { total_v = total_v + s_sum_sq[i]; }
            s_sum_sq[0] = 1.0 / sqrt(total_v / f32(D) + pc.eps);
        }
        workgroupBarrier();
        let inv_rms_v = s_sum_sq[0];

        d = lane;
        while (d < D) {
            let v_raw = select(V_in[in_head_offset + d], K_in[in_head_offset + d], pc.k_eq_v == 1u);
            V_cache[cache_head_offset + d] = v_raw * inv_rms_v;
            d = d + 32u;
        }
    }
}
"""

batch_causal_attn_wgsl = """
struct PushConstants {
    head_dim: u32,
    kv_dim: u32,
    gqa_ratio: u32,
    inv_sqrt_dim: f32,
    num_q_heads: u32,
    N: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K_cache: array<f32>;
@group(0) @binding(2) var<storage, read> V_cache: array<f32>;
@group(0) @binding(3) var<storage, read> Slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> Attn_out: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_dot: array<f32, 32>;
var<workgroup> s_scores: array<f32, 1024>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let q_head = wgid.x;
    let t = wgid.y;
    if (q_head >= pc.num_q_heads || t >= pc.N) {
        return;
    }
    let lane = lid.x;
    let D = pc.head_dim;
    let kv_h = q_head / pc.gqa_ratio;
    let q_dim = pc.num_q_heads * D;
    let q_offset = t * q_dim + q_head * D;

    let S = t + 1u;

    for (var slot_i = 0u; slot_i < S; slot_i = slot_i + 1u) {
        let physical_slot = Slots[slot_i];
        let kv_offset = physical_slot * pc.kv_dim + kv_h * D;

        var dot: f32 = 0.0;
        var d = lane;
        while (d < D) {
            dot = dot + Q[q_offset + d] * K_cache[kv_offset + d]
                      + Q[q_offset + d + 32u] * K_cache[kv_offset + d + 32u]
                      + Q[q_offset + d + 64u] * K_cache[kv_offset + d + 64u]
                      + Q[q_offset + d + 96u] * K_cache[kv_offset + d + 96u];
            d = d + 128u;
        }

        sdata_dot[lane] = dot;
        workgroupBarrier();

        if (lane < 16u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 16u]; }
        workgroupBarrier();
        if (lane < 8u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 8u]; }
        workgroupBarrier();
        if (lane < 4u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 4u]; }
        workgroupBarrier();
        if (lane < 2u) { sdata_dot[lane] = sdata_dot[lane] + sdata_dot[lane + 2u]; }
        workgroupBarrier();

        if (lane == 0u) {
            s_scores[slot_i] = (sdata_dot[0] + sdata_dot[1]) * pc.inv_sqrt_dim;
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        var max_val: f32 = -1e9;
        for (var i = 0u; i < S; i = i + 1u) {
            if (s_scores[i] > max_val) { max_val = s_scores[i]; }
        }
        var sum_exp: f32 = 0.0;
        for (var i = 0u; i < S; i = i + 1u) {
            let exp_val = exp(s_scores[i] - max_val);
            s_scores[i] = exp_val;
            sum_exp = sum_exp + exp_val;
        }
        let inv_sum = 1.0 / sum_exp;
        for (var i = 0u; i < S; i = i + 1u) {
            s_scores[i] = s_scores[i] * inv_sum;
        }
    }
    workgroupBarrier();

    var d = lane;
    while (d < D) {
        var out_val0: f32 = 0.0;
        var out_val1: f32 = 0.0;
        var out_val2: f32 = 0.0;
        var out_val3: f32 = 0.0;
        for (var slot_i = 0u; slot_i < S; slot_i = slot_i + 1u) {
            let physical_slot = Slots[slot_i];
            let kv_offset = physical_slot * pc.kv_dim + kv_h * D;
            let sc = s_scores[slot_i];
            out_val0 = out_val0 + sc * V_cache[kv_offset + d];
            out_val1 = out_val1 + sc * V_cache[kv_offset + d + 32u];
            out_val2 = out_val2 + sc * V_cache[kv_offset + d + 64u];
            out_val3 = out_val3 + sc * V_cache[kv_offset + d + 96u];
        }
        Attn_out[q_offset + d] = out_val0;
        Attn_out[q_offset + d + 32u] = out_val1;
        Attn_out[q_offset + d + 64u] = out_val2;
        Attn_out[q_offset + d + 96u] = out_val3;
        d = d + 128u;
    }
}
"""

shaders = [
    ("batch_gemm_q4.wgsl", batch_gemm_q4_wgsl),
    ("batch_gemm_q8.wgsl", batch_gemm_q8_wgsl),
    ("batch_add_rmsnorm.wgsl", batch_add_rmsnorm_wgsl),
    ("batch_fused_mlp_q4.wgsl", batch_fused_mlp_q4_wgsl),
    ("batch_qkv_rope.wgsl", batch_qkv_rope_wgsl),
    ("batch_causal_attn.wgsl", batch_causal_attn_wgsl),
]

for fname, content in shaders:
    path = os.path.join("shaders", fname)
    with open(path, "w") as f:
        f.write(content.strip() + "\n")
    print(f"Wrote {path}")
