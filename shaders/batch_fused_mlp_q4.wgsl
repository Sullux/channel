struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

// 64 tokens x 8 vec4 = 512 vec4 (8 KB LDS)
var<workgroup> s_X: array<vec4<f32>, 512>;
// 32 rows x 5 u32 = 160 u32 (640 B LDS)
var<workgroup> s_Wg: array<u32, 160>;
var<workgroup> s_Wu: array<u32, 160>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

fn unpack_dot2(w: u32, v_a: vec4<f32>, v_b: vec4<f32>) -> f32 {
    let u_low = vec4<u32>(w & 0xFu, (w >> 4u) & 0xFu, (w >> 8u) & 0xFu, (w >> 12u) & 0xFu);
    let u_high = vec4<u32>((w >> 16u) & 0xFu, (w >> 20u) & 0xFu, (w >> 24u) & 0xFu, w >> 28u);
    let na = vec4<f32>(u_low) - vec4<f32>(8.0);
    let nb = vec4<f32>(u_high) - vec4<f32>(8.0);
    return dot(na, v_a) + dot(nb, v_b);
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.y * 16u + lid.x; // 0..255

    let row_base = wgid.x * 32u + lid.x * 2u; // 2 rows per thread (32 rows/wg)
    let t_base = wgid.y * 64u + lid.y * 4u;   // 4 tokens per thread (64 tokens/wg)

    let num_blocks = pc.K / 32u;
    let k_vec4 = pc.K >> 2u;

    let r0 = row_base + 0u; let r1 = row_base + 1u;
    let t0 = t_base + 0u; let t1 = t_base + 1u;
    let t2 = t_base + 2u; let t3 = t_base + 3u;

    let valid_r0 = (r0 < pc.M); let valid_r1 = (r1 < pc.M);
    let valid_t0 = (t0 < pc.N); let valid_t1 = (t1 < pc.N);
    let valid_t2 = (t2 < pc.N); let valid_t3 = (t3 < pc.N);

    var g00: f32 = 0.0; var g01: f32 = 0.0; var g02: f32 = 0.0; var g03: f32 = 0.0;
    var g10: f32 = 0.0; var g11: f32 = 0.0; var g12: f32 = 0.0; var g13: f32 = 0.0;
    var u00: f32 = 0.0; var u01: f32 = 0.0; var u02: f32 = 0.0; var u03: f32 = 0.0;
    var u10: f32 = 0.0; var u11: f32 = 0.0; var u12: f32 = 0.0; var u13: f32 = 0.0;

    var cur_blk = 0u;
    while (cur_blk < num_blocks) {
        // 1. Cooperative load of 64 tokens x 8 vec4 into s_X (256 threads load 512 vec4)
        let tok_idx0 = tid >> 3u;        // 0..31
        let tok_idx1 = tok_idx0 + 32u;   // 32..63
        let vec_idx = tid & 7u;          // 0..7

        let glob_t0 = wgid.y * 64u + tok_idx0;
        let glob_t1 = wgid.y * 64u + tok_idx1;
        let glob_k = cur_blk * 8u + vec_idx;

        if (glob_t0 < pc.N && glob_k < k_vec4) {
            s_X[tid] = X[glob_t0 * k_vec4 + glob_k];
        } else {
            s_X[tid] = vec4<f32>(0.0);
        }

        if (glob_t1 < pc.N && glob_k < k_vec4) {
            s_X[tid + 256u] = X[glob_t1 * k_vec4 + glob_k];
        } else {
            s_X[tid + 256u] = vec4<f32>(0.0);
        }

        // 2. Cooperative load of 32 rows x 5 u32 into s_Wg & s_Wu
        if (tid < 160u) {
            let row_idx = tid / 5u; // 0..31
            let word_idx = tid % 5u; // 0..4
            let glob_r = wgid.x * 32u + row_idx;
            if (glob_r < pc.M) {
                let off = glob_r * num_blocks * 5u + cur_blk * 5u + word_idx;
                s_Wg[tid] = W_gate[off];
                s_Wu[tid] = W_up[off];
            } else {
                s_Wg[tid] = 0u;
                s_Wu[tid] = 0u;
            }
        }
        workgroupBarrier();

        // 3. Compute Gate and Up from LDS
        let w_off0 = (lid.x * 2u + 0u) * 5u;
        let w_off1 = (lid.x * 2u + 1u) * 5u;

        let gs0 = unpack2x16float(s_Wg[w_off0]).x; let gs1 = unpack2x16float(s_Wg[w_off1]).x;
        let us0 = unpack2x16float(s_Wu[w_off0]).x; let us1 = unpack2x16float(s_Wu[w_off1]).x;

        let gw0_1 = s_Wg[w_off0 + 1u]; let gw0_2 = s_Wg[w_off0 + 2u];
        let gw0_3 = s_Wg[w_off0 + 3u]; let gw0_4 = s_Wg[w_off0 + 4u];
        let gw1_1 = s_Wg[w_off1 + 1u]; let gw1_2 = s_Wg[w_off1 + 2u];
        let gw1_3 = s_Wg[w_off1 + 3u]; let gw1_4 = s_Wg[w_off1 + 4u];

        let uw0_1 = s_Wu[w_off0 + 1u]; let uw0_2 = s_Wu[w_off0 + 2u];
        let uw0_3 = s_Wu[w_off0 + 3u]; let uw0_4 = s_Wu[w_off0 + 4u];
        let uw1_1 = s_Wu[w_off1 + 1u]; let uw1_2 = s_Wu[w_off1 + 2u];
        let uw1_3 = s_Wu[w_off1 + 3u]; let uw1_4 = s_Wu[w_off1 + 4u];

        let x_tok0 = (lid.y * 4u + 0u) * 8u;
        let x_tok1 = (lid.y * 4u + 1u) * 8u;
        let x_tok2 = (lid.y * 4u + 2u) * 8u;
        let x_tok3 = (lid.y * 4u + 3u) * 8u;

        // Token 0
        let va0 = s_X[x_tok0 + 0u]; let vb0 = s_X[x_tok0 + 1u];
        let va1 = s_X[x_tok0 + 2u]; let vb1 = s_X[x_tok0 + 3u];
        let va2 = s_X[x_tok0 + 4u]; let vb2 = s_X[x_tok0 + 5u];
        let va3 = s_X[x_tok0 + 6u]; let vb3 = s_X[x_tok0 + 7u];

        let g0_d0 = unpack_dot2(gw0_1, va0, vb0) + unpack_dot2(gw0_2, va1, vb1) + unpack_dot2(gw0_3, va2, vb2) + unpack_dot2(gw0_4, va3, vb3);
        let g1_d0 = unpack_dot2(gw1_1, va0, vb0) + unpack_dot2(gw1_2, va1, vb1) + unpack_dot2(gw1_3, va2, vb2) + unpack_dot2(gw1_4, va3, vb3);
        let u0_d0 = unpack_dot2(uw0_1, va0, vb0) + unpack_dot2(uw0_2, va1, vb1) + unpack_dot2(uw0_3, va2, vb2) + unpack_dot2(uw0_4, va3, vb3);
        let u1_d0 = unpack_dot2(uw1_1, va0, vb0) + unpack_dot2(uw1_2, va1, vb1) + unpack_dot2(uw1_3, va2, vb2) + unpack_dot2(uw1_4, va3, vb3);

        g00 += gs0 * g0_d0; g10 += gs1 * g1_d0;
        u00 += us0 * u0_d0; u10 += us1 * u1_d0;

        // Token 1
        let va0_1 = s_X[x_tok1 + 0u]; let vb0_1 = s_X[x_tok1 + 1u];
        let va1_1 = s_X[x_tok1 + 2u]; let vb1_1 = s_X[x_tok1 + 3u];
        let va2_1 = s_X[x_tok1 + 4u]; let vb2_1 = s_X[x_tok1 + 5u];
        let va3_1 = s_X[x_tok1 + 6u]; let vb3_1 = s_X[x_tok1 + 7u];

        let g0_d1 = unpack_dot2(gw0_1, va0_1, vb0_1) + unpack_dot2(gw0_2, va1_1, vb1_1) + unpack_dot2(gw0_3, va2_1, vb2_1) + unpack_dot2(gw0_4, va3_1, vb3_1);
        let g1_d1 = unpack_dot2(gw1_1, va0_1, vb0_1) + unpack_dot2(gw1_2, va1_1, vb1_1) + unpack_dot2(gw1_3, va2_1, vb2_1) + unpack_dot2(gw1_4, va3_1, vb3_1);
        let u0_d1 = unpack_dot2(uw0_1, va0_1, vb0_1) + unpack_dot2(uw0_2, va1_1, vb1_1) + unpack_dot2(uw0_3, va2_1, vb2_1) + unpack_dot2(uw0_4, va3_1, vb3_1);
        let u1_d1 = unpack_dot2(uw1_1, va0_1, vb0_1) + unpack_dot2(uw1_2, va1_1, vb1_1) + unpack_dot2(uw1_3, va2_1, vb2_1) + unpack_dot2(uw1_4, va3_1, vb3_1);

        g01 += gs0 * g0_d1; g11 += gs1 * g1_d1;
        u01 += us0 * u0_d1; u11 += us1 * u1_d1;

        // Token 2
        let va0_2 = s_X[x_tok2 + 0u]; let vb0_2 = s_X[x_tok2 + 1u];
        let va1_2 = s_X[x_tok2 + 2u]; let vb1_2 = s_X[x_tok2 + 3u];
        let va2_2 = s_X[x_tok2 + 4u]; let vb2_2 = s_X[x_tok2 + 5u];
        let va3_2 = s_X[x_tok2 + 6u]; let vb3_2 = s_X[x_tok2 + 7u];

        let g0_d2 = unpack_dot2(gw0_1, va0_2, vb0_2) + unpack_dot2(gw0_2, va1_2, vb1_2) + unpack_dot2(gw0_3, va2_2, vb2_2) + unpack_dot2(gw0_4, va3_2, vb3_2);
        let g1_d2 = unpack_dot2(gw1_1, va0_2, vb0_2) + unpack_dot2(gw1_2, va1_2, vb1_2) + unpack_dot2(gw1_3, va2_2, vb2_2) + unpack_dot2(gw1_4, va3_2, vb3_2);
        let u0_d2 = unpack_dot2(uw0_1, va0_2, vb0_2) + unpack_dot2(uw0_2, va1_2, vb1_2) + unpack_dot2(uw0_3, va2_2, vb2_2) + unpack_dot2(uw0_4, va3_2, vb3_2);
        let u1_d2 = unpack_dot2(uw1_1, va0_2, vb0_2) + unpack_dot2(uw1_2, va1_2, vb1_2) + unpack_dot2(uw1_3, va2_2, vb2_2) + unpack_dot2(uw1_4, va3_2, vb3_2);

        g02 += gs0 * g0_d2; g12 += gs1 * g1_d2;
        u02 += us0 * u0_d2; u12 += us1 * u1_d2;

        // Token 3
        let va0_3 = s_X[x_tok3 + 0u]; let vb0_3 = s_X[x_tok3 + 1u];
        let va1_3 = s_X[x_tok3 + 2u]; let vb1_3 = s_X[x_tok3 + 3u];
        let va2_3 = s_X[x_tok3 + 4u]; let vb2_3 = s_X[x_tok3 + 5u];
        let va3_3 = s_X[x_tok3 + 6u]; let vb3_3 = s_X[x_tok3 + 7u];

        let g0_d3 = unpack_dot2(gw0_1, va0_3, vb0_3) + unpack_dot2(gw0_2, va1_3, vb1_3) + unpack_dot2(gw0_3, va2_3, vb2_3) + unpack_dot2(gw0_4, va3_3, vb3_3);
        let g1_d3 = unpack_dot2(gw1_1, va0_3, vb0_3) + unpack_dot2(gw1_2, va1_3, vb1_3) + unpack_dot2(gw1_3, va2_3, vb2_3) + unpack_dot2(gw1_4, va3_3, vb3_3);
        let u0_d3 = unpack_dot2(uw0_1, va0_3, vb0_3) + unpack_dot2(uw0_2, va1_3, vb1_3) + unpack_dot2(uw0_3, va2_3, vb2_3) + unpack_dot2(uw0_4, va3_3, vb3_3);
        let u1_d3 = unpack_dot2(uw1_1, va0_3, vb0_3) + unpack_dot2(uw1_2, va1_3, vb1_3) + unpack_dot2(uw1_3, va2_3, vb2_3) + unpack_dot2(uw1_4, va3_3, vb3_3);

        g03 += gs0 * g0_d3; g13 += gs1 * g1_d3;
        u03 += us0 * u0_d3; u13 += us1 * u1_d3;

        cur_blk += 1u;
        workgroupBarrier();
    }

    if (valid_t0) {
        if (valid_r0) { Y[t0 * pc.M + r0] = gelu_tanh(g00) * u00; }
        if (valid_r1) { Y[t0 * pc.M + r1] = gelu_tanh(g10) * u10; }
    }
    if (valid_t1) {
        if (valid_r0) { Y[t1 * pc.M + r0] = gelu_tanh(g01) * u01; }
        if (valid_r1) { Y[t1 * pc.M + r1] = gelu_tanh(g11) * u11; }
    }
    if (valid_t2) {
        if (valid_r0) { Y[t2 * pc.M + r0] = gelu_tanh(g02) * u02; }
        if (valid_r1) { Y[t2 * pc.M + r1] = gelu_tanh(g12) * u12; }
    }
    if (valid_t3) {
        if (valid_r0) { Y[t3 * pc.M + r0] = gelu_tanh(g03) * u03; }
        if (valid_r1) { Y[t3 * pc.M + r1] = gelu_tanh(g13) * u13; }
    }
}
