struct PushConstants {
    head_dim: u32,
    kv_dim: u32,
    gqa_ratio: u32,
    inv_sqrt_dim: f32,
    is_sliding: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> K_cache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> V_cache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> Active_slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> Attn_out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> Step_params: array<u32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_Q: array<vec4<f32>, 128>;
var<workgroup> s_scores: array<f32, 4096>;
var<workgroup> s_reduce: array<f32, 32>;
var<workgroup> s_max_val: f32;
var<workgroup> s_inv_sum: f32;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let q_head = wgid.x;
    let lane = lid.x;
    let D_vec4 = pc.head_dim >> 2u;
    let kv_vec4 = pc.kv_dim >> 2u;
    let kv_h = q_head / pc.gqa_ratio;
    let q_offset = q_head * D_vec4;

    let is_sliding = (pc.is_sliding != 0u);
    let S = select(Step_params[2], Step_params[3], is_sliding);
    let slot_base = select(0u, 4096u, is_sliding);

    // Pre-load Q for this head into shared memory (32 threads load 4 vec4s each)
    s_Q[lane] = Q[q_offset + lane];
    s_Q[lane + 32u] = Q[q_offset + lane + 32u];
    if (D_vec4 == 128u) {
        s_Q[lane + 64u] = Q[q_offset + lane + 64u];
        s_Q[lane + 96u] = Q[q_offset + lane + 96u];
    }
    workgroupBarrier();

    // 1. Compute dot product scores for all active slots (4x unrolled burst reads)
    var slot_i = lane;
    while (slot_i < S) {
        let physical_slot = Active_slots[slot_base + slot_i];
        let kv_offset = physical_slot * kv_vec4 + kv_h * D_vec4;

        var dot_sum = 0.0;
        var d = 0u;
        while (d + 4u <= D_vec4) {
            let q0 = s_Q[d];
            let q1 = s_Q[d + 1u];
            let q2 = s_Q[d + 2u];
            let q3 = s_Q[d + 3u];

            let k0 = K_cache[kv_offset + d];
            let k1 = K_cache[kv_offset + d + 1u];
            let k2 = K_cache[kv_offset + d + 2u];
            let k3 = K_cache[kv_offset + d + 3u];

            dot_sum = dot_sum + dot(q0, k0) + dot(q1, k1) + dot(q2, k2) + dot(q3, k3);
            d = d + 4u;
        }
        while (d < D_vec4) {
            dot_sum = dot_sum + dot(s_Q[d], K_cache[kv_offset + d]);
            d = d + 1u;
        }
        s_scores[slot_i] = dot_sum * pc.inv_sqrt_dim;
        slot_i += 32u;
    }
    workgroupBarrier();

    // 2. Wave-Parallel Softmax over s_scores[0..S-1]
    var local_max: f32 = -1e9;
    var i = lane;
    while (i < S) {
        if (s_scores[i] > local_max) { local_max = s_scores[i]; }
        i += 32u;
    }
    s_reduce[lane] = local_max;
    workgroupBarrier();

    if (lane < 16u) { if (s_reduce[lane + 16u] > s_reduce[lane]) { s_reduce[lane] = s_reduce[lane + 16u]; } }
    if (lane < 8u)  { if (s_reduce[lane + 8u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 8u]; } }
    if (lane < 4u)  { if (s_reduce[lane + 4u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 4u]; } }
    if (lane < 2u)  { if (s_reduce[lane + 2u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 2u]; } }
    if (lane < 1u)  { if (s_reduce[lane + 1u] > s_reduce[lane])  { s_reduce[lane] = s_reduce[lane + 1u]; } }

    if (lane == 0u) { s_max_val = s_reduce[0]; }
    workgroupBarrier();

    let max_val = s_max_val;
    var local_sum: f32 = 0.0;
    i = lane;
    while (i < S) {
        let exp_v = exp(s_scores[i] - max_val);
        s_scores[i] = exp_v;
        local_sum += exp_v;
        i += 32u;
    }
    s_reduce[lane] = local_sum;
    workgroupBarrier();

    if (lane < 16u) { s_reduce[lane] += s_reduce[lane + 16u]; }
    if (lane < 8u)  { s_reduce[lane] += s_reduce[lane + 8u]; }
    if (lane < 4u)  { s_reduce[lane] += s_reduce[lane + 4u]; }
    if (lane < 2u)  { s_reduce[lane] += s_reduce[lane + 2u]; }
    if (lane < 1u)  { s_reduce[lane] += s_reduce[lane + 1u]; }

    if (lane == 0u) { s_inv_sum = 1.0 / (s_reduce[0] + 1e-9); }
    workgroupBarrier();

    let inv_sum = s_inv_sum;
    i = lane;
    while (i < S) {
        s_scores[i] *= inv_sum;
        i += 32u;
    }
    workgroupBarrier();

    // 3. Weighted Sum over V cache (8x burst accumulation for maximum 128-bit memory throughput)
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    var s = 0u;
    while (s + 8u <= S) {
        let p0 = Active_slots[slot_base + s];
        let p1 = Active_slots[slot_base + s + 1u];
        let p2 = Active_slots[slot_base + s + 2u];
        let p3 = Active_slots[slot_base + s + 3u];
        let p4 = Active_slots[slot_base + s + 4u];
        let p5 = Active_slots[slot_base + s + 5u];
        let p6 = Active_slots[slot_base + s + 6u];
        let p7 = Active_slots[slot_base + s + 7u];

        let w0 = s_scores[s];
        let w1 = s_scores[s + 1u];
        let w2 = s_scores[s + 2u];
        let w3 = s_scores[s + 3u];
        let w4 = s_scores[s + 4u];
        let w5 = s_scores[s + 5u];
        let w6 = s_scores[s + 6u];
        let w7 = s_scores[s + 7u];

        let v0_0 = V_cache[p0 * kv_vec4 + kv_h * D_vec4 + lane];
        let v0_1 = V_cache[p0 * kv_vec4 + kv_h * D_vec4 + lane + 32u];
        let v1_0 = V_cache[p1 * kv_vec4 + kv_h * D_vec4 + lane];
        let v1_1 = V_cache[p1 * kv_vec4 + kv_h * D_vec4 + lane + 32u];
        let v2_0 = V_cache[p2 * kv_vec4 + kv_h * D_vec4 + lane];
        let v2_1 = V_cache[p2 * kv_vec4 + kv_h * D_vec4 + lane + 32u];
        let v3_0 = V_cache[p3 * kv_vec4 + kv_h * D_vec4 + lane];
        let v3_1 = V_cache[p3 * kv_vec4 + kv_h * D_vec4 + lane + 32u];
        let v4_0 = V_cache[p4 * kv_vec4 + kv_h * D_vec4 + lane];
        let v4_1 = V_cache[p4 * kv_vec4 + kv_h * D_vec4 + lane + 32u];
        let v5_0 = V_cache[p5 * kv_vec4 + kv_h * D_vec4 + lane];
        let v5_1 = V_cache[p5 * kv_vec4 + kv_h * D_vec4 + lane + 32u];
        let v6_0 = V_cache[p6 * kv_vec4 + kv_h * D_vec4 + lane];
        let v6_1 = V_cache[p6 * kv_vec4 + kv_h * D_vec4 + lane + 32u];
        let v7_0 = V_cache[p7 * kv_vec4 + kv_h * D_vec4 + lane];
        let v7_1 = V_cache[p7 * kv_vec4 + kv_h * D_vec4 + lane + 32u];

        acc0 += w0 * v0_0 + w1 * v1_0 + w2 * v2_0 + w3 * v3_0 + w4 * v4_0 + w5 * v5_0 + w6 * v6_0 + w7 * v7_0;
        acc1 += w0 * v0_1 + w1 * v1_1 + w2 * v2_1 + w3 * v3_1 + w4 * v4_1 + w5 * v5_1 + w6 * v6_1 + w7 * v7_1;

        if (D_vec4 == 128u) {
            let v0_2 = V_cache[p0 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v0_3 = V_cache[p0 * kv_vec4 + kv_h * D_vec4 + lane + 96u];
            let v1_2 = V_cache[p1 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v1_3 = V_cache[p1 * kv_vec4 + kv_h * D_vec4 + lane + 96u];
            let v2_2 = V_cache[p2 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v2_3 = V_cache[p2 * kv_vec4 + kv_h * D_vec4 + lane + 96u];
            let v3_2 = V_cache[p3 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v3_3 = V_cache[p3 * kv_vec4 + kv_h * D_vec4 + lane + 96u];
            let v4_2 = V_cache[p4 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v4_3 = V_cache[p4 * kv_vec4 + kv_h * D_vec4 + lane + 96u];
            let v5_2 = V_cache[p5 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v5_3 = V_cache[p5 * kv_vec4 + kv_h * D_vec4 + lane + 96u];
            let v6_2 = V_cache[p6 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v6_3 = V_cache[p6 * kv_vec4 + kv_h * D_vec4 + lane + 96u];
            let v7_2 = V_cache[p7 * kv_vec4 + kv_h * D_vec4 + lane + 64u];
            let v7_3 = V_cache[p7 * kv_vec4 + kv_h * D_vec4 + lane + 96u];

            acc2 += w0 * v0_2 + w1 * v1_2 + w2 * v2_2 + w3 * v3_2 + w4 * v4_2 + w5 * v5_2 + w6 * v6_2 + w7 * v7_2;
            acc3 += w0 * v0_3 + w1 * v1_3 + w2 * v2_3 + w3 * v3_3 + w4 * v4_3 + w5 * v5_3 + w6 * v6_3 + w7 * v7_3;
        }
        s += 8u;
    }

    while (s < S) {
        let physical_slot = Active_slots[slot_base + s];
        let kv_offset = physical_slot * kv_vec4 + kv_h * D_vec4;
        let weight = s_scores[s];

        acc0 += weight * V_cache[kv_offset + lane];
        acc1 += weight * V_cache[kv_offset + lane + 32u];
        if (D_vec4 == 128u) {
            acc2 += weight * V_cache[kv_offset + lane + 64u];
            acc3 += weight * V_cache[kv_offset + lane + 96u];
        }
        s += 1u;
    }

    Attn_out[q_offset + lane] = acc0;
    Attn_out[q_offset + lane + 32u] = acc1;
    if (D_vec4 == 128u) {
        Attn_out[q_offset + lane + 64u] = acc2;
        Attn_out[q_offset + lane + 96u] = acc3;
    }
}
