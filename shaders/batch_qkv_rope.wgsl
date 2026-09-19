struct PushConstants {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    k_eq_v: u32,
    rope_theta: f32,
    eps: f32,
    N: u32,
    start_clock: u32,
    slot_offset: u32,
    pad0: u32,
    pad1: u32,
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
    let slot_idx = Slots[pc.slot_offset + t];
    let clock = pc.start_clock + t;
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
