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
    if (lane < 1u) { s_sum_sq[lane] = s_sum_sq[lane] + s_sum_sq[lane + 1u]; }
    workgroupBarrier();

    if (lane == 0u) {
        let mean_sq = s_sum_sq[0] / f32(H);
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
