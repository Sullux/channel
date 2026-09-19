struct PushConstants {
    dim: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let D_vec = pc.dim / 4u;

    var sum_sq: f32 = 0.0;
    var idx = tid;
    while (idx < D_vec) {
        let v = X[idx];
        sum_sq = sum_sq + dot(v, v);
        idx = idx + 256u;
    }

    sdata[tid] = sum_sq;
    workgroupBarrier();

    // Workgroup reduction
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
    }

    let total_sum = sdata[0];
    let mean_sq = total_sum / f32(pc.dim);
    let rsqrt_val = inverseSqrt(mean_sq + pc.eps);

    idx = tid;
    while (idx < D_vec) {
        Y[idx] = X[idx] * rsqrt_val * W[idx];
        idx = idx + 256u;
    }
}
