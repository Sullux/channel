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

var<workgroup> sdata0: array<f32, 128>;
var<workgroup> sdata1: array<f32, 128>;
var<workgroup> sdata2: array<f32, 128>;
var<workgroup> sdata3: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;
    let lane = lid.x & 31u;
    let row = wgid.x * 4u + local_row;
    let t_base = wgid.y * 4u;
    if (t_base >= pc.N) {
        return;
    }
    let tid = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;
    let lane_word_idx = 1u + (lane >> 2u);
    let byte_shift = (lane & 3u) * 8u;

    let has_t1 = (t_base + 1u < pc.N);
    let has_t2 = (t_base + 2u < pc.N);
    let has_t3 = (t_base + 3u < pc.N);

    let x_off0 = t_base * pc.K;
    let x_off1 = (t_base + 1u) * pc.K;
    let x_off2 = (t_base + 2u) * pc.K;
    let x_off3 = (t_base + 3u) * pc.K;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    if (row < pc.M) {
        var b = 0u;
        while (b < num_blocks) {
            let blk_off = row_word_offset + b * 9u;
            let s = bitcast<f32>(W[blk_off]);
            let packed_word = W[blk_off + lane_word_idx];
            let byte_val = (packed_word >> byte_shift) & 0xFFu;
            let signed_val = f32(bitcast<i32>(byte_val << 24u) >> 24);
            let w_val = signed_val * s;
            let k_idx = b * 32u + lane;

            acc0 = acc0 + w_val * X[x_off0 + k_idx];
            if (has_t1) { acc1 = acc1 + w_val * X[x_off1 + k_idx]; }
            if (has_t2) { acc2 = acc2 + w_val * X[x_off2 + k_idx]; }
            if (has_t3) { acc3 = acc3 + w_val * X[x_off3 + k_idx]; }

            b = b + 1u;
        }
    }

    sdata0[tid] = acc0;
    sdata1[tid] = acc1;
    sdata2[tid] = acc2;
    sdata3[tid] = acc3;
    workgroupBarrier();

    if (lane < 16u) {
        sdata0[tid] = sdata0[tid] + sdata0[tid + 16u];
        sdata1[tid] = sdata1[tid] + sdata1[tid + 16u];
        sdata2[tid] = sdata2[tid] + sdata2[tid + 16u];
        sdata3[tid] = sdata3[tid] + sdata3[tid + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        sdata0[tid] = sdata0[tid] + sdata0[tid + 8u];
        sdata1[tid] = sdata1[tid] + sdata1[tid + 8u];
        sdata2[tid] = sdata2[tid] + sdata2[tid + 8u];
        sdata3[tid] = sdata3[tid] + sdata3[tid + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        sdata0[tid] = sdata0[tid] + sdata0[tid + 4u];
        sdata1[tid] = sdata1[tid] + sdata1[tid + 4u];
        sdata2[tid] = sdata2[tid] + sdata2[tid + 4u];
        sdata3[tid] = sdata3[tid] + sdata3[tid + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        sdata0[tid] = sdata0[tid] + sdata0[tid + 2u];
        sdata1[tid] = sdata1[tid] + sdata1[tid + 2u];
        sdata2[tid] = sdata2[tid] + sdata2[tid + 2u];
        sdata3[tid] = sdata3[tid] + sdata3[tid + 2u];
    }
    workgroupBarrier();
    if (lane < 1u) {
        sdata0[tid] = sdata0[tid] + sdata0[tid + 1u];
        sdata1[tid] = sdata1[tid] + sdata1[tid + 1u];
        sdata2[tid] = sdata2[tid] + sdata2[tid + 1u];
        sdata3[tid] = sdata3[tid] + sdata3[tid + 1u];
    }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        Y[t_base * pc.M + row] = sdata0[tid];
        if (has_t1) { Y[(t_base + 1u) * pc.M + row] = sdata1[tid]; }
        if (has_t2) { Y[(t_base + 2u) * pc.M + row] = sdata2[tid]; }
        if (has_t3) { Y[(t_base + 3u) * pc.M + row] = sdata3[tid]; }
    }
}
