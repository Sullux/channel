struct PushConstants {
    M: u32,
    K: u32,
    x_offset: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_reduce: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;      // 0..7 (8 rows per workgroup)
    let lane = lid.x & 31u;            // 0..31 (lane in wave)
    let row = wgid.x * 8u + local_row;

    let blk_in_wave = lane >> 2u;     // 0..7 (which of the 8 blocks)
    let sub_lane = lane & 3u;         // 0..3 (which pair of words in the block)
    let w_offset = 1u + sub_lane * 2u; // 1, 3, 5, 7
    let vec4_base = sub_lane * 2u;    // 0, 2, 4, 6

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;
    let x_vec4_base = pc.x_offset >> 2u;

    var thread_acc: f32 = 0.0;

    if (row < pc.M) {
        var base_blk = 0u;
        while (base_blk < num_blocks) {
            let cur_blk = base_blk + blk_in_wave;
            let blk_off = row_word_offset + cur_blk * 9u;

            let s = bitcast<f32>(W[blk_off]);
            let w_a = W[blk_off + w_offset + 0u];
            let w_b = W[blk_off + w_offset + 1u];

            let cur_k_vec = (cur_blk * 32u >> 2u) + vec4_base;

            let b0 = f32((i32(w_a & 0xFFu) << 24) >> 24);
            let b1 = f32((i32((w_a >> 8u) & 0xFFu) << 24) >> 24);
            let b2 = f32((i32((w_a >> 16u) & 0xFFu) << 24) >> 24);
            let b3 = f32(i32(w_a) >> 24);

            let b4 = f32((i32(w_b & 0xFFu) << 24) >> 24);
            let b5 = f32((i32((w_b >> 8u) & 0xFFu) << 24) >> 24);
            let b6 = f32((i32((w_b >> 16u) & 0xFFu) << 24) >> 24);
            let b7 = f32(i32(w_b) >> 24);

            let v_a = X[x_vec4_base + cur_k_vec + 0u];
            let v_b = X[x_vec4_base + cur_k_vec + 1u];

            let sum_wx = b0 * v_a.x + b1 * v_a.y + b2 * v_a.z + b3 * v_a.w +
                         b4 * v_b.x + b5 * v_b.y + b6 * v_b.z + b7 * v_b.w;

            thread_acc += s * sum_wx;

            base_blk += 8u;
        }
    }

    s_reduce[lid.x] = thread_acc;
    workgroupBarrier();

    let base = local_row * 32u;
    if (lane < 16u) { s_reduce[base + lane] += s_reduce[base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u)  { s_reduce[base + lane] += s_reduce[base + lane + 8u];  }
    workgroupBarrier();
    if (lane < 4u)  { s_reduce[base + lane] += s_reduce[base + lane + 4u];  }
    workgroupBarrier();
    if (lane < 2u)  { s_reduce[base + lane] += s_reduce[base + lane + 2u];  }
    workgroupBarrier();
    if (lane < 1u)  { s_reduce[base + lane] += s_reduce[base + lane + 1u];  }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        Y[row] = s_reduce[base];
    }
}
