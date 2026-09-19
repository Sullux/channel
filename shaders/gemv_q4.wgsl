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

fn unpack_dot2(w: u32, v_a: vec4<f32>, v_b: vec4<f32>) -> f32 {
    let n0 = f32(w & 0xFu) - 8.0;
    let n1 = f32((w >> 4u) & 0xFu) - 8.0;
    let n2 = f32((w >> 8u) & 0xFu) - 8.0;
    let n3 = f32((w >> 12u) & 0xFu) - 8.0;
    let n4 = f32((w >> 16u) & 0xFu) - 8.0;
    let n5 = f32((w >> 20u) & 0xFu) - 8.0;
    let n6 = f32((w >> 24u) & 0xFu) - 8.0;
    let n7 = f32(w >> 28u) - 8.0;
    return n0 * v_a.x + n1 * v_a.y + n2 * v_a.z + n3 * v_a.w +
           n4 * v_b.x + n5 * v_b.y + n6 * v_b.z + n7 * v_b.w;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;      // 0..7 (8 rows per workgroup)
    let lane = lid.x & 31u;            // 0..31 (lane in wave)
    let row = wgid.x * 8u + local_row;

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let x_vec4_base = pc.x_offset >> 2u;

    var thread_acc: f32 = 0.0;

    if (row < pc.M) {
        var cur_blk = lane;
        while (cur_blk < num_blocks) {
            let blk_off = row_word_offset + cur_blk * 5u;
            let s = unpack2x16float(W[blk_off]).x;

            let w0 = W[blk_off + 1u];
            let w1 = W[blk_off + 2u];
            let w2 = W[blk_off + 3u];
            let w3 = W[blk_off + 4u];

            let x_base = x_vec4_base + cur_blk * 8u;
            let va0 = X[x_base + 0u];
            let vb0 = X[x_base + 1u];
            let va1 = X[x_base + 2u];
            let vb1 = X[x_base + 3u];
            let va2 = X[x_base + 4u];
            let vb2 = X[x_base + 5u];
            let va3 = X[x_base + 6u];
            let vb3 = X[x_base + 7u];

            let dot_sum = unpack_dot2(w0, va0, vb0) +
                          unpack_dot2(w1, va1, vb1) +
                          unpack_dot2(w2, va2, vb2) +
                          unpack_dot2(w3, va3, vb3);

            thread_acc += (s * dot_sum);

            cur_blk += 32u;
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
