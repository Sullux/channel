struct PushConstants {
    M: u32,
    K: u32,
    x_offset: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_reduce: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;      // 0..3 (4 rows per workgroup)
    let lane = lid.x & 31u;            // 0..31 (lane in wave)
    let row = wgid.x * 4u + local_row;

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;

    var thread_acc: f32 = 0.0;

    if (row < pc.M) {
        var blk = lane;
        while (blk < num_blocks) {
            let blk_word_off = row_word_offset + blk * 5u;
            let sm = unpack2x16float(W[blk_word_off]);
            let s = sm.x;
            let m = sm.y;

            let w0 = W[blk_word_off + 1u];
            let w1 = W[blk_word_off + 2u];
            let w2 = W[blk_word_off + 3u];
            let w3 = W[blk_word_off + 4u];

            let x_base = pc.x_offset + blk * 32u;

            var sum_nx: f32 = 0.0;
            var sum_x: f32 = 0.0;

            // Word 0 (nibbles 0..7)
            var cur_w = w0;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let nib = f32(cur_w & 0x0Fu);
                let xv = X[x_base + j];
                sum_nx = sum_nx + nib * xv;
                sum_x = sum_x + xv;
                cur_w = cur_w >> 4u;
            }

            // Word 1 (nibbles 8..15)
            cur_w = w1;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let nib = f32(cur_w & 0x0Fu);
                let xv = X[x_base + 8u + j];
                sum_nx = sum_nx + nib * xv;
                sum_x = sum_x + xv;
                cur_w = cur_w >> 4u;
            }

            // Word 2 (nibbles 16..23)
            cur_w = w2;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let nib = f32(cur_w & 0x0Fu);
                let xv = X[x_base + 16u + j];
                sum_nx = sum_nx + nib * xv;
                sum_x = sum_x + xv;
                cur_w = cur_w >> 4u;
            }

            // Word 3 (nibbles 24..31)
            cur_w = w3;
            for (var j = 0u; j < 8u; j = j + 1u) {
                let nib = f32(cur_w & 0x0Fu);
                let xv = X[x_base + 24u + j];
                sum_nx = sum_nx + nib * xv;
                sum_x = sum_x + xv;
                cur_w = cur_w >> 4u;
            }

            thread_acc = thread_acc + (s * sum_nx + m * sum_x);
            blk = blk + 32u;
        }
    }

    s_reduce[lid.x] = thread_acc;
    workgroupBarrier();

    // Reduction across the 32 lanes of this row's wave
    let base = local_row * 32u;
    if (lane < 16u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 2u]; }
    workgroupBarrier();
    if (lane < 1u) { s_reduce[base + lane] = s_reduce[base + lane] + s_reduce[base + lane + 1u]; }
    workgroupBarrier();

    if (lane == 0u and row < pc.M) {
        Y[row] = s_reduce[base];
    }
}
