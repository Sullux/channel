enable subgroups;

struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = wgid.x;
    if (row >= pc.M) { return; }

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 5u;
    let lane_word_idx = 1u + (lane >> 3u);
    let nib_shift = (lane & 7u) * 4u;

    var acc: f32 = 0.0;
    for (var b = 0u; b < num_blocks; b = b + 1u) {
        let blk_word_off = row_word_offset + b * 5u;
        let sm = unpack2x16float(W[blk_word_off]);
        let p = W[blk_word_off + lane_word_idx];
        let w = f32((p >> nib_shift) & 0x0Fu) * sm.x + sm.y;
        let x = X[b * 32u + lane];
        acc = acc + w * x;
    }

    let total = subgroupAdd(acc);
    if (lane == 0u) {
        Y[row] = total;
    }
}
