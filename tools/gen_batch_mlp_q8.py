batch_fused_mlp_q8_wgsl = """
struct PushConstants {
    N: u32,
    M: u32,
    K: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_gate: array<f32, 32>;
var<workgroup> sdata_up: array<f32, 32>;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wgid.x;
    let t = wgid.y;
    if (row >= pc.M || t >= pc.N) {
        return;
    }
    let lane = lid.x;
    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;
    let x_offset = t * pc.K;
    let lane_word_idx = 1u + (lane >> 2u);
    let byte_shift = (lane & 3u) * 8u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;
    var b = 0u;
    while (b < num_blocks) {
        let blk_off0 = row_word_offset + b * 9u;
        let gs0 = bitcast<f32>(W_gate[blk_off0]);
        let gp0 = W_gate[blk_off0 + lane_word_idx];
        let gb0 = (gp0 >> byte_shift) & 0xFFu;
        let gv0 = f32(bitcast<i32>(gb0 << 24u) >> 24) * gs0;

        let us0 = bitcast<f32>(W_up[blk_off0]);
        let up0 = W_up[blk_off0 + lane_word_idx];
        let ub0 = (up0 >> byte_shift) & 0xFFu;
        let uv0 = f32(bitcast<i32>(ub0 << 24u) >> 24) * us0;

        let x0 = X[x_offset + b * 32u + lane];
        gate_acc = gate_acc + gv0 * x0;
        up_acc = up_acc + uv0 * x0;

        let blk_off1 = blk_off0 + 9u;
        let gs1 = bitcast<f32>(W_gate[blk_off1]);
        let gp1 = W_gate[blk_off1 + lane_word_idx];
        let gb1 = (gp1 >> byte_shift) & 0xFFu;
        let gv1 = f32(bitcast<i32>(gb1 << 24u) >> 24) * gs1;

        let us1 = bitcast<f32>(W_up[blk_off1]);
        let up1 = W_up[blk_off1 + lane_word_idx];
        let ub1 = (up1 >> byte_shift) & 0xFFu;
        let uv1 = f32(bitcast<i32>(ub1 << 24u) >> 24) * us1;

        let x1 = X[x_offset + (b + 1u) * 32u + lane];
        gate_acc = gate_acc + gv1 * x1;
        up_acc = up_acc + uv1 * x1;

        b = b + 2u;
    }

    sdata_gate[lane] = gate_acc;
    sdata_up[lane] = up_acc;
    workgroupBarrier();

    if (lane < 16u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 16u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 8u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 4u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 2u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 2u];
    }
    workgroupBarrier();
    if (lane < 1u) {
        sdata_gate[lane] = sdata_gate[lane] + sdata_gate[lane + 1u];
        sdata_up[lane] = sdata_up[lane] + sdata_up[lane + 1u];
    }
    workgroupBarrier();

    if (lane == 0u) {
        let g_final = sdata_gate[0];
        let u_final = sdata_up[0];
        Y[t * pc.M + row] = gelu_tanh(g_final) * u_final;
    }
}
"""

with open("shaders/batch_fused_mlp_q8.wgsl", "w") as f:
    f.write(batch_fused_mlp_q8_wgsl.strip() + "\n")
print("Wrote shaders/batch_fused_mlp_q8.wgsl")
