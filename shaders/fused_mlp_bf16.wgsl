struct PushConstants {
    M: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> W_gate: array<u32>;
@group(0) @binding(1) var<storage, read> W_up: array<u32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> sdata_gate: array<f32, 128>;
var<workgroup> sdata_up: array<f32, 128>;

fn bf16_to_f32(u: u32) -> f32 {
    return bitcast<f32>(u << 16u);
}

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_row = lid.x >> 5u;
    let lane = lid.x & 31u;
    let row = wgid.x * 4u + local_row;

    let k_words = pc.K / 2u;
    let row_offset = row * k_words;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    if (row < pc.M) {
        var idx = lane * 4u;
        while (idx < k_words) {
            let gp0 = W_gate[row_offset + idx];
            let gp1 = W_gate[row_offset + idx + 1u];
            let gp2 = W_gate[row_offset + idx + 2u];
            let gp3 = W_gate[row_offset + idx + 3u];

            let up0 = W_up[row_offset + idx];
            let up1 = W_up[row_offset + idx + 1u];
            let up2 = W_up[row_offset + idx + 2u];
            let up3 = W_up[row_offset + idx + 3u];

            let x_base = idx * 2u;
            let x0 = X[x_base];
            let x1 = X[x_base + 1u];
            let x2 = X[x_base + 2u];
            let x3 = X[x_base + 3u];
            let x4 = X[x_base + 4u];
            let x5 = X[x_base + 5u];
            let x6 = X[x_base + 6u];
            let x7 = X[x_base + 7u];

            gate_acc = gate_acc + bf16_to_f32(gp0 & 0xFFFFu) * x0 + bf16_to_f32(gp0 >> 16u) * x1;
            gate_acc = gate_acc + bf16_to_f32(gp1 & 0xFFFFu) * x2 + bf16_to_f32(gp1 >> 16u) * x3;
            gate_acc = gate_acc + bf16_to_f32(gp2 & 0xFFFFu) * x4 + bf16_to_f32(gp2 >> 16u) * x5;
            gate_acc = gate_acc + bf16_to_f32(gp3 & 0xFFFFu) * x6 + bf16_to_f32(gp3 >> 16u) * x7;

            up_acc = up_acc + bf16_to_f32(up0 & 0xFFFFu) * x0 + bf16_to_f32(up0 >> 16u) * x1;
            up_acc = up_acc + bf16_to_f32(up1 & 0xFFFFu) * x2 + bf16_to_f32(up1 >> 16u) * x3;
            up_acc = up_acc + bf16_to_f32(up2 & 0xFFFFu) * x4 + bf16_to_f32(up2 >> 16u) * x5;
            up_acc = up_acc + bf16_to_f32(up3 & 0xFFFFu) * x6 + bf16_to_f32(up3 >> 16u) * x7;

            idx = idx + 128u;
        }
    }

    sdata_gate[lid.x] = gate_acc;
    sdata_up[lid.x] = up_acc;
    workgroupBarrier();

    if (lane < 16u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 16u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 8u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 4u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 2u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 2u];
    }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        let final_g = sdata_gate[lid.x] + sdata_gate[lid.x + 1u];
        let final_u = sdata_up[lid.x] + sdata_up[lid.x + 1u];
        Y[row] = gelu_tanh(final_g) * final_u;
    }
}
