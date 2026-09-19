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

    let num_blocks = pc.K / 32u;
    let row_word_offset = row * num_blocks * 9u;
    let lane_word_idx = 1u + (lane >> 2u);
    let byte_shift = (lane & 3u) * 8u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    if (row < pc.M) {
        var b = 0u;
        while (b < num_blocks) {
            let bb0 = row_word_offset + b * 9u;
            let x0 = X[b * 32u + lane];
            let gs0 = bitcast<f32>(W_gate[bb0]);
            let gp0 = W_gate[bb0 + lane_word_idx];
            let gr0 = (gp0 >> byte_shift) & 0xFFu;
            let gsb0 = (i32(gr0) << 24) >> 24;
            gate_acc = gate_acc + f32(gsb0) * gs0 * x0;
            let us0 = bitcast<f32>(W_up[bb0]);
            let up0 = W_up[bb0 + lane_word_idx];
            let ur0 = (up0 >> byte_shift) & 0xFFu;
            let usb0 = (i32(ur0) << 24) >> 24;
            up_acc = up_acc + f32(usb0) * us0 * x0;

            let bb1 = bb0 + 9u;
            let x1 = X[(b + 1u) * 32u + lane];
            let gs1 = bitcast<f32>(W_gate[bb1]);
            let gp1 = W_gate[bb1 + lane_word_idx];
            let gr1 = (gp1 >> byte_shift) & 0xFFu;
            let gsb1 = (i32(gr1) << 24) >> 24;
            gate_acc = gate_acc + f32(gsb1) * gs1 * x1;
            let us1 = bitcast<f32>(W_up[bb1]);
            let up1 = W_up[bb1 + lane_word_idx];
            let ur1 = (up1 >> byte_shift) & 0xFFu;
            let usb1 = (i32(ur1) << 24) >> 24;
            up_acc = up_acc + f32(usb1) * us1 * x1;

            let bb2 = bb0 + 18u;
            let x2 = X[(b + 2u) * 32u + lane];
            let gs2 = bitcast<f32>(W_gate[bb2]);
            let gp2 = W_gate[bb2 + lane_word_idx];
            let gr2 = (gp2 >> byte_shift) & 0xFFu;
            let gsb2 = (i32(gr2) << 24) >> 24;
            gate_acc = gate_acc + f32(gsb2) * gs2 * x2;
            let us2 = bitcast<f32>(W_up[bb2]);
            let up2 = W_up[bb2 + lane_word_idx];
            let ur2 = (up2 >> byte_shift) & 0xFFu;
            let usb2 = (i32(ur2) << 24) >> 24;
            up_acc = up_acc + f32(usb2) * us2 * x2;

            let bb3 = bb0 + 27u;
            let x3 = X[(b + 3u) * 32u + lane];
            let gs3 = bitcast<f32>(W_gate[bb3]);
            let gp3 = W_gate[bb3 + lane_word_idx];
            let gr3 = (gp3 >> byte_shift) & 0xFFu;
            let gsb3 = (i32(gr3) << 24) >> 24;
            gate_acc = gate_acc + f32(gsb3) * gs3 * x3;
            let us3 = bitcast<f32>(W_up[bb3]);
            let up3 = W_up[bb3 + lane_word_idx];
            let ur3 = (up3 >> byte_shift) & 0xFFu;
            let usb3 = (i32(ur3) << 24) >> 24;
            up_acc = up_acc + f32(usb3) * us3 * x3;

            b = b + 4u;
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
    if (lane < 1u) {
        sdata_gate[lid.x] = sdata_gate[lid.x] + sdata_gate[lid.x + 1u];
        sdata_up[lid.x] = sdata_up[lid.x] + sdata_up[lid.x + 1u];
    }
    workgroupBarrier();

    if (lane == 0u && row < pc.M) {
        let g_final = sdata_gate[lid.x];
        let u_final = sdata_up[lid.x];
        let act = gelu_tanh(g_final);
        Y[row] = act * u_final;
    }
}
