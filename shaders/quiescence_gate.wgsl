struct PushConstants {
    hidden_size: u32,
    threshold_sq: f32,
    base_cmd_idx: u32,
    num_cmds: u32,
    targets: array<u32, 16>,
};

@group(0) @binding(0) var<storage, read> X_curr: array<f32>;
@group(0) @binding(1) var<storage, read_write> X_prev: array<f32>;
@group(0) @binding(2) var<storage, read_write> Indirect_args: array<u32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_diff_sq: array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let lane = lid.x;
    let H = pc.hidden_size;

    var sum_sq: f32 = 0.0;
    var i = lane;
    while (i < H) {
        let diff = X_curr[i] - X_prev[i];
        sum_sq = sum_sq + diff * diff;
        X_prev[i] = X_curr[i];
        i = i + 32u;
    }

    s_diff_sq[lane] = sum_sq;
    workgroupBarrier();

    if (lane < 16u) { s_diff_sq[lane] = s_diff_sq[lane] + s_diff_sq[lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { s_diff_sq[lane] = s_diff_sq[lane] + s_diff_sq[lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { s_diff_sq[lane] = s_diff_sq[lane] + s_diff_sq[lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { s_diff_sq[lane] = s_diff_sq[lane] + s_diff_sq[lane + 2u]; }
    workgroupBarrier();

    if (lane == 0u) {
        let mean_sq = (s_diff_sq[0] + s_diff_sq[1]) / f32(H);
        let is_active = (mean_sq >= pc.threshold_sq) || (pc.threshold_sq <= 0.0);
        var c = 0u;
        while (c < pc.num_cmds) {
            let offset = (pc.base_cmd_idx + c) * 3u;
            if (is_active) {
                Indirect_args[offset + 0u] = pc.targets[c];
                Indirect_args[offset + 1u] = 1u;
                Indirect_args[offset + 2u] = 1u;
            } else {
                Indirect_args[offset + 0u] = 0u;
                Indirect_args[offset + 1u] = 0u;
                Indirect_args[offset + 2u] = 0u;
            }
            c = c + 1u;
        }
    }
}
