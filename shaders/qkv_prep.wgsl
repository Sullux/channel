struct PushConstants {
    q_dim: u32,
    k_dim: u32,
    v_dim: u32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> Q_in: array<f32>;
@group(0) @binding(1) var<storage, read> K_in: array<f32>;
@group(0) @binding(2) var<storage, read> V_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> Q_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> K_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> V_out: array<f32>;
@group(0) @binding(6) var<storage, read> Norm_weight: array<u32>;
var<push_constant> pc: PushConstants;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx < pc.q_dim) {
        Q_out[idx] = Q_in[idx] * pc.scale;
    }
    if (idx < pc.k_dim) {
        K_out[idx] = K_in[idx];
    }
    if (idx < pc.v_dim) {
        V_out[idx] = V_in[idx];
    }
}
