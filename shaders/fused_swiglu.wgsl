struct PushConstants {
    dim: u32,
};

@group(0) @binding(0) var<storage, read> Gate: array<f32>;
@group(0) @binding(1) var<storage, read> Up: array<f32>;
@group(0) @binding(2) var<storage, read_write> Act: array<f32>;
var<push_constant> pc: PushConstants;

fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= pc.dim) {
        return;
    }
    let g = Gate[idx];
    let u = Up[idx];
    let act = gelu_tanh(g);
    Act[idx] = act * u;
}
