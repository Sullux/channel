struct PushConstants {
    vocab_size: u32,
};

@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> OutToken: array<u32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let V = pc.vocab_size;

    var best_val: f32 = -1e30;
    var best_idx: u32 = 0u;

    var i = tid;
    while (i < V) {
        let v = Logits[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
        i = i + 256u;
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        OutToken[0] = s_idx[0];
    }
}
