struct PushConstants {
    vocab_size: u32,
};

@group(0) @binding(0) var<storage, read> Logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> OutIDs: array<u32>;
@group(0) @binding(2) var<storage, read_write> OutVals: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_top_v: array<f32, 64>;
var<workgroup> s_top_i: array<u32, 64>;
var<workgroup> s_cand_v: array<f32, 64>;
var<workgroup> s_cand_i: array<u32, 64>;
var<workgroup> s_comb_v: array<f32, 128>;
var<workgroup> s_comb_i: array<u32, 128>;

fn softcap(v: f32) -> f32 {
    return 30.0 * tanh(v * (1.0 / 30.0));
}

fn bitonic_sort_top(tid: u32) {
    var k: u32 = 2u;
    while (k <= 64u) {
        var j = k >> 1u;
        while (j > 0u) {
            let l = tid ^ j;
            if (l > tid) {
                let dir = (tid & k) == 0u;
                let v_i = s_top_v[tid];
                let v_l = s_top_v[l];
                if ((dir && v_i < v_l) || (!dir && v_i > v_l)) {
                    s_top_v[tid] = v_l; s_top_v[l] = v_i;
                    let idx_i = s_top_i[tid]; s_top_i[tid] = s_top_i[l]; s_top_i[l] = idx_i;
                }
            }
            workgroupBarrier();
            j = j >> 1u;
        }
        k = k << 1u;
    }
}

fn bitonic_sort_cand(tid: u32) {
    var k: u32 = 2u;
    while (k <= 64u) {
        var j = k >> 1u;
        while (j > 0u) {
            let l = tid ^ j;
            if (l > tid) {
                let dir = (tid & k) == 0u;
                let v_i = s_cand_v[tid];
                let v_l = s_cand_v[l];
                if ((dir && v_i < v_l) || (!dir && v_i > v_l)) {
                    s_cand_v[tid] = v_l; s_cand_v[l] = v_i;
                    let idx_i = s_cand_i[tid]; s_cand_i[tid] = s_cand_i[l]; s_cand_i[l] = idx_i;
                }
            }
            workgroupBarrier();
            j = j >> 1u;
        }
        k = k << 1u;
    }
}

fn bitonic_merge_top_cand(tid: u32) {
    s_comb_v[tid] = s_top_v[tid];
    s_comb_i[tid] = s_top_i[tid];
    s_comb_v[64u + tid] = s_cand_v[63u - tid];
    s_comb_i[64u + tid] = s_cand_i[63u - tid];
    workgroupBarrier();

    var j: u32 = 64u;
    while (j > 0u) {
        let chunk = tid / j;
        let rem = tid % j;
        let i = chunk * (2u * j) + rem;
        let l = i + j;
        if (s_comb_v[i] < s_comb_v[l]) {
            let tv = s_comb_v[i]; s_comb_v[i] = s_comb_v[l]; s_comb_v[l] = tv;
            let ti = s_comb_i[i]; s_comb_i[i] = s_comb_i[l]; s_comb_i[l] = ti;
        }
        workgroupBarrier();
        j = j >> 1u;
    }

    s_top_v[tid] = s_comb_v[tid];
    s_top_i[tid] = s_comb_i[tid];
    workgroupBarrier();
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let chunk_start = wgid.x * 4096u;
    let V = pc.vocab_size;

    // Load initial 64 elements
    let init_idx = chunk_start + tid;
    if (init_idx < V) {
        s_top_v[tid] = softcap(Logits[init_idx]);
        s_top_i[tid] = init_idx;
    } else {
        s_top_v[tid] = -1e30;
        s_top_i[tid] = 0u;
    }
    workgroupBarrier();

    bitonic_sort_top(tid);

    // Process remaining 63 chunks of 64
    for (var c = 1u; c < 64u; c = c + 1u) {
        let idx = chunk_start + c * 64u + tid;
        if (idx < V) {
            s_cand_v[tid] = softcap(Logits[idx]);
            s_cand_i[tid] = idx;
        } else {
            s_cand_v[tid] = -1e30;
            s_cand_i[tid] = 0u;
        }
        workgroupBarrier();

        bitonic_sort_cand(tid);
        // If the best in cand is greater than the smallest in top, merge
        if (s_cand_v[0] > s_top_v[63]) {
            bitonic_merge_top_cand(tid);
        }
    }

    // Write top 64 for this workgroup
    let out_offset = wgid.x * 64u + tid;
    OutIDs[out_offset] = s_top_i[tid];
    OutVals[out_offset] = s_top_v[tid];
}
