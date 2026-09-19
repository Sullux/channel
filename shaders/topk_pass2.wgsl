struct PushConstants {
    num_groups: u32,
};

@group(0) @binding(0) var<storage, read> IntermediateIDs: array<u32>;
@group(0) @binding(1) var<storage, read> IntermediateVals: array<f32>;
@group(0) @binding(2) var<storage, read_write> OutIDs: array<u32>;
@group(0) @binding(3) var<storage, read_write> OutVals: array<f32>;
var<push_constant> pc: PushConstants;

var<workgroup> s_top_v: array<f32, 64>;
var<workgroup> s_top_i: array<u32, 64>;
var<workgroup> s_cand_v: array<f32, 64>;
var<workgroup> s_cand_i: array<u32, 64>;
var<workgroup> s_comb_v: array<f32, 128>;
var<workgroup> s_comb_i: array<u32, 128>;

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
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let G = pc.num_groups;

    // Load initial 64 elements (group 0)
    s_top_v[tid] = IntermediateVals[tid];
    s_top_i[tid] = IntermediateIDs[tid];
    workgroupBarrier();

    bitonic_sort_top(tid);

    // Process remaining groups
    for (var g = 1u; g < G; g = g + 1u) {
        let offset = g * 64u + tid;
        s_cand_v[tid] = IntermediateVals[offset];
        s_cand_i[tid] = IntermediateIDs[offset];
        workgroupBarrier();

        bitonic_sort_cand(tid);
        if (s_cand_v[0] > s_top_v[63]) {
            bitonic_merge_top_cand(tid);
        }
    }

    // Write final sorted top 64 to output
    OutIDs[tid] = s_top_i[tid];
    OutVals[tid] = s_top_v[tid];
}
