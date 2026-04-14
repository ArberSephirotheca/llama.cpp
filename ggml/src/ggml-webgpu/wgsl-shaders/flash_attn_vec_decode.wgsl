diagnostic(off, subgroup_uniformity);
enable f16;
enable subgroups;

#define HEAD_DIM_QK 64
#define HEAD_DIM_V 64
#define Q_TILE 1
#define KV_TILE 64
#define WG_SIZE 32

#define Q_VECS (HEAD_DIM_QK / 4)
#define V_VECS (HEAD_DIM_V / 4)
#define Q_VECS_PER_THREAD ((Q_VECS + WG_SIZE - 1) / WG_SIZE)
#define V_VECS_PER_THREAD ((V_VECS + WG_SIZE - 1) / WG_SIZE)
#define SCORE_SLOTS ((KV_TILE + WG_SIZE - 1) / WG_SIZE)

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_sinks: u32,
    offset_dst: u32,

    n_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,

    stride_q1: u32,
    stride_q2: u32,
    stride_q3: u32,
    stride_k1: u32,
    stride_k2: u32,
    stride_k3: u32,
    stride_v1: u32,
    stride_v2: u32,
    stride_v3: u32,
    stride_mask3: u32,

    q_per_kv: u32,

    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,

    tmp_data_base: u32,
    tmp_stats_base: u32,
    nwg: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> V: array<vec4<f16>>;
#if defined(MASK) && defined(SINKS)
@group(0) @binding(3) var<storage, read> mask: array<f16>;
@group(0) @binding(4) var<storage, read> sinks: array<f32>;
#define TMP_BINDING 5
#define DST_BINDING 6
#define PARAMS_BINDING 7
#elif defined(MASK)
@group(0) @binding(3) var<storage, read> mask: array<f16>;
#define TMP_BINDING 4
#define DST_BINDING 5
#define PARAMS_BINDING 6
#elif defined(SINKS)
@group(0) @binding(3) var<storage, read> sinks: array<f32>;
#define TMP_BINDING 4
#define DST_BINDING 5
#define PARAMS_BINDING 6
#else
#define TMP_BINDING 3
#define DST_BINDING 4
#define PARAMS_BINDING 5
#endif

@group(0) @binding(TMP_BINDING) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;

const FLOAT_MIN: f32 = -1.0e9;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
    if (subgroup_id != 0u || subgroup_size != WG_SIZE) {
        return;
    }

    let wg_per_head = params.seq_len_q;
    let wg_per_batch = wg_per_head * params.n_heads;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;

    let iwg = wg_id.x % params.nwg;
    let base_wg_id = wg_id.x / params.nwg;

    let batch_idx = base_wg_id / wg_per_batch;
    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
    let wg_in_batch = base_wg_id % wg_per_batch;

    let head_idx = wg_in_batch / wg_per_head;
    let q_row = wg_in_batch % wg_per_head;
    if (q_row >= params.seq_len_q) {
        return;
    }

    let q_head_offset = q_batch_offset + head_idx * params.stride_q2 + q_row * params.stride_q1;
    let k_head_idx = head_idx / params.q_per_kv;
    let v_head_idx = k_head_idx;
    let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;
    let v_head_offset = v_batch_offset + v_head_idx * params.stride_v2;

#ifdef MASK
    let mask_global_offset = params.offset_mask + batch_idx * params.stride_mask3 + q_row * params.seq_len_kv;
#endif

    let head = f32(head_idx);
    let has_bias = params.max_bias > 0.0;
    let slope = select(1.0,
                       select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0),
                              pow(params.m0, head + 1.0),
                              head < params.n_head_log2),
                       has_bias);

    var q_reg: array<vec4<f32>, Q_VECS_PER_THREAD>;
    for (var i = 0u; i < Q_VECS_PER_THREAD; i += 1u) {
        let q_vec = sg_inv_id + i * WG_SIZE;
        if (q_vec < Q_VECS) {
            let q_base = q_head_offset + q_vec * 4u;
            q_reg[i] = vec4<f32>(Q[q_base + 0u], Q[q_base + 1u], Q[q_base + 2u], Q[q_base + 3u]) * params.scale;
        } else {
            q_reg[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    }

    var out_reg: array<vec4<f32>, V_VECS_PER_THREAD>;
    for (var i = 0u; i < V_VECS_PER_THREAD; i += 1u) {
        out_reg[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var row_max = FLOAT_MIN;
    var exp_sum = 0.0;

    let tile_stride = params.nwg * KV_TILE;
    for (var kv_tile = iwg * KV_TILE; kv_tile < params.seq_len_kv; kv_tile += tile_stride) {
        var scores: array<f32, SCORE_SLOTS>;
        var local_max = FLOAT_MIN;

        for (var slot = 0u; slot < SCORE_SLOTS; slot += 1u) {
            let kv_rel = slot * WG_SIZE + sg_inv_id;
            let k_row = kv_tile + kv_rel;
            var score = FLOAT_MIN;
            if (kv_rel < KV_TILE && k_row < params.seq_len_kv) {
                var partial = 0.0;
                for (var i = 0u; i < Q_VECS_PER_THREAD; i += 1u) {
                    let q_vec = sg_inv_id + i * WG_SIZE;
                    if (q_vec < Q_VECS) {
                        let k_vec_index = (k_head_offset + k_row * params.stride_k1 + q_vec * 4u) >> 2u;
                        partial += dot(q_reg[i], vec4<f32>(K[k_vec_index]));
                    }
                }
                score = subgroupAdd(partial);
#ifdef LOGIT_SOFTCAP
                score = params.logit_softcap * tanh(score);
#endif
#ifdef MASK
                let mask_val = f32(mask[mask_global_offset + k_row]);
                score += select(mask_val, slope * mask_val, has_bias);
#endif
            }
            scores[slot] = score;
            local_max = max(local_max, score);
        }

        let new_max = max(row_max, subgroupMax(local_max));
        let scale = exp(row_max - new_max);
        exp_sum *= scale;
        for (var i = 0u; i < V_VECS_PER_THREAD; i += 1u) {
            out_reg[i] *= scale;
        }

        var local_sum = 0.0;
        for (var slot = 0u; slot < SCORE_SLOTS; slot += 1u) {
            let kv_rel = slot * WG_SIZE + sg_inv_id;
            let v_row = kv_tile + kv_rel;
            if (kv_rel >= KV_TILE || v_row >= params.seq_len_kv) {
                continue;
            }

            let weight = exp(scores[slot] - new_max);
            local_sum += weight;
            for (var i = 0u; i < V_VECS_PER_THREAD; i += 1u) {
                let v_vec = sg_inv_id + i * WG_SIZE;
                if (v_vec < V_VECS) {
                    let v_idx = (v_head_offset + v_row * params.stride_v1 + v_vec * 4u) >> 2u;
                    out_reg[i] += weight * vec4<f32>(V[v_idx]);
                }
            }
        }

        exp_sum += subgroupAdd(local_sum);
        row_max = new_max;
    }

#ifdef SINKS
    let sink_score = select(FLOAT_MIN, sinks[params.offset_sinks + head_idx], sg_inv_id == 0u);
    let sink_max = subgroupMax(max(row_max, sink_score));
    let sink_scale = exp(row_max - sink_max);
    let sink_weight = exp(sink_score - sink_max);
    exp_sum = exp_sum * sink_scale + subgroupAdd(sink_weight);
    row_max = sink_max;
    for (var i = 0u; i < V_VECS_PER_THREAD; i += 1u) {
        out_reg[i] *= sink_scale;
    }
#endif

    if (params.nwg == 1u) {
        let inv_sum = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
        let row_base = params.offset_dst + batch_idx * dst3_stride + q_row * dst2_stride + head_idx * HEAD_DIM_V;
        for (var i = 0u; i < V_VECS_PER_THREAD; i += 1u) {
            let v_vec = sg_inv_id + i * WG_SIZE;
            if (v_vec < V_VECS) {
                let dst_vec_index = (row_base + v_vec * 4u) >> 2u;
                dst[dst_vec_index] = out_reg[i] * inv_sum;
            }
        }
    } else {
        let rid = batch_idx * params.n_heads * params.seq_len_q + head_idx * params.seq_len_q + q_row;
        let tmp_row_data_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg) + iwg * HEAD_DIM_V;
        let tmp_row_stats_base = params.tmp_stats_base + rid * (2u * params.nwg) + 2u * iwg;

        for (var i = 0u; i < V_VECS_PER_THREAD; i += 1u) {
            let v_vec = sg_inv_id + i * WG_SIZE;
            if (v_vec < V_VECS) {
                let out_base = tmp_row_data_base + v_vec * 4u;
                tmp[out_base + 0u] = out_reg[i].x;
                tmp[out_base + 1u] = out_reg[i].y;
                tmp[out_base + 2u] = out_reg[i].z;
                tmp[out_base + 3u] = out_reg[i].w;
            }
        }

        if (sg_inv_id == 0u) {
            tmp[tmp_row_stats_base + 0u] = exp_sum;
            tmp[tmp_row_stats_base + 1u] = row_max;
        }
    }
}
