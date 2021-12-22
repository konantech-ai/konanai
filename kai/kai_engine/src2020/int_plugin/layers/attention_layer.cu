/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "attention_layer.cuh"
#include "dropout_layer.cuh"

AttentionLayer::AttentionLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine), m_pDropoutLayers(NULL) {
    if (!get_option("multi-head", true)) throw KaiException(KERR_ASSERT);

    m_num_heads = get_option("attention_heads", 8);

    int64 L = shape[0]; // timesteps
    int64 V = shape[1]; // word vector
    int64 H = m_num_heads;
    int64 R = V / H;

    assert(V % H == 0);

    m_param["QKV"] = alloc_affine_param(Shape(V, 3 * H * R), true);
    m_param["O"] = alloc_affine_param(Shape(H * R, V), true);

    m_coef = 1.0f / sqrt(float(R));

    float dropout_ratio = get_option("dropout", 0);

    if (dropout_ratio > 0) {
        Dict drop_options;
        drop_options["drop_ratio"] = dropout_ratio;
        Shape bn_shape(H, L, L);
        bool bn_seq = false;
        m_pDropoutLayers = new DropoutLayer(drop_options, bn_shape, bn_seq, engine);
    }
}

AttentionLayer::~AttentionLayer() {
    delete m_pDropoutLayers;
}

Array<float> AttentionLayer::m_forward_farr(Array<float> hidden) {
    FuncTimer("AttentionLayer::m_forward_farr");

    assert(hidden.dim() == 3);

    int B = (int)hidden.axis_size(0);       // mb_size
    int L = (int)hidden.axis_size(1);       // timesteps
    int V = (int)hidden.axis_size(2);       // word vector
    int H = m_num_heads;
    int R = V / H;

    Array<float> hidden_2d = hidden.reshape(Shape(-1, V));

    Array<float> qkv = forward_affine(m_param["QKV"], hidden_2d);

    Array<float> query, key, value;

    query = qkv[Axis(_all_, Ax(0 * V, 1 * V))];
    key   = qkv[Axis(_all_, Ax(1 * V, 2 * V))];
    value = qkv[Axis(_all_, Ax(2 * V, 3 * V))];

    query = query.reshape(Shape(B, L, H, R)).transpose(Idx(0, 2, 1, 3)); // [B, H, L, R]
    key = key.reshape(Shape(B, L, H, R)).transpose(Idx(0, 2, 3, 1)); // [B, H, R, L]

    Array<float> att_score = kmath->matmul(query, key) * m_coef;

    if (m_trace) att_score.print("AttentionLayer::m_forward::att_score");

    // 시간 순으로 과거와 현재 항목만 바로볼 수 있게 마스킹 처리함
    // att_score = kmath->mask_future_timesteps(att_score, L);

    Array<float> att_prob1 = kmath->softmax(att_score);
    Array<float> att_prob2 = att_prob1;

    if (m_pDropoutLayers) {
        att_prob2 = m_pDropoutLayers->forward_subnet(att_prob1);
    }

    value = value.reshape(Shape(B, L, H, R)).transpose(Idx(0, 2, 1, 3)); // [B, H, L, R]

    Array<float> att_out = kmath->matmul(att_prob2, value);
    att_out = att_out.transpose(Idx(0, 2, 1, 3)); // [B, L, H, R]
    att_out = att_out.reshape(Shape(B * L, H * R)); // [B*L, H*R]

    Array<float> delta = forward_affine(m_param["O"], att_out); // [B*L, d_model(H*R)]

    m_aux["hidden_2d"] = hidden_2d;
    m_aux["key"] = key;
    m_aux["query"] = query;
    m_aux["value"] = value;
    m_aux["att_out"] = att_out;
    m_aux["att_prob1"] = att_prob1;
    m_aux["att_prob2"] = att_prob2;

    hidden = delta.reshape(hidden.shape());

    return hidden;
}

Array<float> AttentionLayer::m_backprop_farr(Array<float> G_hidden) {
    FuncTimer("AttentionLayer::m_backprop_farr");

    int B = (int)G_hidden.axis_size(0);        // mb_size
    int L = (int)G_hidden.axis_size(1);        // timesteps
    int V = (int)G_hidden.axis_size(2);  // word vector
    int H = m_num_heads;
    int R = V / H;

    Array<float> hidden_2d = m_aux["hidden_2d"];
    Array<float> key = m_aux["key"];
    Array<float> query = m_aux["query"];
    Array<float> value = m_aux["value"];
    Array<float> att_out = m_aux["att_out"];
    Array<float> att_prob1 = m_aux["att_prob1"];
    Array<float> att_prob2 = m_aux["att_prob2"];

    Dict pm_O = m_param["O"];

    Array<float> G_delta = G_hidden.reshape(Shape(-1, V));
    Array<float> G_att_out = backprop_affine(pm_O, att_out, G_delta);

    G_att_out = G_att_out.reshape(Shape(B, L, H, R)).transpose(Idx(0, 2, 1, 3)); // [B, H, L, R]

    Array<float> G_att_prob = kmath->matmul(G_att_out, value.transpose(Idx(0, 1, 3, 2)));
    Array<float> G_value = kmath->matmul(att_prob2.transpose(Idx(0, 1, 3, 2)), G_att_out);

    G_value = G_value.transpose(Idx(0, 2, 1, 3)).reshape(Shape(B * L, H * R));

    if (m_pDropoutLayers) {
        G_att_prob = m_pDropoutLayers->backprop_subnet(G_att_prob);
    }

    Array<float> G_att_prob_flat = G_att_prob.reshape(Shape(-1, L));
    Array<float> g_att_prob_flat = kmath->softmax_derv(att_prob1.reshape(Shape(-1, L)));
    Array<float> G_att_score_flat = kmath->dotmul(g_att_prob_flat, G_att_prob_flat);
    Array<float> G_att_score = G_att_score_flat.reshape(Shape(B, H, L, L));
    
    // 마스크된 항목들은 어차피 경사도 0 값이 돌아오고 있어서 경사도에 대한 마스크 처리는 불필요
    //Array<float> G_att_score = kmath->mask_future_timesteps(G_att_score1, L, false);

    Array<float> G_query = kmath->matmul(G_att_score, key.transpose(Idx(0, 1, 3, 2))) * m_coef;
    Array<float> G_key = kmath->matmul(query.transpose(Idx(0, 1, 3, 2)), G_att_score) * m_coef;

    G_query = G_query.transpose(Idx(0, 2, 1, 3)).reshape(Shape(B * L, H * R));
    G_key = G_key.transpose(Idx(0, 3, 1, 2)).reshape(Shape(B * L, H * R));

    Array<float> G_qkv = kmath->zeros(Shape(G_key.axis_size(0), 3 * V));

    G_qkv[Axis(_all_, Ax(0 * V, 1 * V))] = G_query;
    G_qkv[Axis(_all_, Ax(1 * V, 2 * V))] = G_key;
    G_qkv[Axis(_all_, Ax(2 * V, 3 * V))] = G_value;

    Dict pm_QKV = m_param["QKV"];

    Array<float> G_hidden_2d = backprop_affine(pm_QKV, hidden_2d, G_qkv);
    Array<float> G_input = G_hidden_2d.reshape(Shape(B, L, V));

    return G_input;
}

int64 AttentionLayer::dump_structure(int64 depth) {
    int64 pm_cnt_QKV, pm_cnt_O;

    string pm_desc_QKV = m_get_affine_param_desc(m_param["QKV"], &pm_cnt_QKV);
    string pm_desc_O = m_get_affine_param_desc(m_param["O"], &pm_cnt_O);

    int64 param_cnt = pm_cnt_QKV + pm_cnt_O;

    logger.Print("%*s%s: %s(%d) : %s => %s : (QKV:%s+OUTPUT:%s)=%lld pms",
        depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(),
        pm_desc_QKV.c_str(), pm_desc_O.c_str(),  param_cnt);

    return param_cnt;
    /*
    Dict pm_O = m_param["O"];
    Array<float> w = pm_o["w"];
    Shape pmshape = w.shape();
    int64 param_cnt = (pmshape[0] * pmshape[1] + pmshape[1]) * 4;
    logger.Print("%*s%s: %s(%d) : %s => %s : ((%lldx%lld)+%lld)*4=%lld pms",
        depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(),
        pmshape[0], pmshape[1], pmshape[1], param_cnt);
    return param_cnt;
    */
}


Array<float> AttentionLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    m_aux["input"] = hidden;

    int64 B = (int)hidden.axis_size(0);        // mb_size
    int64 L = (int)hidden.axis_size(1);        // timesteps
    int64 V = (int)hidden.axis_size(2);        // word vector
    int64 H = m_num_heads;
    int64 R = V / H;

    float keep_ratio = 1.0;

    //Dict pm_qkv = m_pm["QKV"];
    //Dict pm_o = m_pm["O"];

    float* cuda_h = cuda.copy_to_buffer(hidden, "x_clone");

    //float* cuda_qkv_w = CudaConn::GetCudaMem(pm_qkv["w"], "w_qkv");
    //float* cuda_qkv_b = CudaConn::GetCudaMem(pm_qkv["b"], "b_qkv");
    //float* cuda_o_w = CudaConn::GetCudaMem(pm_o["w"], "w_o");
    //float* cuda_o_b = CudaConn::GetCudaMem(pm_o["b"], "b_o");

    float* cuda_qkv = cuda.alloc_float_mem(Shape(B, L, 3 * V), "qkv");
    float* cuda_query = cuda.alloc_float_mem(Shape(B, H, L, R), "q");
    float* cuda_key = cuda.alloc_float_mem(Shape(B, H, R, L), "k");
    float* cuda_value = cuda.alloc_float_mem(Shape(B, H, L, R), "value");
    float* cuda_att_score = cuda.alloc_float_mem(Shape(B, H, L, L), "score");
    float* cuda_att_probs1 = cuda.alloc_float_mem(Shape(B, H, L, L), "probs1");
    float* cuda_att_probs2 = cuda.alloc_float_mem(Shape(B, H, L, L), "probs2");
    float* cuda_att_mul = cuda.alloc_float_mem(Shape(B * H, L, R), "mul");
    float* cuda_att_out = cuda.alloc_float_mem(Shape(B * L, H * R), "out");
    float* cuda_output = cuda.alloc_float_mem(Shape(B, L, V), "#netout#");

    float* cuda_dm = NULL;

    int64 qkvsize = B * L * 3 * V;
    int64 attsize = B * H * L * L;
    int64 outsize = B * L * H * R;

    forward_affine_cuda(m_param["QKV"], true, cuda_qkv, cuda_h, B * L, V, 3 * V);
    /*
    cu_call(ker_matmul, qkvsize, (qkvsize, cuda_qkv, cuda_h, cuda_qkv_w, B * L, V, 3 * V));
    cu_call(ker_add_bias, qkvsize, (qkvsize, cuda_qkv, cuda_qkv_b, B * L, V, 3 * V));
    */

    cu_call(ker_attention_split, qkvsize, (qkvsize, cuda_query, cuda_key, cuda_value, cuda_qkv, L, H, R));
    cu_call(ker_multi_matmul, attsize, (attsize, cuda_att_score, cuda_query, cuda_key, B * H, L, R, L));
    cu_call(ker_mult_scalar_on, attsize, (attsize, cuda_att_score, m_coef));

    // 시간 순으로 과거와 현재 항목만 바로볼 수 있게 마스킹 처리함
    //cu_call(ker_attention_mask_future, attsize, (attsize, cuda_att_score, L));

    cu_call(ker_softmax, attsize, (attsize, cuda_att_probs1, cuda_att_score, L));

    if (m_pDropoutLayers) {
        keep_ratio = m_pDropoutLayers->m_keep_prob;
        throw KaiException(KERR_ASSERT);
        // m_pDropoutLayers->create_mask() 인자구조를 바꾸었으니 필요한 내용을 확인해 아래 줄의 Shape(B, L) 인자를 알맞게 수정할 것
        Array<float> drop_mask = m_pDropoutLayers->create_mask(Shape(B, L));
        cuda_dm = drop_mask.data_ptr();
        m_aux["drop_mask"] = drop_mask;
        cu_call(ker_dropout, attsize, (attsize, cuda_att_probs2, cuda_att_probs1, cuda_dm, keep_ratio));
    }
    else {
        cu_call(ker_copy_to, attsize, (attsize, cuda_att_probs2, cuda_att_probs1));
    }

    cu_call(ker_multi_matmul, outsize, (outsize, cuda_att_mul, cuda_att_probs2, cuda_value, B * H, L, L, R));
    cu_call(ker_attention_reshape_out, outsize, (outsize, cuda_att_out, cuda_att_mul, L, H, R));

    forward_affine_cuda(m_param["O"], true, cuda_output, cuda_att_out, B * L, V, V);
    /*
    cu_call(ker_matmul, outsize, (outsize, cuda_output, cuda_att_out, cuda_o_w, B * L, V, V));
    cu_call(ker_add_bias, outsize, (outsize, cuda_output, cuda_o_b, B * L, V, V));
    */

    m_aux["att_out"] = cuda.detach(cuda_att_out, "att_out");
    m_aux["att_probs1"] = cuda.detach(cuda_att_probs1, "att_probs before dropout");
    m_aux["att_probs2"] = cuda.detach(cuda_att_probs2, "att_probs after dropout");
    m_aux["query"] = cuda.detach(cuda_query, "query");
    m_aux["key"] = cuda.detach(cuda_key, "key");
    m_aux["value"] = cuda.detach(cuda_value, "value");
    m_aux["x"] = cuda.detach(cuda_h, "x");

    Array<float> output = cuda.detach(cuda_output, "#netout#");

    return output;
}

Array<float> AttentionLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    FuncTimer("AttentionLayer::m_backprop_cuda_farr");

    if (m_trace) G_hidden.print("AttentionLayer::m_backprop::G_hidden");

    CudaConn cuda("backprop", this);

    int64 B = (int)G_hidden.axis_size(0);        // mb_size
    int64 L = (int)G_hidden.axis_size(1);        // timesteps
    int64 V = (int)G_hidden.axis_size(2);        // word vector
    int64 H = m_num_heads;
    int64 R = V / H;

    if (m_trace) logger.Print("AttentionLayer::m_backprop::B = %d", B);
    if (m_trace) logger.Print("AttentionLayer::m_backprop::L = %d", L);
    if (m_trace) logger.Print("AttentionLayer::m_backprop::V = %d", V);
    if (m_trace) logger.Print("AttentionLayer::m_backprop::H = %d", H);
    if (m_trace) logger.Print("AttentionLayer::m_backprop::R = %d", R);

    Shape att_out_shape(B * L, H * R);

    float* cuda_g_h = cuda.copy_to_buffer(G_hidden, "g_h");

    //Dict pm_qkv = m_pm["QKV"];
    //Dict pm_o = m_pm["O"];

    //float* cuda_qkv_w = CudaConn::GetCudaMem(pm_qkv["w"], "w_qkv");
    //float* cuda_qkv_b = CudaConn::GetCudaMem(pm_qkv["b"], "b_qkv");
    //float* cuda_o_w = CudaConn::GetCudaMem(pm_o["w"], "w_o");
    //float* cuda_o_b = CudaConn::GetCudaMem(pm_o["b"], "b_o");

    float* cuda_att_out = cuda.attach(m_aux["att_out"], "att_out");
    float* cuda_g_att_out = cuda.alloc_float_mem(att_out_shape, "g_att_out");
    //float* cuda_g_o_w = cuda.alloc_float_mem(Shape(V, V), "gow");
    //float* cuda_g_o_b = cuda.alloc_float_mem(Shape(V), "gob");

    int64 outsize = att_out_shape.total_size();
    //int64 owsize = V * V;
    //int64 obsize = V;

    backprop_affine_cuda(m_param["O"], true, cuda_g_att_out, cuda_g_h, cuda_att_out, B * L, V, V);
    /*
    cu_call(ker_matmul_derv_x, outsize, (outsize, cuda_g_att_out, cuda_g_h, cuda_o_w, B * L, V, V));
    cu_call(ker_matmul_derv_w, owsize, (owsize, cuda_g_o_w, cuda_g_h, cuda_att_out, B * L, V, V));
    cu_call(ker_add_bias_derv, obsize, (obsize, cuda_g_o_b, cuda_g_h, B * L, V, V));
    cuda.optimize(pm_o, "w", m_engine, cuda_g_o_w);
    cuda.optimize(pm_o, "b", m_engine, cuda_g_o_b);
    */

    float* cuda_value = cuda.attach(m_aux["value"], "value");
    float* cuda_att_probs2 = cuda.attach(m_aux["att_probs2"], "att_probs after dropout");

    float* cuda_g_att_mul = cuda.alloc_float_mem(Shape(B, H, L, R), "mul");
    float* cuda_g_att_probs1 = cuda.alloc_float_mem(Shape(B, H, L, L), "probs1");
    float* cuda_g_att_probs2 = cuda.alloc_float_mem(Shape(B, H, L, L), "probs2");
    float* cuda_g_value = cuda.alloc_float_mem(Shape(B, H, L, R), "value");

    int64 attsize = B * H * L * L;

    cu_call(ker_attention_reshape_mul, outsize, (outsize, cuda_g_att_mul, cuda_g_att_out, L, H, R));
    cu_call(ker_multi_matmul_derv_x, attsize, (attsize, cuda_g_att_probs2, cuda_g_att_mul, cuda_value, B * H, L, L, R));
    cu_call(ker_multi_matmul_derv_w, outsize, (outsize, cuda_g_value, cuda_g_att_mul, cuda_att_probs2, B * H, L, L, R));

    if (m_pDropoutLayers) {
        float keep_ratio = m_pDropoutLayers->m_keep_prob;
        Array<float> drop_mask = m_aux["drop_mask"];
        float* cuda_dm = drop_mask.data_ptr();
        cu_call(ker_dropout_derv, attsize, (attsize, cuda_g_att_probs1, cuda_g_att_probs2, cuda_dm, keep_ratio));
        if (m_trace) cuda.DumpArr(cuda_g_att_probs1, "g_att_probs after dropout");
    }
    else {
        cu_call(ker_copy_to, attsize, (attsize, cuda_g_att_probs1, cuda_g_att_probs2));
    }

    float* cuda_g_att_score = cuda.alloc_float_mem(Shape(B, H, L, L), "score");
    float* cuda_att_probs1 = cuda.attach(m_aux["att_probs1"], "att_probs before dropout");

    cu_call(ker_softmax_derv, attsize, (attsize, cuda_g_att_score, cuda_g_att_probs1, cuda_att_probs1, L));

    cu_call(ker_mult_scalar_on, attsize, (attsize, cuda_g_att_score, m_coef));

    float* cuda_query = cuda.attach(m_aux["query"], "query");
    float* cuda_key = cuda.attach(m_aux["key"], "key");
    float* cuda_g_query = cuda.alloc_float_mem(Shape(B, H, L, R), "g_query");
    float* cuda_g_key = cuda.alloc_float_mem(Shape(B, H, R, L), "g_key");

    cu_call(ker_multi_matmul_derv_x, outsize, (outsize, cuda_g_query, cuda_g_att_score, cuda_key, B * H, L, R, L));
    cu_call(ker_multi_matmul_derv_w, outsize, (outsize, cuda_g_key, cuda_g_att_score, cuda_query, B * H, L, R, L));

    if (m_trace) cuda.DumpArr(cuda_g_query, "g_query");
    if (m_trace) cuda.DumpArr(cuda_g_key, "g_key");

    float* cuda_x = cuda.attach(m_aux["x"], "x");
    float* cuda_g_qkv = cuda.alloc_float_mem(Shape(B * L, 3 * V), "g_qkv");
    float* cuda_g_qkv_w = cuda.alloc_float_mem(Shape(V, 3 * V), "g_qkv_w");
    float* cuda_g_qkv_b = cuda.alloc_float_mem(Shape(3 * V), "g_qkv_b");

    int64 qkvsize = B * L * 3 * V;

    cu_call(ker_attention_combine, qkvsize, (qkvsize, cuda_g_qkv, cuda_g_query, cuda_g_key, cuda_g_value, L, H, R));

    backprop_affine_cuda(m_param["QKV"], true, cuda_g_h, cuda_g_qkv, cuda_x, B * L, V, 3 * V);
    /*
    cu_call(ker_matmul_derv_x, xsize, (xsize, cuda_g_h, cuda_g_qkv, cuda_qkv_w, B * L, V, 3 * V));
    cu_call(ker_matmul_derv_w, qkv_wsize, (qkv_wsize, cuda_g_qkv_w, cuda_g_qkv, cuda_x, B * L, V, 3 * V));
    cu_call(ker_add_bias_derv, qkv_bsize, (qkv_bsize, cuda_g_qkv_b, cuda_g_qkv, B * L, V, 3 * V));

    cuda.optimize(pm_qkv, "w", m_engine, cuda_g_qkv_w);
    cuda.optimize(pm_qkv, "b", m_engine, cuda_g_qkv_b);
    */

    Array<float> G_input = cuda.detach(cuda_g_h, "G_input");

    return G_input;
}