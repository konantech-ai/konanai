/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "embedding_layer.cuh"

EmbeddingLayer::EmbeddingLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine) : ComplexLayer(options, layer_subnet, shape, seq, engine) {
    List voc_sz_lst = m_engine.get_dataset_ext_param("voc_sizes");

    int64 dic_cnt = voc_sz_lst.size();

    vector<int64> voc_sizes;

    for (int64 n = 0; n < dic_cnt; n++) voc_sizes.push_back((int64)voc_sz_lst[n]);
    
    m_vec_size = get_option("vec_size");

    m_in_cnt = m_engine.get_dataset_ext_param("win_cnt");
    m_out_cnt = m_engine.get_dataset_ext_param("wout_cnt");

    assert(shape.size() == 2);
    assert(shape[0] == m_in_cnt + m_out_cnt);
    assert(shape[1] == dic_cnt);

    m_param["win"] = alloc_embed_param(voc_sizes, m_vec_size);
    m_param["wout"] = alloc_embed_param(voc_sizes, m_vec_size);

    List pms;

    shape = Shape(m_in_cnt, m_vec_size);
    engine.build_hidden_net(layer_subnet, shape, seq, m_layers, pms);

    assert(shape.size() == 1);
    assert(shape[0] == m_vec_size);

    m_param["pms"] = pms;

    shape = m_output_shape = Shape(m_out_cnt);
}

EmbeddingLayer::~EmbeddingLayer() {
}

Dict EmbeddingLayer::m_get_wvec_param() {
    return m_param["win"];
}

Dict EmbeddingLayer::m_forward_main(Dict hidden) {
    Array<int64> hint = hidden["hint"];
    Array<int64> noms = hidden["noms"];

    Array<float> out = m_forward_embedding(hint, noms);

    Dict hout = Value::wrap_dict("data", out);

    return hout;
}

Dict EmbeddingLayer::m_backprop_main(Dict G_hidden) {
    Array<float> gy = G_hidden["data"];

    m_backprop_embedding(gy);

    return Dict();
}

Dict EmbeddingLayer::m_forward_cuda_main(Dict hidden) {
    Array<int64> hint = hidden["hint"];
    Array<int64> noms = hidden["noms"];

    Array<float> out = m_forward_cuda_embedding(hint, noms);

    Dict hout = Value::wrap_dict("data", out);

    return hout;
}

Dict EmbeddingLayer::m_backprop_cuda_main(Dict G_hidden) {
    Array<float> gy = G_hidden["data"];

    m_backprop_cuda_embedding(gy);

    return Dict();
}

Array<float> EmbeddingLayer::m_forward_embedding(Array<int64> hint, Array<int64> noms) {
    Array<float> winvecs = m_engine.get_optimizer()->forward_embed(m_param["win"], hint);
    Array<float> woutvecs = m_engine.get_optimizer()->forward_embed(m_param["wout"], noms);

    /*
    Array<float> win_pm = m_fetch_embed(m_param["win"]);
    Array<float> wout_pm = m_fetch_embed(m_param["wout"]);

    //Array<int> wids = hidden.to_int();
    //Array<int> in_wids, out_wids;

    //in_wids = wids[Axis(_all_, Ax(0, m_in_cnt))];
    //out_wids = wids[Axis(_all_, Ax(m_in_cnt, m_in_cnt+m_out_cnt))];

    Array<float> winvecs = win_pm.wvec_select(hint);
    Array<float> woutvecs = wout_pm.wvec_select(noms);
    */

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        winvecs = pLayer->forward_subnet(winvecs);
    }

    Array<float> hidden = kmath->dotmul(winvecs, woutvecs);

    m_aux["hint"] = hint;
    m_aux["noms"] = noms;
    m_aux["winvecs"] = winvecs;
    m_aux["woutvecs"] = woutvecs;

    return hidden;
}

Array<float> EmbeddingLayer::m_backprop_embedding(Array<float> G_hidden) {
    Array<float> G_winvecs = kmath->dotmul_derv(G_hidden, m_aux["woutvecs"]);
    Array<float> G_woutvecs = kmath->dotmul_derv(G_hidden, m_aux["winvecs"]);

    int64 mb_size = G_winvecs.axis_size(0), vec_size = G_winvecs.axis_size(2);
    G_winvecs = G_winvecs.reshape(Shape(mb_size, vec_size));

    for (vector<Layer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); it++) {
        Layer* pLayer = *it;
        G_winvecs = pLayer->backprop_subnet(G_winvecs);
    }

    Array<int64> hint = m_aux["hint"];
    Array<int64> noms = m_aux["noms"];

    m_engine.get_optimizer()->regist_update_embed(m_param["win"], G_winvecs, hint);
    m_engine.get_optimizer()->regist_update_embed(m_param["wout"], G_woutvecs, noms);

    //backprop_embed(m_param["win"], hint, G_winvecs);
    //backprop_embed(m_param["wout"], noms, G_woutvecs);

    //update_param_select(win_dic, "w", m_aux["hint"], G_winvecs);
    //update_param_select(wout_dic, "w", m_aux["noms"], G_woutvecs);

    return Array<float>(Shape(mb_size, m_in_cnt + m_out_cnt));
}

Array<float> EmbeddingLayer::m_forward_cuda_embedding(Array<int64> hint, Array<int64> noms) {
    CudaConn cuda("forward", this);

    assert(m_in_cnt == hint.axis_size(1));
    assert(m_out_cnt == noms.axis_size(1));

    Array<int64> c_hint = cuda.ToCudaArray(hint, "hint");
    Array<int64> c_noms = cuda.ToCudaArray(noms, "noms");

    int64 mb_size = hint.axis_size(0);

    Array<float> winvecs = CudaConn::CreateFloatArray(Shape(mb_size, m_in_cnt, m_vec_size), "winvec");
    Array<float> woutvecs = CudaConn::CreateFloatArray(Shape(mb_size, m_out_cnt, m_vec_size), "woutvec");

    forward_embed_cuda(m_param["win"], winvecs, c_hint);
    forward_embed_cuda(m_param["wout"], woutvecs, c_noms);

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        winvecs = pLayer->forward_subnet(winvecs);
    }

    float* cuda_sub = winvecs.data_ptr();
    float* cuda_wov = woutvecs.data_ptr();

    float* cuda_out = cuda.alloc_float_mem(m_output_shape.add_front(mb_size), "#netout#");

    int64 dsize = mb_size * m_out_cnt;

    cu_call(ker_embedding_dotmul, dsize, (dsize, cuda_out, cuda_sub, cuda_wov, m_out_cnt, m_vec_size));

    m_aux["hint"] = c_hint;
    m_aux["noms"] = c_noms;
    m_aux["sub"] = winvecs;
    m_aux["wov"] = woutvecs;

    Array<float> output = cuda.detach(cuda_out, "outputs:rnn(for)");

    return output;
}

Array<float> EmbeddingLayer::m_backprop_cuda_embedding(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    int64 mb_size = G_hidden.axis_size(0);

    float* cuda_sub = cuda.attach(m_aux["sub"], "sub");
    float* cuda_wov = cuda.attach(m_aux["wov"], "wov");

    int64 vsize = mb_size * m_out_cnt * m_vec_size;

    Shape nshape = G_hidden.shape().append(m_vec_size);
    Shape sshape(mb_size, m_vec_size);

    float* cuda_gh = CudaConn::GetCudaMem(G_hidden, "gy");

    float* cuda_gsub = cuda.alloc_float_mem(sshape, "gsub");
    float* cuda_gwov = cuda.alloc_float_mem(nshape, "gwov");

    cu_call(ker_embedding_dotmul_derv, vsize, (vsize, cuda_gsub, cuda_gwov, cuda_gh, cuda_sub, cuda_wov, m_out_cnt, m_vec_size));

    Array<float> G_sub = cuda.detach(cuda_gsub, "gsub");
    Array<float> G_wov = cuda.detach(cuda_gwov, "gwov");

    for (vector<Layer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); it++) {
        Layer* pLayer = *it;
        G_sub = pLayer->backprop_subnet(G_sub);
    }

    //float* cuda_gwiv = cuda.attach(G_sub, "gwiv");

    Array<int64> hint = m_aux["hint"];
    Array<int64> noms = m_aux["noms"];

    backprop_embed_cuda(m_param["win"], G_sub, hint);
    backprop_embed_cuda(m_param["wout"], G_wov, noms);

    //Dict win_dic = m_pm["win"], wout_dic = m_pm["wout"];

    //cuda.DumpArr(cuda_hint, "hint back");
    //cuda.DumpArr(cuda_noms, "noms back");

    // ileegal memory access detected here!!!
    //cuda.optimize_select(win_dic, "w", m_engine, m_voc_size, mb_size * m_in_cnt, cuda_hint, cuda_gwiv);
    //cuda.optimize_select(wout_dic, "w", m_engine, m_voc_size, mb_size * m_out_cnt, cuda_noms, cuda_gwov);

    return kmath->zeros(Shape(mb_size, m_in_cnt + m_out_cnt));
}
