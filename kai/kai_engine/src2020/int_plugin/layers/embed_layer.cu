/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "embed_layer.cuh"

EmbedLayer::EmbedLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    string dic_path = get_option("fileload", "");

    int64 voc_size = engine.get_dataset_ext_param("voc_size");
    int64 vec_size = engine.get_dataset_ext_param("vec_size");

    voc_size = get_option("voc_size", voc_size);
    vec_size = get_option("vec_size", vec_size);

    vector<int64> voc_sizes;

    voc_sizes.push_back(voc_size);

    Value plugin_val = get_option("plugin", None);

    if (plugin_val.type() == vt::list) {
        m_plugin_names = plugin_val;
        for (List::iterator it = m_plugin_names.begin(); it != m_plugin_names.end(); it++) {
            string name = *it;
            int64 plugin_size = engine.get_dataset_ext_param(name + "_size");
            voc_sizes.push_back(plugin_size);
        }
    }

    if (dic_path != "") {
        throw KaiException(KERR_ASSERT);
        //m_load_dictionary(dic_path);
    }
    else {
        m_param["dic"] = alloc_embed_param(voc_sizes, vec_size);
        //Array<float> weight = kmath->random_normal(0, rand_std, dshape);
        //kmath->set_row(weight, 0, 0);
    }

    shape = m_output_shape = shape.replace_nth(-1, vec_size);
}

EmbedLayer::~EmbedLayer() {
}

Dict EmbedLayer::m_get_wvec_param() {
    return m_param["dic"];
}

Dict EmbedLayer::m_forward_main(Dict hidden) {
    Array<int64> x = hidden["wids"];
    Array<float> y = m_forward_narr(x);
    Dict hout = Value::wrap_dict("data", y);

    return hout;
}

Dict EmbedLayer::m_backprop_main(Dict G_hidden) {
    Array<float> gy = G_hidden["data"];
    m_backprop_narr(gy);
    return Dict();
}

// default-data 이외의 성분을 전처리 과정에 포함시키려면 아래 메서드를 재정의할 것
Dict EmbedLayer::m_forward_cuda_main(Dict hidden) {
    Array<int64> x = hidden["wids"];
    Array<float> y = m_forward_cuda_narr(x);
    Dict hout = Value::wrap_dict("data", y);

    return hout;
}

Dict EmbedLayer::m_backprop_cuda_main(Dict G_hidden) {
    Array<float> gy = G_hidden["data"];
    m_backprop_cuda_narr(gy);
    return Dict();
}

Array<float> EmbedLayer::m_forward_narr(Array<int64> hidden) {
    if (m_trace) hidden.print("EmbedLayer::m_forward::hidden");

    Array<float> winvecs = m_engine.get_optimizer()->forward_embed(m_param["dic"], hidden);
    /*
    Array<float> weight = m_pm["w"];

    if (m_trace) weight.print("EmbedLayer::m_forward::weight");
    if (m_trace) {
        logger.Print("m_dic_count = %lld", m_dic_count);
        for (int64 n = 0; n < m_dic_count; n++) {
            logger.Print("m_voc_counts[%d] = %lld", n, m_voc_counts[n]);
        }
    }
    Array<float> winvecs = weight.wvec_select_idx(hidden, m_dic_count, m_voc_counts);
    if (m_trace) winvecs.print("EmbedLayer::m_forward::winvecs");
    */

    m_aux["wids"] = hidden;

    return winvecs;
}

Array<float> EmbedLayer::m_backprop_narr(Array<float> G_hidden) {
    Array<int64> wids = m_aux["wids"];
    m_update_embed(m_param["dic"], G_hidden, wids);
    return kmath->zeros(wids.shape());
}

int64 EmbedLayer::dump_structure(int64 depth) {
    int64 param_cnt;
    string dic_desc = m_get_embed_param_desc(m_param["dic"], &param_cnt);
    logger.Print("%*s%s: %s(%lld) : %s => %s : %s => %lld pms",
        depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(), dic_desc.c_str(), param_cnt);
    return param_cnt;
}

Array<float> EmbedLayer::m_forward_cuda_narr(Array<int64> hidden) {
    CudaConn cuda("forward", this);

    int64 mb_size = hidden.axis_size(0);
    int64 vec_size = m_output_shape[-1];

    Shape vshape = m_output_shape.add_front(mb_size);
    
    Array<float> word_vecs = CudaConn::CreateFloatArray(vshape, "vec");

    Array<int64> wids = cuda.ToCudaArray(hidden, "hidden");

    m_engine.get_optimizer()->forward_embed_cuda(m_param["dic"], word_vecs, wids);

    m_aux["wids"] = wids;

    Array<float> output = word_vecs;

    return output;
}

Array<float> EmbedLayer::m_backprop_cuda_narr(Array<float> G_hidden) {
    Array<int64> wids = m_aux["wids"];

    m_update_embed(m_param["dic"], G_hidden, wids);

    return kmath->zeros(wids.shape());
}
