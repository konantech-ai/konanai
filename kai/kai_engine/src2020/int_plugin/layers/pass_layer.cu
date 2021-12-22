/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "pass_layer.cuh"

PassLayer::PassLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine) : ComplexLayer(options, layer_subnet, shape, seq, engine) {
    Shape subnet_shape = shape;
    bool subnet_seq = seq;
    List pms;

    engine.build_hidden_net(layer_subnet, subnet_shape, subnet_seq, m_layers, pms);

    m_param["pms"] = pms;
}

PassLayer::~PassLayer() {
}

Array<float> PassLayer::m_forward_farr(Array<float> hidden) {
    //hidden.print("Pass: hidden in");
    if (m_trace) hidden.print("PassLayer::m_forward::hidden in");
    Array<float> sub_hidden = hidden.deepcopy();
    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        sub_hidden = pLayer->forward_subnet(sub_hidden);
        if (m_trace) sub_hidden.print("PassLayer::m_forward::sub_hidden step");
    }
    if (m_trace) hidden.print("PassLayer::m_forward::hidden out");

    //hidden.print("PassLayer::m_forward::hidden out");
    return hidden;
}

Array<float> PassLayer::m_backprop_farr(Array<float> G_hidden) {
    if (m_trace) G_hidden.print("PassLayer::m_backprop::G_hidden in");
    Array<float> G_sub_hidden;
    for (vector<Layer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); it++) {
		Layer* pLayer = *it;
        G_sub_hidden = pLayer->backprop_subnet(G_sub_hidden);
        if (m_trace) G_sub_hidden.print("PassLayer::m_backprop::G_sub_hidden steo");
    }

    G_hidden += G_sub_hidden;
    if (m_trace) G_hidden.print("PassLayer::m_backprop::G_hidden out");

	return G_hidden;
}

Array<float> PassLayer::m_forward_cuda_farr(Array<float> hidden) {
    Array<float> sub_hidden = CudaConn::Copy(hidden.data_ptr(), "deepcopy");

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        sub_hidden = pLayer->forward_subnet(sub_hidden);
        if (m_trace) sub_hidden.print("PassLayer::m_forward::sub_hidden step");
    }

    if (m_trace) hidden.print("PassLayer::m_forward::hidden out");
    return hidden;
}

Array<float> PassLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    if (m_trace) G_hidden.print("PassLayer::m_backprop::G_hidden in");

    if (m_layers.size() == 0) return G_hidden;

    Array<float> G_sub_hidden;
    for (vector<Layer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); it++) {
        Layer* pLayer = *it;
        G_sub_hidden = pLayer->backprop_subnet(G_sub_hidden);
        if (m_trace) G_sub_hidden.print("PassLayer::m_backprop::G_sub_hidden steo");
    }

    int64 hisze = G_hidden.total_size();

    cu_call(ker_add_on, hisze, (hisze, G_hidden.data_ptr(), G_sub_hidden.data_ptr()));

    if (m_trace) G_hidden.print("PassLayer::m_backprop::G_hidden out");

    return G_hidden;
}
