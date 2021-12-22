/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "serial_layer.cuh"

SerialLayer::SerialLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine) : ComplexLayer(options, layer_subnet, shape, seq, engine) {
    m_wrap_axis = get_option("wrap_axis", -1);
    if (m_wrap_axis != -1) {
        assert(m_wrap_axis == 0);
        m_wrap_size = (int) shape[0];
        shape = shape.remove_front();
    }

    List pms;
    int repeat = get_option("repeat", 1);

    for (int m = 0; m < repeat; m++) {
        engine.build_hidden_net(layer_subnet, shape, seq, m_layers, pms);
    }
    if (m_wrap_axis != -1) {
        shape = shape.add_front(m_wrap_size);
    }

    m_param["pms"] = pms;

    m_output_shape = shape;
}

SerialLayer::~SerialLayer() {
}

Array<float> SerialLayer::m_forward_farr(Array<float> hidden) {
    if (m_trace) hidden.print("hidden in");
    int64 mb_size = hidden.axis_size(0);
    if (m_wrap_axis != -1) {
        hidden = hidden.merge_time_axis();
        if (m_trace) hidden.print("hidden_merged");
    }
    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        hidden = pLayer->forward_subnet(hidden);
        if (m_trace) hidden.print("hidden step");
    }
    if (m_wrap_axis != -1) {
        hidden = hidden.split_time_axis(mb_size);
        if (m_trace) hidden.print("hidden unmerged");
    }
    if (m_trace) hidden.print("hidden out");
    return hidden;
}

Array<float> SerialLayer::m_backprop_farr(Array<float> G_hidden) {
    int64 mb_size = G_hidden.axis_size(0);
    if (m_trace) G_hidden.print("Serial: G_hidden in");
    if (m_wrap_axis != -1) {
        Shape tshape = G_hidden.shape().merge_time_axis();
        Array<float> G_hidden_temp = G_hidden.reshape(tshape);
        if (m_trace) G_hidden_temp.print("Serial: G_hidden_temp");
        G_hidden = G_hidden.merge_time_axis();
        if (m_trace) G_hidden.print("Serial: G_hidden merged");
    }
    for (vector<Layer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); it++) {
		Layer* pLayer = *it;
		G_hidden = pLayer->backprop_subnet(G_hidden);
        if (m_trace) G_hidden.print("Serial: G_hidden step");
    }
    if (m_wrap_axis != -1) {
        G_hidden = G_hidden.split_time_axis(mb_size);
        if (m_trace) G_hidden.print("Serial: G_hidden unmerged");
    }
    if (m_trace) G_hidden.print("Serial: G_hidden out");
    return G_hidden;
}

Array<float> SerialLayer::m_forward_cuda_farr(Array<float> hidden) {
    return m_forward_farr(hidden);
}

Array<float> SerialLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    return m_backprop_farr(G_hidden);
}
