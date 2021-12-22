/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "reshape_layer.cuh"

ReshapeLayer::ReshapeLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    List slist = get_option("shape");
    Shape out_shape(slist);
    assert(shape.total_size() == out_shape.total_size());
    shape = out_shape;
    m_output_shape = shape;
}

ReshapeLayer::~ReshapeLayer() {
}

Array<float> ReshapeLayer::m_forward_farr(Array<float> hidden) {
    int64 mb_size = hidden.axis_size(0);
    Shape out_shape = m_output_shape.add_front(mb_size);
    hidden = hidden.reshape(out_shape);
    return hidden;
}

Array<float> ReshapeLayer::m_backprop_farr(Array<float> G_hidden) {
    int64 mb_size = G_hidden.axis_size(0);
    Shape in_shape = m_input_shape.add_front(mb_size);
    G_hidden = G_hidden.reshape(in_shape);
    return G_hidden;
}

Array<float> ReshapeLayer::m_forward_cuda_farr(Array<float> hidden) {
    int64 mb_size = hidden.axis_size(0);
    Shape out_shape = m_output_shape.add_front(mb_size);
    hidden = hidden.reshape(out_shape);
    return hidden;
}

Array<float> ReshapeLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    int64 mb_size = G_hidden.axis_size(0);
    Shape in_shape = m_input_shape.add_front(mb_size);
    G_hidden = G_hidden.reshape(in_shape);
    return G_hidden;
}
