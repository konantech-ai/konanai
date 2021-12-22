/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "expand_layer.cuh"

ExpandLayer::ExpandLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    m_ratio = get_2d_option("ratio");
    assert(shape.size() >= 2);
    assert(m_ratio[0] >= 1 && m_ratio[1] >= 1);
    shape[0] *= m_ratio[0];
    shape[1] *= m_ratio[1];
    m_output_shape = shape;
}

ExpandLayer::~ExpandLayer() {
}

Array<float> ExpandLayer::m_forward_farr(Array<float> hidden) {
    hidden.print("hidden");
    logger.Print("m_ratio = %s", m_ratio.desc().c_str());

    if (m_ratio[0] > 1) hidden = kmath->expand(hidden, m_ratio[0], 1);
    if (m_ratio[1] > 1) hidden = kmath->expand(hidden, m_ratio[1], 2);
    return hidden;
}

Array<float> ExpandLayer::m_backprop_farr(Array<float> G_hidden) {
    if (m_ratio[1] > 1) G_hidden = kmath->sum_on_expanded(G_hidden, m_ratio[1], 2);
    if (m_ratio[0] > 1) G_hidden = kmath->sum_on_expanded(G_hidden, m_ratio[0], 1);
    return G_hidden;
}

Array<float> ExpandLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    if (m_trace) hidden.print("hidden");

    Shape eshape = hidden.shape();

    eshape[1] *= m_ratio[0];
    eshape[2] *= m_ratio[1];

    float* cuda_x = hidden.data_ptr();
    float* cuda_y = cuda.alloc_float_mem(eshape, "expanded");

    int64 esize = eshape.total_size();

    cu_call(ker_expand, esize, (esize, cuda_y, cuda_x, eshape[1], eshape[2], eshape[3], m_ratio[0], m_ratio[1]));

    Array<float> output = cuda.detach(cuda_y, "#netout#");

    if (m_trace) output.print("output");

    return output;
}

Array<float> ExpandLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    if (m_trace) G_hidden.print("G_hidden");

    Shape xshape = G_hidden.shape();

    xshape[1] /= m_ratio[0];
    xshape[2] /= m_ratio[1];

    float* cuda_gy = G_hidden.data_ptr();
    float* cuda_gx = cuda.alloc_float_mem(xshape, "g_expanded");

    int64 xsize = xshape.total_size();

    cu_call(ker_expand_derv, xsize, (xsize, cuda_gx, cuda_gy, xshape[1], xshape[2], xshape[3], m_ratio[0], m_ratio[1]));

    Array<float> G_input = cuda.detach(cuda_gx, "#netout#");

    if (m_trace) G_input.print("G_input");

    return G_input;
}
