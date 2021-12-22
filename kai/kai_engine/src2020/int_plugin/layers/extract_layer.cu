/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "extract_layer.cuh"

ExtractLayer::ExtractLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    List filters = get_option("filter");
    assert(filters.size() == 1);
    m_filter = filters[0];
    int axis = m_filter["axis"];
    assert(axis >= 1 && axis <= shape.size());
    shape = m_output_shape = shape.remove_nth(axis-1);
}

ExtractLayer::~ExtractLayer() {
}

Array<float> ExtractLayer::m_forward_farr(Array<float> hidden) {
    if (m_trace) hidden.print("ExtractLayer::m_forward::hidden in");

    int axis = m_filter["axis"];
    int index = m_filter["index"];
    if (m_trace) logger.Print("axis = %d, index=%d", axis, index);

    m_aux["shape"] = hidden.shape();
    if (m_trace) hidden.print("ExtractLayer::m_forward::hidden in");

    hidden = hidden.filter(axis, index);
    if (m_trace) hidden.print("ExtractLayer::m_forward::hidden after");

    return hidden;
}

Array<float> ExtractLayer::m_backprop_farr(Array<float> G_hidden) {
    if (m_trace) G_hidden.print("ExtractLayer::m_backprop::G_hidden in");

    int axis = m_filter["axis"];
    int index = m_filter["index"];

    Shape shape = m_aux["shape"];

    Array<float> G_full_hidden = kmath->zeros(shape);

    G_full_hidden = G_full_hidden.unfilter(G_hidden, axis, index);

    if (m_trace) G_full_hidden.print("ExtractLayer::m_backprop::G_full_hidden out");
    
    return G_full_hidden;
}

Array<float> ExtractLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    m_aux["shape"] = hidden.shape();

    int axis = m_filter["axis"];
    int index = m_filter["index"];

    Shape hshape = hidden.shape();
    Shape eshape = hshape.remove_nth(axis);

    float* cuda_h = hidden.data_ptr();
    float* cuda_e = cuda.alloc_float_mem(eshape, "extract");

    int64 esize = eshape.total_size();
    int64 nprod = 1;

    for (int n = axis + 1; n < hshape.size(); n++) nprod *= hshape[n];

    if (m_trace) logger.Print("axis = %d, hshape[axis] = %lld, index = %d, nprod = %lld", axis, hshape[axis], index, nprod);

    cu_call(ker_extract, esize, (esize, cuda_e, cuda_h, hshape[axis], index, nprod));

    Array<float> output = cuda.detach(cuda_e, "#netout#");

    return output;
}

Array<float> ExtractLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    if (m_trace) G_hidden.print("ExtractLayer::m_backprop::G_hidden in");

    int axis = m_filter["axis"];
    int index = m_filter["index"];

    Shape eshape = G_hidden.shape();
    Shape xshape = m_aux["shape"];

    float* cuda_gh = G_hidden.data_ptr();
    float* cuda_gx = cuda.alloc_float_mem(xshape, "extract");

    int64 xsize = xshape.total_size();
    int64 nprod = 1;

    for (int n = axis + 1; n < xshape.size(); n++) nprod *= xshape[n];

    if (m_trace) logger.Print("axis = %d, hshape[axis] = %lld, index = %d, nprod = %lld", axis, xshape[axis], index, nprod);

    cu_call(ker_unextract, xsize, (xsize, cuda_gx, cuda_gh, xshape[axis], index, nprod));

    Array<float> output = cuda.detach(cuda_gx, "#netout#");

    if (m_trace) output.print("ExtractLayer::m_backprop::G_full_hidden out");

    return output;
}
