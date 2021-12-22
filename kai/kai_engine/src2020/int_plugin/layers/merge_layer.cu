/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "merge_layer.cuh"

MergeLayer::MergeLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    m_method = (string) get_option("method");

    if (m_method == "mean") {
        assert(shape.size() == 2);
        shape = m_output_shape = shape.remove_front();
    }
    else {
        throw KaiException(KERR_ASSERT);
    }
}

MergeLayer::~MergeLayer() {
}

Array<float> MergeLayer::m_forward_farr(Array<float> hidden) {
    if (m_method == "mean") {
        m_aux["vec_cnt"] = hidden.axis_size(1);
        assert(hidden.dim() == 3);
        Shape shape = hidden.shape();
        //hidden.print("hidden");
        hidden = hidden.transpose(Idx(0, 2, 1));
        hidden = hidden.reshape(Shape(-1, shape[1]));
        hidden = hidden.avg(0);
        hidden = hidden.reshape(Shape(shape[0], shape[2]));
        //hidden.print("hidden");
    }
    else {
        throw KaiException(KERR_ASSERT);
    }
    return hidden;
}

Array<float> MergeLayer::m_backprop_farr(Array<float> G_hidden) {
    if (m_method == "mean") {
        int64 vec_cnt = m_aux["vec_cnt"];
        G_hidden = G_hidden.tile((int)vec_cnt);
        G_hidden = G_hidden / (float)vec_cnt;
        G_hidden = G_hidden.transpose(Idx(0, 2, 1));
        return G_hidden;
    }
    else {
        throw KaiException(KERR_ASSERT);
    }
    return G_hidden;
}

Array<float> MergeLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    if (m_method == "mean") {
        assert(hidden.dim() == 3);

        int64 mb_size = hidden.axis_size(0), vec_cnt = hidden.axis_size(1), vec_size = hidden.axis_size(2);

        m_aux["vec_cnt"] = hidden.axis_size(1);

        Shape mshape(mb_size, vec_size);

        float* cuda_h = CudaConn::GetCudaMem(hidden, "x");
        float* cuda_m = cuda.alloc_float_mem(mshape, "merged");

        int64 msize = mshape.total_size();

        cu_call(ker_merge_avg, msize, (msize, cuda_m, cuda_h, vec_cnt, vec_size));

        Array<float> output = cuda.detach(cuda_m, "#netout#");

        return output;
    }
    else {
        throw KaiException(KERR_ASSERT);
    }
    return hidden;
}

Array<float> MergeLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    if (m_method == "mean") {
        int64 mb_size = G_hidden.axis_size(0), vec_size = G_hidden.axis_size(1);
        int64 vec_cnt = m_aux["vec_cnt"];

        Shape xshape(mb_size, vec_cnt, vec_size);

        float* cuda_gh = CudaConn::GetCudaMem(G_hidden, "gy");
        float* cuda_gx = cuda.alloc_float_mem(xshape, "gx");

        int64 xsize = xshape.total_size();

        cu_call(ker_merge_avg_derv, xsize, (xsize, cuda_gx, cuda_gh, vec_cnt, vec_size));

        Array<float> G_input = cuda.detach(cuda_gx, "G_input");

        return G_input;
    }
    else {
        throw KaiException(KERR_ASSERT);
    }
    return G_hidden;
}
