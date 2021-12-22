/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "dropout_layer.cuh"

DropoutLayer::DropoutLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    m_keep_prob = get_option("keep_prob", 1.0f);
    if (m_keep_prob <= 0 || m_keep_prob >= 1) {
        m_keep_prob = 1.0f - (float) get_option("drop_ratio", 0.0f);
    }
    assert(m_keep_prob > 0 && m_keep_prob < 1);
}

DropoutLayer::~DropoutLayer() {
}

Array<float> DropoutLayer::create_mask(Shape shape) {
    m_drop_mask = kmath->random_bernoulli(shape, m_keep_prob);
    return m_drop_mask;
}

Array<float> DropoutLayer::m_forward_farr(Array<float> hidden) {
    if (m_trace) hidden.print("DropoutLayer::m_forward::hidden");
    if (m_trace) logger.Print("DropoutLayer::m_forward::m_keep_prob = %f", m_keep_prob);
    if (m_engine.m_is_training) {
        m_drop_mask = create_mask(hidden.shape());
        hidden = hidden * m_drop_mask / m_keep_prob;
        if (m_trace) hidden.print("DropoutLayer::m_forward::hidden after mask");
    }
    if (m_trace) hidden.print("DropoutLayer::m_forward::hidden out");
    
    return hidden;
}

Array<float> DropoutLayer::m_backprop_farr(Array<float> G_hidden) {
    if (m_engine.m_is_training) {
        G_hidden = G_hidden * m_drop_mask / m_keep_prob;
    }
    return G_hidden;
}

Array<float> DropoutLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    if (m_trace) hidden.print("hidden");
    //hidden.print("hidden");

    if (m_engine.m_is_training) {
        Shape hshape = hidden.shape();
        int64 hsize = hshape.total_size();

        create_mask(hshape);

        float* cuda_h = cuda.copy_to_buffer(hidden, "x_clone");
        float* cuda_m = m_drop_mask.data_ptr();

        cu_call(ker_dropout, hsize, (hsize, cuda_h, cuda_h, cuda_m, m_keep_prob));

        hidden = cuda.detach(cuda_h, "#netout#");
    }

    if (m_trace) hidden.print("DropoutLayer::m_forward::hidden out");

    //hidden.print("DropoutLayer::m_forward::hidden out");
    //throw KaiException(KERR_ASSERT);

    return hidden;
}

Array<float> DropoutLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    if (m_trace) G_hidden.print("Dropout:backprop:G_hidden in");

    if (m_engine.m_is_training) {
        Array<float> G_input = G_hidden;
        return G_input;
    }
    else {
        Shape hshape = G_hidden.shape();

        int64 hsize = hshape.total_size();

        float* cuda_gh = cuda.copy_to_buffer(G_hidden, "Gy_clone");
        float* cuda_m = cuda.attach(m_drop_mask, "mask");

        if (m_trace) cuda.DumpArr(cuda_m, "mask");

        cu_call(ker_dropout_derv, hsize, (hsize, cuda_gh, cuda_gh, cuda_m, m_keep_prob));

        Array<float> G_input = cuda.detach(cuda_gh, "G_input");

        if (m_trace) G_input.print("Dropout:backprop:G_input out");

        return G_input;
    }
}
