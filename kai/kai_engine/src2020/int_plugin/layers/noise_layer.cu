/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "noise_layer.cuh"

NoiseLayer::NoiseLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    m_noise_type = (string) get_option("type", "normal");
    m_mean = get_option("mean", 0.0f);
    m_std = get_option("std", 1.0f);
    m_ratio = get_option("ratio", 1.0f);
    
    assert(m_noise_type == "normal");
    assert(m_ratio == 1.0f);
}

NoiseLayer::~NoiseLayer() {
}

Array<float> NoiseLayer::m_forward_farr(Array<float> hidden) {
    if (m_engine.m_is_training) {
        Array<float> noise = kmath->random_normal(m_mean, m_std, m_input_shape);
        hidden += noise;
    }
    return hidden;
}

Array<float> NoiseLayer::m_backprop_farr(Array<float> G_hidden) {
    return G_hidden;
}

Array<float> NoiseLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    if (m_trace) hidden.print("hidden");

    if (m_engine.m_is_training) {
        Shape hshape = hidden.shape();
        int64 hsize = hshape.total_size();

        Array<float> noise = kmath->random_normal(m_mean, m_std, hshape);

        float* cuda_h = cuda.copy_to_buffer(hidden, "x_clone");
        float* cuda_n = cuda.attach(noise, "noise");

        cu_call(ker_add_on, hsize, (hsize, cuda_h, cuda_n));

        hidden = cuda.detach(cuda_h, "noise-out");
    }

    if (m_trace) hidden.print("DropoutLayer::m_forward::hidden out");

    return hidden;
}

Array<float> NoiseLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    return G_hidden;
}
