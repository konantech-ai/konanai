/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "parallel_layer.cuh"

ParallelLayer::ParallelLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine) : ComplexLayer(options, layer_subnet, shape, seq, engine) {
    Shape oshape;
    List pms;
    int64 ochn = 0;

    for (List::iterator it = layer_subnet.begin(); it != layer_subnet.end(); it++) {
        Shape bshape = m_input_shape;
        Layer* pPlayer = Layer::CreateLayer(*it, bshape, seq, engine);
        m_layers.push_back(pPlayer);
        pms.push_back(pPlayer->m_param);
        if (oshape.size() == 0) oshape = bshape;
        int64 bchn = oshape[-1] = bshape[-1];
        ochn += bchn;
        m_chns.push_back((int) bchn);
        assert(oshape == bshape);
    }

    oshape[-1] = ochn;

    m_param["pms"] = pms;

    shape = m_output_shape = oshape;
}

ParallelLayer::~ParallelLayer() {
}

Array<float> ParallelLayer::m_forward_farr(Array<float> hidden) {
    vector<Array<float>> bhiddens;

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        bhiddens.push_back(pLayer->forward_subnet(hidden));
    }
    hidden = kmath->hstack(bhiddens);
    return hidden;
}

Array<float> ParallelLayer::m_backprop_farr(Array<float> G_hidden) {
    vector<Array<float>> G_bhiddens = kmath->hsplit(G_hidden, m_chns);

    int64 mb_size = G_hidden.axis_size(0);
    G_hidden = kmath->zeros(m_input_shape.add_front(mb_size));

    int n = 0;
    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++, n++) {
        Layer* pLayer = *it;
        Array<float> G_bhidden = pLayer->backprop_subnet(G_bhiddens[n]);
        G_hidden += G_bhidden;
    }

    return G_hidden;
}

Array<float> ParallelLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    int64 mb_size = hidden.axis_size(0);
    int64 ychn = m_output_shape[-1];
    int64 chn_from = 0;

    Shape yshape = m_output_shape.add_front(mb_size);

    float* cuda_y = cuda.alloc_float_mem(yshape, "out:parallel(forward)");

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        Array<float> bhidden = pLayer->forward_subnet(hidden);

        float* cuda_b = bhidden.data_ptr();

        int64 bsize = bhidden.total_size();
        int64 bchn = bhidden.axis_size(-1);

        cu_call(ker_get_branch, bsize, (bsize, cuda_y, cuda_b, ychn, bchn, chn_from));

        chn_from += bchn;
    }

    Array<float> output = cuda.detach(cuda_y, "#netout#");

    return output;
}

Array<float> ParallelLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    int64 mb_size = G_hidden.axis_size(0);
    int64 ychn = m_output_shape[-1];
    int64 chn_from = 0;

    Shape hshape = G_hidden.shape();
    Shape xshape = m_input_shape.add_front(mb_size);
    int64 xsize = xshape.total_size();

    float* cuda_gh = cuda.copy(G_hidden, "G_hidden::parallel(backprop)");
    float* cuda_gx = cuda.alloc_float_mem(xshape, "out:parallel(backprop)");

    int n = 0;
    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++, n++) {
        int64 bchn = m_chns[n];
        Shape bshape = hshape.replace_nth(-1, bchn);
        int64 bsize = bshape.total_size();

        float* cuda_gby = cuda.alloc_float_mem(bshape, "bh:parallel(backprop)");

        cu_call(ker_set_branch, bsize, (bsize, cuda_gby, cuda_gh, ychn, bchn, chn_from));

        Array<float> G_bOutput = cuda.detach(cuda_gby, "G_bOutput:parallel(backprop)");

        Layer* pLayer = *it;
        Array<float> G_bInput = pLayer->backprop_subnet(G_bOutput);

        float* cuda_gbx = cuda.attach(G_bInput, "G_bInput:parallel(backprop)");

        cu_call(ker_add_on, xsize, (xsize, cuda_gx, cuda_gbx));
        chn_from += bchn;
    }

    Array<float> G_input = cuda.detach(cuda_gx, "parallel::G_input");

    return G_input;
}
