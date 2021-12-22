/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "batch_norm_layer.cuh"

BatchNormalLayer::BatchNormalLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    m_rescale = get_option("rescale", true);
    m_epsilon = get_option("epsilon", 0.001f);
    m_momentum = get_option("momentum", 0.99f);

    m_bn_shape = Shape(m_input_shape[-1]);

    m_param["mavg"] = kmath->zeros(m_bn_shape);
    m_param["mvar"] = kmath->ones(m_bn_shape);
    
    if (m_rescale) {
        m_param["scale"] = alloc_affine_param(m_bn_shape, false, param_init::ones); 
        m_param["shift"] = alloc_affine_param(m_bn_shape, false, param_init::zeros);
    }
}

BatchNormalLayer::~BatchNormalLayer() {
}

Array<float> BatchNormalLayer::m_forward_farr(Array<float> hidden) {
    Array<float> old_avg = m_param["mavg"];
    Array<float> old_var = m_param["mvar"];

    Array<float> new_avg = old_avg;
    Array<float> new_var = old_var;

    if (m_engine.m_is_training) {
        Array<float> x_flat = hidden.reshape(Shape(-1, m_bn_shape[0]));

        new_var = x_flat.var(-1, &new_avg);

        m_param["mavg"] = old_avg * m_momentum + new_avg * (1 - m_momentum);
        m_param["mvar"] = old_var * m_momentum + new_var * (1 - m_momentum);
    }

    m_std = kmath->sqrt(new_var + m_epsilon);
    hidden = (hidden - new_avg) / m_std;

    m_aux["norm_x"] = hidden;

    if (m_rescale) {
        Array<float> scale = m_fetch_weight(m_param["scale"]);
        Array<float> shift = m_fetch_weight(m_param["shift"]);

        hidden = hidden * scale + shift;
    }

    return hidden;
}

Array<float> BatchNormalLayer::m_backprop_farr(Array<float> G_hidden) {
    if (m_rescale) {
        Array<float> hidden = m_aux["norm_x"];
        
        Array<float> G_scale = kmath->sum(G_hidden * hidden, -1);
        Array<float> G_shift = G_hidden.sum(-1);
        Array<float> scale = m_fetch_weight(m_param["scale"]);

        G_hidden = G_hidden * scale;

        m_update_weight(m_param["scale"], G_scale);
        m_update_weight(m_param["shift"], G_shift);
    }

    G_hidden = G_hidden / m_std;

    return G_hidden;
}

int64 BatchNormalLayer::dump_structure(int64 depth) {
    if (m_rescale) {
        int64 param_cnt = (int) (m_bn_shape[0] + m_bn_shape[0]);
        logger.Print("%*s%s: %s(%d) : %s => %s : %lld+%lld => %lld pms",
            depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(),
            m_bn_shape[0], m_bn_shape[0], param_cnt);
        return param_cnt;
    }
    else {
        return Layer::dump_structure(depth);
    }
}

Array<float> BatchNormalLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    Shape hshape = hidden.shape();

    float* cuda_h = cuda.copy_to_buffer(hidden, "x::BN(forward)");

    m_forward_bn_core(cuda, cuda_h, hshape);

    Array<float> output = cuda.detach(cuda_h, "bn:output");

    return output;
}

void BatchNormalLayer::m_forward_bn_core(CudaConn& cuda, float* cuda_h, Shape hshape) {
    float* cuda_mavg = CudaConn::GetCudaFloatMem(m_param["mavg"], "mavg::BN(forward)");
    float* cuda_mvar = CudaConn::GetCudaFloatMem(m_param["mvar"], "mvar::BN(forward)");

    Shape bshape(m_input_shape[-1]);

    int64 hsize = hshape.total_size();
    int64 bsize = m_input_shape[-1];

    if (m_engine.m_is_training) {
        float* cuda_avg = cuda.alloc_float_mem(bshape, "avg");
        float* cuda_var = cuda.alloc_float_mem(bshape, "var");

        cu_call(ker_bn_collect, bsize, (bsize, cuda_avg, cuda_var, cuda_mavg, cuda_mvar, cuda_h, hsize, m_momentum));
        cu_call(ker_bn_normalize, hsize, (hsize, cuda_h, cuda_avg, cuda_var, bsize, m_epsilon));

        m_aux["var"] = cuda.detach(cuda_var, "var");;
    }
    else {
        cu_call(ker_bn_normalize, hsize, (hsize, cuda_h, cuda_mavg, cuda_mvar, bsize, m_epsilon));
    }

    m_aux["norm_x"] = CudaConn::Copy(cuda_h, "norm_x:BN(forward)");

    if (m_rescale) {
        float* cuda_scale = m_fetch_weight_ptr(m_param["scale"]);
        float* cuda_shift = m_fetch_weight_ptr(m_param["shift"]);

        cu_call(ker_bn_rescale, hsize, (hsize, cuda_h, cuda_scale, cuda_shift, bsize));
    }
}

Array<float> BatchNormalLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    Shape hshape = G_hidden.shape();

    float* cuda_gh = cuda.copy_to_buffer(G_hidden, "gy:BN(backprop)");

    m_backprop_bn_core(cuda, cuda_gh, hshape);

    G_hidden = cuda.detach(cuda_gh, "G_input:BN(backprop)");

    return G_hidden;
}

void BatchNormalLayer::m_backprop_bn_core(CudaConn& cuda, float* cuda_gh, Shape hshape) {
    int64 hsize = hshape.total_size();
    int64 bsize = m_input_shape[-1];

    Shape bshape(bsize);

    if (m_rescale) {
        float* cuda_x = cuda.attach(m_aux["norm_x"], "norm_x:BN(backprop)");

        float* cuda_scale = m_fetch_weight_ptr(m_param["scale"]);

        float* cuda_gscale = cuda.alloc_float_mem(bshape, "gscale::BN(backprop)");
        float* cuda_gshift = cuda.alloc_float_mem(bshape, "shift::BN(backprop)");

        cu_call(ker_bn_rescale_derv_pm, bsize, (bsize, cuda_gscale, cuda_gshift, cuda_gh, cuda_x, hsize));
        cu_call(ker_bn_rescale_derv_x, hsize, (hsize, cuda_gh, cuda_scale, bsize));

        Array<float> G_scale = cuda.detach(cuda_gscale);
        Array<float> G_shift = cuda.detach(cuda_gshift);

        m_update_weight(m_param["scale"], G_scale);
        m_update_weight(m_param["shift"], G_shift);
    }

    Array<float> var = m_aux["var"];
    float* cuda_var = var.data_ptr();

    cu_call(ker_bn_norm_derv, hsize, (hsize, cuda_gh, cuda_var, bsize, m_epsilon));
}
