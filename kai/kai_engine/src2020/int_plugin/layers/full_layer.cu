/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "full_layer.cuh"

FullLayer::FullLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    int64 input_size = m_input_shape.total_size();
    int64 output_cize = get_option("width");

    //m_input_shape = Shape(input_size);    // 형상 점검 위해 실제 들어오는 데이터에 맞춤, 어차피 다차원 텐서는 reshape로 2차원화하기 때문에 이 정보 활용 않음
    m_output_shape = Shape(output_cize);
    
    m_use_bias = get_option("bias", true);

    set_activate((string) get_option("actfunc", "relu"));

    Shape param_shape(input_size, output_cize);

    m_param = alloc_affine_param(param_shape, m_use_bias);

    shape = m_output_shape;
}

FullLayer::~FullLayer() {
}

Array<float> FullLayer::m_forward_farr(Array<float> hidden) {
    int64 nvec = m_input_shape.total_size();

    Shape xshape = hidden.shape();
    Shape yshape = xshape.replace_tail(m_input_shape.size(), m_output_shape);

    Array<float> hidden_2d = hidden.reshape(Shape(-1, nvec));

    Array<float> affine = forward_affine(m_param, hidden_2d);
    Array<float> output_2d = activate(affine);

    Array<float> output = output_2d.reshape(yshape);

    m_aux["affine"] = affine;
    m_aux["y"] = output_2d;
    m_aux["x"] = hidden_2d;

    return output;
}

Array<float> FullLayer::m_backprop_farr(Array<float> G_hidden) {
    int64 nvec = m_output_shape.total_size();

    Shape yshape = G_hidden.shape();
    Shape xshape = yshape.replace_tail(m_output_shape.size(), m_input_shape);

    Array<float> G_hidden_2d = G_hidden.reshape(Shape(-1, nvec));

    Array<float> G_affine = activate_derv(G_hidden_2d, m_aux["affine"], m_aux["y"]);
    Array<float> G_input = backprop_affine(m_param, m_aux["x"], G_affine);

    G_input = G_input.reshape(xshape);

    return G_input;
}

int64 FullLayer::dump_structure(int64 depth) {
    int64 param_cnt;
    string pm_desc = m_get_affine_param_desc(m_param, &param_cnt);
    
    logger.Print("%*s%s: %s(%d) : %s => %s : %s ==> %lld pms",
        depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(),
        pm_desc.c_str(), param_cnt);

    return param_cnt;
}

Array<float> FullLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    int64 nvec = m_input_shape.total_size(), ncol = m_output_shape.total_size();
    int64 nrow = hidden.total_size() / nvec;

    Shape ashape = hidden.shape().replace_tail(m_input_shape.size(), m_output_shape);

    int64 asize = ashape.total_size();

    float* cuda_h = cuda.copy_to_buffer(hidden);

    float* cuda_a = cuda.alloc_float_mem(ashape);
    float* cuda_y = cuda.alloc_float_mem(ashape);

    forward_affine_cuda(m_param, m_use_bias, cuda_a, cuda_h, nrow, nvec, ncol);

    cu_call(ker_activate, asize, (asize, cuda_y, cuda_a, m_nActFunc, m_leaky_alpha));

    Array<float> output = cuda.detach(cuda_y, "#netout#");

    m_aux["input"] = hidden;
    m_aux["netout"] = output;
    m_aux["affine"] = cuda.detach(cuda_a, "affine");

    return output;
    /*
    CudaConn cuda("forward", this);

    int64 nvec = m_input_shape.total_size(), ncol = m_output_shape.total_size();
    int64 nrow = hidden.total_size() / ncol;

    Shape ashape = hidden.shape().replace_end(ncol);

    int64 asize = ashape.total_size();

    float* cuda_h = cuda.copy_to_buffer(hidden);

    float* cuda_a = cuda.alloc_float_mem(ashape);
    float* cuda_y = cuda.alloc_float_mem(ashape);

    forward_affine_cuda(m_param, m_use_bias, cuda_a, cuda_h, nrow, nvec, ncol);

    cu_call(ker_activate, asize, (asize, cuda_y, cuda_a, m_nActFunc, m_leaky_alpha));

    Array<float> output = cuda.detach(cuda_y, "#netout#");

    m_aux["input"] = hidden;
    m_aux["netout"] = output;
    m_aux["affine"] = cuda.detach(cuda_a, "affine");

    return output;
    */
}

Array<float> FullLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    float* cuda_x = CudaConn::GetCudaFloatMem(m_aux["input"], "input");
    float* cuda_y = CudaConn::GetCudaFloatMem(m_aux["netout"], "netout");
    float* cuda_a = cuda.attach(m_aux["affine"], "affine");

    // CudaConn::DumpArr(cuda_x, "backprop x", Shape(), true);
    // CudaConn::DumpArr(cuda_y, "backprop activated", Shape(), true);
    // CudaConn::DumpArr(cuda_a, "backprop affine", Shape(), true);

    int64 nvec = m_input_shape.total_size(), ncol = m_output_shape.total_size();
    int64 nrow = G_hidden.total_size() / ncol;

    Shape hshape = G_hidden.shape();
    //Shape xshape = m_input_shape.add_front(nrow);
    Shape xshape = hshape.replace_tail(1, m_input_shape);

    float* cuda_gh = cuda.copy(G_hidden, "G_hidden::Full(backprop)");
    float* cuda_ga = cuda.alloc_float_mem(hshape, "cuda_ga");
    float* cuda_gx = cuda.alloc_float_mem(xshape, "cuda_gx");

    // CudaConn::DumpArr(cuda_gh, "backprop hidden-grad", Shape(), true);

    //float* cuda_w = CudaConn::GetCudaMem(w, "w::Full(backprop)");
    //float* cuda_gw = cuda.alloc_float_mem(wshape, "gw::Full(backprop)");

    int64 hsize = G_hidden.total_size();
    int64 xsize = xshape.total_size();
    //int64 wsize = wshape.total_size();

    cu_call(ker_activate_derv, hsize, (hsize, cuda_ga, cuda_gh, cuda_a, cuda_y, m_nActFunc, m_leaky_alpha));
    // CudaConn::DumpArr(cuda_ga, "backprop affine-grad", Shape(), true);

    backprop_affine_cuda(m_param, m_use_bias, cuda_gx, cuda_ga, cuda_x, nrow, nvec, ncol);
    // CudaConn::DumpArr(cuda_gx, "backprop x-grad", Shape(), true);
    /*
    cu_call(ker_matmul_derv_x, xsize, (xsize, cuda_gx, cuda_ga, cuda_w, nrow, nvec, ncol));
    cu_call(ker_matmul_derv_w, wsize, (wsize, cuda_gw, cuda_ga, cuda_x, nrow, nvec, ncol));

    cuda.optimize_weight(m_pm, m_engine, cuda_gw);

    if (m_use_bias) {
        Array<float> b = m_pm["b"];
        Shape bshape = b.shape();
        float* cuda_gb = cuda.alloc_float_mem(bshape, "gb::Full(backprop)");
        int64 bsize = bshape.total_size();

        cu_call(ker_add_bias_derv, bsize, (bsize, cuda_gb, cuda_ga, nrow, nvec, ncol));

        if (m_trace) cuda.DumpArr(cuda_gb, "gb");

        cuda.optimize_bias(m_pm, m_engine, cuda_gb);
    }
    */

    Array<float> G_input = cuda.detach(cuda_gx, "full::G_input");

    return G_input;
}
