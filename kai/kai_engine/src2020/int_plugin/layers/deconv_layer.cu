/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "deconv_layer.cuh"

DeconvLayer::DeconvLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : ConvPoolLayer(options, shape, seq, engine) {
    m_ksize = get_2d_option("ksize");
    m_stride = get_2d_option("stride", 1);
    m_ychn = get_option("chn");

    set_activate(get_option_string("actfunc", "relu"));

    Shape kshape(m_ksize[0], m_ksize[1], m_xchn, m_ychn);

    m_param["k"] = alloc_affine_param(kshape, true);
    //float rand_std = m_engine.lookup_option("rand_std");
    //m_pm["k"] = kmath->random_normal(0, rand_std, kshape);
    //m_pm["b"] = kmath->zeros(m_ychn);

    if (m_engine.lookup_option("show_maps")) {
        m_engine.regist_kernel(m_param["k"]);
    }

    m_output_shape = m_xshape.mul(m_stride).append(m_ychn);
    shape = m_output_shape;
}

DeconvLayer::~DeconvLayer() {
}

Array<float> DeconvLayer::m_forward_farr(Array<float> hidden) {
    hidden = hidden.expand_images(m_stride);

    Array<float> kernel = m_fetch_weight(m_param["k"]);
    Array<float> bias = m_fetch_bias(m_param["k"]);

    m_aux["x"] = hidden;
    hidden = kmath->conv(hidden, kernel, bias);

    m_aux["pre_y"] = hidden;
    hidden = activate(hidden);
    m_aux["post_y"] = hidden;

    if (m_engine.lookup_option("need_maps")) {
        m_engine.add_map(hidden);
    }

    return hidden;
}

Array<float> DeconvLayer::m_backprop_farr(Array<float> G_hidden) {
    G_hidden = activate_derv(G_hidden, m_aux["pre_y"], m_aux["post_y"]);

    Dict param_k = m_param["k"];

    Array<float> kernel = m_fetch_weight(m_param["k"]);
    Array<float> bias = m_fetch_bias(m_param["k"]);
    Array<float> x = m_aux["x"];
    Array<float> G_kernel, G_bias;

    G_hidden = kmath->conv_backprop(G_hidden, x, kernel, G_kernel, G_bias);
    //G_hidden.print("G_hidden after");

    m_update_weight(m_param["k"], G_kernel);
    m_update_bias(m_param["k"], G_bias);
    //update_param(m_pm, "k", G_kernel);
    //update_param(m_pm, "b", G_bias);

    G_hidden = G_hidden.expand_undo_images(m_stride);

    return G_hidden;
}

int64 DeconvLayer::dump_structure(int64 depth) {
    int64 param_cnt;
    string kernel_desc = m_get_affine_param_desc(m_param["k"], &param_cnt);
    logger.Print("%*s%s: %s(%d) : %s => %s : %s => %lld pms",
        depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(), kernel_desc.c_str(), param_cnt);
    return param_cnt;
}

Array<float> DeconvLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    Array<float> kernel = m_fetch_weight(m_param["k"]);
    Array<float> bias = m_fetch_bias(m_param["k"]);

    Shape xshape = hidden.shape();
    Shape kshape = kernel.shape();

    int64 xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
    int64 kh = kshape[0], kw = kshape[1], ychn = kshape[3];
    int64 sh = m_stride[0], sw = m_stride[1];

    float* cuda_h = cuda.copy(hidden, "hidden::Deconv(forward)");
    float* cuda_k = CudaConn::GetCudaMem(kernel, "kernel::Conv(forward)");
    float* cuda_b = CudaConn::GetCudaMem(bias, "bias::Conv(forward)");

    m_aux["x"] = CudaConn::Copy(cuda_h, "x::deconv(forward)");

    //CudaConn::DumpArr(cuda_k, "kernel BP1");
    //CudaConn::DumpArr(cuda_b, "bias BP1");
    //CudaConn::DumpArr(cuda_h, "hidden BP1(Lx)");

    Shape eshape = xshape.replace_end(Shape(xh * sh, xw * sw, xchn));
    int64 esize = eshape.total_size();
    float* cuda_e = cuda.alloc_float_mem(eshape, "cuda_e::deconv(forward)");
    int64 eh = eshape[1], ew = eshape[2];
    cu_call(ker_stride_expand, esize, (esize, cuda_e, cuda_h, xh, xw, eh, ew, xchn, sh, sw));

    //CudaConn::DumpArr(cuda_e, "after expand");

    m_aux["exnd"] = CudaConn::Copy(cuda_e, "exnd::deconv(forward)");

    Shape yshape = eshape.replace_nth(-1, ychn);
    Shape bshape = bias.shape();
    Shape cshape = yshape.append(xchn);

    int64 ysize = yshape.total_size();
    int64 csize = ysize * xchn;

    float* cuda_y = cuda.alloc_float_mem(yshape, "cuda_y::deconv(forward)");
    float* cuda_c = cuda.alloc_float_mem(cshape, "cuda_c::deconv(forward)");

    cu_call(ker_conv_kernel, csize, (csize, cuda_c, cuda_e, cuda_k, eh, ew, kh, kw, xchn, ychn));
    //CudaConn::DumpArr(cuda_c, "conv");
    cu_call(ker_conv_sum, ysize, (ysize, cuda_y, cuda_c, xchn));
    cu_call(ker_conv_add_bias, ysize, (ysize, cuda_y, cuda_b, ychn));

    //CudaConn::DumpArr(cuda_y, "convolution");

    m_aux["conv"] = CudaConn::Copy(cuda_y, "conv::deconv(forward)");

    cu_call(ker_activate, ysize, (ysize, cuda_y, cuda_y, m_nActFunc, m_leaky_alpha));

    m_aux["actv"] = CudaConn::Copy(cuda_y, "actv::deconv(forward)");

    //CudaConn::DumpArr(cuda_y, "activate");

    Array<float> output;

    output = cuda.detach(cuda_y, "cuda_y=>output::deconv(forward)");

    if (m_engine.lookup_option("need_maps")) {
        m_engine.add_map(output);
    }

    return output;
}

Array<float> DeconvLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    float* cuda_gh = CudaConn::GetCudaMem(G_hidden, "G_hidden::deconv(backprop)");

    // backprop for activate step
    float* cuda_ax = cuda.attach(m_aux["conv"], "conv");
    float* cuda_ay = cuda.attach(m_aux["actv"], "actv");

    Shape hshape = G_hidden.shape();

    int64 hsize = hshape.total_size();
    cu_call(ker_activate_derv, hsize, (hsize, cuda_gh, cuda_gh, cuda_ax, cuda_ay, m_nActFunc, m_leaky_alpha));

    Array<float> kernel = m_fetch_weight(m_param["k"]);
    Array<float> bias = m_fetch_bias(m_param["k"]);

    Shape kshape = kernel.shape();
    Shape bshape = bias.shape();

    int64 ksize = kshape.total_size();

    int64 kh = kshape[0], kw = kshape[1], xchn = kshape[2], ychn = kshape[3];
    int64 xh = hshape[-3], xw = hshape[-2];
    int64 mb_size = hshape.total_size() / (xh * xw * ychn);

    Shape eshape = hshape.replace_end(xchn);
    Shape cshape = eshape.append(ychn);
    Shape dshape = kshape.append(mb_size).append(xh);

    int64 csize = cshape.total_size();
    int64 dsize = dshape.total_size();
    int64 esize = eshape.total_size();

    float* cuda_k = CudaConn::GetCudaMem(kernel, "k::deconv(backprop)");

    float* cuda_c = cuda.alloc_float_mem(cshape, "c::deconv(backprop)");
    float* cuda_d = cuda.alloc_float_mem(dshape, "d::deconv(backprop)");
    float* cuda_e = cuda.attach(m_aux["exnd"], "exnd");

    float* cuda_gk = cuda.alloc_float_mem(kshape, "gk::deconv(backprop)");
    float* cuda_gb = cuda.alloc_float_mem(bshape, "gb::deconv(backprop)");
    float* cuda_ge = cuda.alloc_float_mem(eshape, "ge::deconv(backprop)");

    cu_call(ker_conv_derv_x_kernel, csize, (csize, cuda_c, cuda_gh, cuda_k, xh, xw, kh, kw, xchn, ychn));
    cu_call(ker_conv_derv_x_sum, esize, (esize, cuda_ge, cuda_c, ychn));

    int64 sum_size = ksize * mb_size;

    cu_call(ker_conv_derv_kw_x, dsize, (dsize, cuda_d, cuda_gh, cuda_e, mb_size, xh, xw, kh, kw, xchn, ychn));
    cu_call(ker_conv_derv_kw_sum1, sum_size, (sum_size, cuda_d, xw));
    cu_call(ker_conv_derv_kw_sum2, ksize, (ksize, cuda_gk, cuda_d, mb_size, xw));

    int64 size1 = ychn * mb_size * xh;
    int64 size2 = ychn * mb_size;

    cu_call(ker_conv_derv_kb_sum1, size1, (size1, cuda_d, cuda_gh, mb_size, xh, xw, ychn));
    cu_call(ker_conv_derv_kb_sum2, size2, (size2, cuda_d, mb_size, xh, xw, ychn));
    cu_call(ker_conv_derv_kb_sum3, ychn, (ychn, cuda_gb, cuda_d, mb_size, xh, xw, ychn));

    Array<float> G_kernel = cuda.detach(cuda_gk);
    Array<float> G_bias = cuda.detach(cuda_gb);

    // backprop for stride_expand step
    Shape xshape = m_input_shape.add_front(mb_size);
    int64 xsize = xshape.total_size();
    int64 sh = m_stride[0], sw = m_stride[1];

    float* cuda_gx = cuda.alloc_float_mem(xshape, "gx::deconv(backprop)");

    cu_call(ker_stride_expand_derv, xsize, (xsize, cuda_gx, cuda_ge, xh, xw, xh, xw, ychn, sh, sw));

    Array<float> G_input = cuda.detach(cuda_gx, "conv::G_input");

    return G_input;
}
