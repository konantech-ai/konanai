/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "conv_layer.cuh"
#include "batch_norm_layer.cuh"

ConvPoolLayer::ConvPoolLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    assert(m_input_shape.size() == 3);

    m_xshape = Shape(m_input_shape[0], m_input_shape[1]);
    m_padding = get_option_string("padding", "SAME");

    m_xchn = (int) m_input_shape[2];
}

ConvPoolLayer::~ConvPoolLayer() {
}

Array<float> ConvPoolLayer::m_stride_filter(Array<float> hidden) {
    Shape nshape = m_xshape;
    Shape hshape = hidden.shape();

    m_aux["hshape"] = hshape;

    int64 xh = hshape[-3], xw = hshape[-2], xchn = hshape[-1];
    int64 mb_size = hshape.total_size() / (xh * xw * xchn);

    Array<float> h_4d = hidden.reshape(Shape(mb_size, xh, xw, xchn));

    if (m_padding == "VALID") {
        Shape b = (m_ksize - 1) / 2;
        nshape = m_xshape.sub(m_ksize) + 1;
        h_4d = h_4d[Axis(_all_, Ax(b[0], b[0] + nshape[0]), Ax(b[1], b[1] + nshape[1]), _all_)];
    }

    if (m_stride != Shape(1, 1)) {
        Shape begin = (m_stride - 1) / 2;
        Shape cnt = nshape.div(m_stride);
        Shape end = begin.add(cnt.mul(m_stride));
        h_4d = h_4d[Axis(_all_, Ax(begin[0], end[0], m_stride[0]), Ax(begin[1], end[1], m_stride[1]), _all_)];
    }

    m_aux["nshape"] = nshape;

    hshape[-3] = h_4d.shape()[1];
    hshape[-2] = h_4d.shape()[2];

    hidden = h_4d.reshape(hshape);

    return hidden;
}

Array<float> ConvPoolLayer::m_stride_filter_derv(Array<float> G_hidden) {
    int64 mb_size = G_hidden.axis_size(0);
    Shape nshape = m_aux["nshape"];

    Array<float> G_h_tmp;

    if (m_stride != Shape(1, 1)) {
        Shape begin = (m_stride - 1) / 2;
        Shape cnt = nshape.div(m_stride);
        Shape end = begin.add(cnt.mul(m_stride));

        G_h_tmp = kmath->zeros(Shape(mb_size, nshape[0], nshape[1], m_ychn));
        G_h_tmp[Axis(_all_, Ax(begin[0], end[0], m_stride[0]), Ax(begin[1], end[1], m_stride[1]), _all_)] = G_hidden;
        G_hidden = G_h_tmp;
    }

    if (m_padding == "VALID") {
        Shape b = (m_ksize - 1) / 2;
        Shape nshape = m_xshape.sub(m_ksize) + 1;
        Shape hshape = m_aux["hshape"];

        G_h_tmp = kmath->zeros(Shape(mb_size, hshape[0], hshape[1], m_ychn));
        G_h_tmp[Axis(_all_, Ax(b[0], b[0] + nshape[0]), Ax(b[1], b[1] + nshape[1]), _all_)] = G_hidden;
        G_hidden = G_h_tmp;
    }

    return G_hidden;
}

Array<float> ConvPoolLayer::m_get_ext_regions(Array<float> x, float fill) {
    Shape xss = x.shape();
    int64 mb_size = xss[0], xh = xss[1], xw = xss[2], xchn = xss[3];
    int64 kh = m_ksize[0], kw = m_ksize[1];
    Shape xs(xh, xw);

    Shape e = xs.add(m_ksize) - 1;
    Shape b = (m_ksize - 1) / 2;

    Array<float> x_ext = kmath->zeros(Shape(mb_size, e[0], e[1], xchn));
    if (fill != 0) x_ext += fill;

    x_ext[Axis(_all_, Ax(b[0], b[0] + xs[0]), Ax(b[1], b[1] + xs[1]), _all_)] = x;

    m_aux["x_shape"] = xs;
    m_aux["ext_shape"] = x_ext.shape();

    Array<float> regs = kmath->zeros(Shape(xs[0], xs[1], mb_size * kh * kw * xchn));

    for (int r = 0; r < xs[0]; r++) {
        for (int c = 0; c < xs[1]; c++) {
            Array<float> part;
            part = x_ext[Axis(_all_, Ax(r, r + kh), Ax(c, c + kw), _all_)];
            regs[Axis(r, c, _all_)] = part.reshape(Shape(1, 1, -1));
        }
    }

    return regs.reshape(Shape(xh, xw, mb_size, kh, kw, xchn));
}

Array<float> ConvPoolLayer::m_undo_ext_regions(Array<float> G_regs) {
    Shape xs = m_aux["x_shape"];
    Shape es = m_aux["ext_shape"];

    G_regs = G_regs.reshape(Shape(xs[0], xs[1], -1));

    Array<float> G_ext = kmath->zeros(es);

    for (int r = 0; r < xs[0]; r++) {
        for (int c = 0; c < xs[1]; c++) {
            Array<float> part;
            part = G_regs[Axis(r, c, _all_)];
            part = part.reshape(Shape(es[0], m_ksize[0], m_ksize[1], es[3]));
            G_ext[Axis(_all_, Ax(r, r + m_ksize[0]), Ax(c, c + m_ksize[1]), _all_)] += part;
        }
    }

    Shape b = (m_ksize - 1) / 2;

    G_regs = G_ext[Axis(_all_, Ax(b[0], b[0] + xs[0]), Ax(b[1], b[1] + xs[1]), _all_)];

    return G_regs;
}

ConvLayer::ConvLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : ConvPoolLayer(options, shape, seq, engine), m_pBNLayers(NULL) {
    m_ksize = get_2d_option("ksize");
    m_stride = get_2d_option("stride", 1);
    m_ychn = get_option("chn");

    set_activate(get_option_string("actfunc", "relu"));

    Shape kshape(m_ksize[0], m_ksize[1], m_xchn, m_ychn);

    m_param["k"] = alloc_affine_param(kshape, true);

    if (m_engine.lookup_option("show_maps")) {
        m_engine.regist_kernel(m_param["k"]);
    }

    Shape bn_input_shape = m_input_shape;
    Shape output_shape = m_xshape.append(m_ychn);

    m_actions = (string)get_option("actions", "LA");

    for (int n = 0; n < (int) m_actions.length(); n++) {
        if (m_actions[n] == 'L') {
            bn_input_shape = output_shape;
        }
        else if (m_actions[n] == 'B') {
            Dict bn_options;
            bn_options["rescale"] = false;
            m_pBNLayers = new BatchNormalLayer(bn_options, bn_input_shape, seq, engine);
            m_pBNLayers->m_layer_name = "(batchnorm)";
            m_param["batch_norm"] = m_pBNLayers->m_param;
        }
    }

    if (m_padding == "VALID") {
        m_xshape = m_xshape.sub(m_ksize) + 1;
    }

    m_output_shape = m_xshape.div(m_stride).append(m_ychn);
    shape = m_output_shape;
}

ConvLayer::~ConvLayer() {
    delete m_pBNLayers;
}

Array<float> ConvLayer::m_forward_farr(Array<float> hidden) {
    for (int n = 0; n < (int) m_actions.length(); n++) {
        if (m_actions[n] == 'L') {
            Array<float> kernel = m_fetch_weight(m_param["k"]);
            Array<float> bias = m_fetch_bias(m_param["k"]);
            m_aux["x"] = hidden;
            hidden = kmath->conv(hidden, kernel, bias);
        }
        else if (m_actions[n] == 'A') {
            m_aux["pre_y"] = hidden;
            hidden = activate(hidden);
            m_aux["post_y"] = hidden;
        }
        else if (m_actions[n] == 'B') {
            hidden = m_pBNLayers->forward_subnet(hidden);
        }
    }

    hidden = m_stride_filter(hidden);

    if (m_engine.lookup_option("need_maps")) {
        m_engine.add_map(hidden);
    }

    return hidden;
}

Array<float> ConvLayer::m_backprop_farr(Array<float> G_hidden) {
    G_hidden = m_stride_filter_derv(G_hidden);

    for (int n = (int) m_actions.length() - 1; n >= 0; n--) {
        if (m_actions[n] == 'L') {
            Array<float> kernel = m_fetch_weight(m_param["k"]);
            Array<float> bias = m_fetch_bias(m_param["k"]);
            Array<float> x = m_aux["x"];
            Array<float> G_kernel, G_bias;

            //G_hidden.print("G_hidden before");
            //x.print("x");
            G_hidden = kmath->conv_backprop(G_hidden, x, kernel, G_kernel, G_bias);
            //G_hidden.print("G_hidden after");

            m_update_weight(m_param["k"], G_kernel);
            m_update_bias(m_param["k"], G_bias);
            //update_param(m_pm, "k", G_kernel);
            //update_param(m_pm, "b", G_bias);

            /*
            Array<float> kernel = m_pm["k"], bias = m_pm["b"];
            Array<float> G_hidden_old, G_hidden_new;
            Array<float> G_kernel, G_bias;
            Array<float> G_kernel_new, G_bias_new;

            if (1) {
                FuncTimer func_timer("conv_backprop_old_style");

                Array<float> x_flat = m_aux["x_flat"];
                Array<float> k_flat = m_aux["k_flat"];

                Array<float> G_conv_flat = G_hidden.reshape(Shape(-1, m_ychn));
                Array<float> g_conv_k_flat = x_flat.transpose();
                Array<float> g_conv_x_flat = k_flat.transpose();
                Array<float> G_k_flat = kmath->matmul(g_conv_k_flat, G_conv_flat);
                Array<float> G_x_flat = kmath->matmul(G_conv_flat, g_conv_x_flat);

                G_bias = kmath->sum(G_conv_flat, -1);
                G_kernel = G_k_flat.reshape(Shape(m_ksize[0], m_ksize[1], m_xchn, m_ychn));
                G_hidden_old = m_undo_ext_regions_for_conv(G_x_flat);
            }

            if (1) {
                FuncTimer func_timer("conv_backprop_new_style");
                Array<float> x = m_aux["x"];
                G_hidden_new = kmath->conv_backprop(G_hidden, x, kernel, G_kernel_new, G_bias_new);

                //G_hidden_new.print("G_hidden_new");
                //G_kernel_new.print("G_kernel_new");
                //G_bias_new.print("G_bias_new");
            }

            float check1 = kmath->sum(kmath->square(G_hidden_old - G_hidden_new));
            float check2 = kmath->sum(kmath->square(G_kernel - G_kernel_new));
            float check3 = kmath->sum(kmath->square(G_bias - G_bias_new));

            logger.Print("conv backprop check1 = %f, check2 = %f, check3 = %f", check1, check2, check3);

            update_param(m_pm, "k", G_kernel);
            update_param(m_pm, "b", G_bias);

            G_hidden = G_hidden_old;
            */
        }
        else if (m_actions[n] == 'A') {
            G_hidden = activate_derv(G_hidden, m_aux["pre_y"], m_aux["post_y"]);
        }
        else if (m_actions[n] == 'B') {
            G_hidden = m_pBNLayers->backprop_subnet(G_hidden);
        }
    }

    return G_hidden;
}

/*
Array<float> ConvLayer::m_get_ext_regions_for_conv(Array<float> x) {
    Array<float> regs = m_get_ext_regions(x, 0);
    regs = regs.transpose(Idx(2, 0, 1, 3, 4, 5));
    return regs.reshape(Shape(-1, m_ksize[0] * m_ksize[1] * m_xchn));
}

Array<float> ConvLayer::m_undo_ext_regions_for_conv(Array<float> regs) {
    regs = regs.reshape(Shape(-1, m_xshape[0], m_xshape[1], m_ksize[0], m_ksize[1], m_xchn));
    regs = regs.transpose(Idx(1, 2, 0, 3, 4, 5));
    return m_undo_ext_regions(regs);
}
*/

int64 ConvLayer::dump_structure(int64 depth) {
    int64 param_cnt;
    string kernel_desc = m_get_affine_param_desc(m_param["k"], &param_cnt);
    logger.Print("%*s%s: %s(%d) : %s => %s : %s => %lld pms",
        depth*2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(), kernel_desc.c_str(), param_cnt);
    if (m_pBNLayers) {
        param_cnt += m_pBNLayers->dump_structure(depth + 1);
        logger.Print("%*s%s: %s(%d) : %lld pms", depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, param_cnt);
    }
    return param_cnt;
}

Array<float> ConvLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    Array<float> kernel = m_fetch_weight(m_param["k"]);
    Array<float> bias = m_fetch_bias(m_param["k"]);

    Shape xshape = hidden.shape();
    Shape kshape = kernel.shape();

    assert(xshape.size() >= 4);

    int64 xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
    int64 mb_size = xshape.total_size() / (xh * xw * xchn);
    int64 kh = kshape[0], kw = kshape[1], ychn = kshape[3];
    int64 xs = m_stride[0], ys = m_stride[1];

    Shape yshape = xshape.replace_nth(-1, ychn);
    Shape hshape = xshape;
    Shape bshape = bias.shape();
    Shape cshape = xshape.append(ychn);

    int64 xsize = xshape.total_size();
    int64 ysize = yshape.total_size();

    float* cuda_h = cuda.copy(hidden, "hidden::Conv(forward)");

    float* cuda_k = CudaConn::GetCudaMem(kernel, "kernel::Conv(forward)");
    float* cuda_b = CudaConn::GetCudaMem(bias, "bias::Conv(forward)");

    for (int n = 0; n < (int)m_actions.length(); n++) {
        if (m_actions[n] == 'L') {
            m_aux["Lx"] = CudaConn::Copy(cuda_h, "Lx::Conv(forward)");

            int64 csize = xsize * ychn;

            float* cuda_y = cuda.alloc_float_mem(yshape, "cuda_y::Conv(forward)");
            float* cuda_c = cuda.alloc_float_mem(cshape, "cuda_y::Conv(forward)");

            cu_call(ker_conv_kernel, csize, (csize, cuda_c, cuda_h, cuda_k, xh, xw, kh, kw, xchn, ychn));
            cu_call(ker_conv_sum, ysize, (ysize, cuda_y, cuda_c, xchn));
            cu_call(ker_conv_add_bias, ysize, (ysize, cuda_y, cuda_b, ychn));

            hshape = yshape;
            cuda_h = cuda_y;
        }
        else if (m_actions[n] == 'A') {
            m_aux["Ax"] = CudaConn::Copy(cuda_h, "Ax::Conv(forward)");
            int64 hsize = hshape.total_size();
            cu_call(ker_activate, hsize, (hsize, cuda_h, cuda_h, m_nActFunc, m_leaky_alpha));
            m_aux["Ay"] = CudaConn::Copy(cuda_h, "Ay::Conv(forward)");
        }
        else if (m_actions[n] == 'B') {
            m_pBNLayers->m_forward_bn_core(cuda, cuda_h, hshape);
        }
    }

    Array<float> output;
    bool valid_padding = m_padding == "VALID";

    if (xs != 1 || ys != 1 || valid_padding) {
        Shape sshape = xshape;
        sshape[-3] = m_output_shape[-3];
        sshape[-2] = m_output_shape[-2];
        sshape[-1] = m_output_shape[-1];

        int64 ssize = sshape.total_size();
        float* cuda_s = cuda.alloc_float_mem(sshape, "cuda_s::Conv(forward)");
        int64 sh = sshape[-3], sw = sshape[-2];
        cu_call(ker_stride, ssize, (ssize, cuda_s, cuda_h, xh, xw, sh, sw, ychn, kh, kw, xs, ys, valid_padding));
        output = cuda.detach(cuda_s, "cuda_s=>output::Conv(forward)");
    }
    else {
        output = cuda.detach(cuda_h, "output::Conv(forward)");
    }

    if (m_engine.lookup_option("need_maps")) {
        m_engine.add_map(output);
    }

    return output;
}

Array<float> ConvLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    Array<float> kernel = m_fetch_weight(m_param["k"]);
    Array<float> bias = m_fetch_bias(m_param["k"]);

    Shape sshape = G_hidden.shape();
    Shape kshape = kernel.shape();
    Shape bshape = bias.shape();

    int64 kh = kshape[0], kw = kshape[1], xchn = kshape[2], ychn = kshape[3];
    int64 xs = m_stride[0], ys = m_stride[1];

    Shape xshape = sshape.replace_end(m_input_shape);
    Shape yshape = xshape.replace_nth(-1, ychn);
    Shape hshape = sshape;

    int64 xh = xshape[-3], xw = xshape[-2];

    int64 xsize = xshape.total_size();
    int64 ysize = yshape.total_size();
    int64 ksize = kshape.total_size();

    //float* cuda_gh = cuda.copy(G_hidden, "G_hidden::Conv(backprop)");
    float* cuda_gh = CudaConn::GetCudaMem(G_hidden, "G_hidden::Conv(backprop)");
    float* cuda_k = CudaConn::GetCudaMem(kernel, "k::Conv(backprop)");
    //float* cuda_b = CudaConn::GetCudaMem(bias, "k::Conv(backprop)");

    float* cuda_gk = cuda.alloc_float_mem(kshape, "gk::Conv(backprop)");
    float* cuda_gb = cuda.alloc_float_mem(bshape, "gb::Conv(backprop)");

    bool valid_padding = m_padding == "VALID";

    if (xs != 1 || ys != 1 || valid_padding) {
        float* cuda_gy = cuda.alloc_float_mem(yshape, "gy for stride::Conv(backprop)");
        int64 sh = sshape[-3], sw = sshape[-2];
        cu_call(ker_stride_derv, ysize, (ysize, cuda_gy, cuda_gh, xh, xw, sh, sw, ychn, kh, kw, xs, ys, valid_padding));
        cuda_gh = cuda_gy;
    }

    for (int n = (int)m_actions.length() - 1; n >= 0; n--) {
        if (m_actions[n] == 'L') {
            int64 mb_size = xshape.total_size() / m_input_shape.total_size();
            Shape cshape = xshape.append(ychn);
            Shape dshape = kshape.append(mb_size).append(xh);

            int64 csize = cshape.total_size();
            int64 dsize = dshape.total_size();

            float* cuda_gx = cuda.alloc_float_mem(xshape, "gx for conv::Conv(backprop)");
            float* cuda_c = cuda.alloc_float_mem(cshape, "c::Conv(backprop)");
            float* cuda_d = cuda.alloc_float_mem(dshape, "d::Conv(backprop)");
            float* cuda_x = cuda.attach(m_aux["Lx"], "Lx");

            cu_call(ker_conv_derv_x_kernel, csize, (csize, cuda_c, cuda_gh, cuda_k, xh, xw, kh, kw, xchn, ychn));
            cu_call(ker_conv_derv_x_sum, xsize, (xsize, cuda_gx, cuda_c, ychn));

            int64 sum_size = ksize * mb_size;

            cu_call(ker_conv_derv_kw_x, dsize, (dsize, cuda_d, cuda_gh, cuda_x, mb_size, xh, xw, kh, kw, xchn, ychn));
            cu_call(ker_conv_derv_kw_sum1, sum_size, (sum_size, cuda_d, xw));
            cu_call(ker_conv_derv_kw_sum2, ksize, (ksize, cuda_gk, cuda_d, mb_size, xw));

            int64 size1 = ychn * mb_size * xh;
            int64 size2 = ychn * mb_size;

            cu_call(ker_conv_derv_kb_sum1, size1, (size1, cuda_d, cuda_gh, mb_size, xh, xw, ychn));
            cu_call(ker_conv_derv_kb_sum2, size2, (size2, cuda_d, mb_size, xh, xw, ychn));
            cu_call(ker_conv_derv_kb_sum3, ychn, (ychn, cuda_gb, cuda_d, mb_size, xh, xw, ychn));

            Array<float> G_kernel = cuda.detach(cuda_gk);
            Array<float> G_bias = cuda.detach(cuda_gb);

            hshape = xshape;
            cuda_gh = cuda_gx;
        }
        else if (m_actions[n] == 'A') {
            float* cuda_x = cuda.attach(m_aux["Ax"], "Ax");
            float* cuda_y = cuda.attach(m_aux["Ay"], "Ay");
            int64 hsize = hshape.total_size();
            cu_call(ker_activate_derv, hsize, (hsize, cuda_gh, cuda_gh, cuda_x, cuda_y, m_nActFunc, m_leaky_alpha));
        }
        else if (m_actions[n] == 'B') {
            m_pBNLayers->m_backprop_bn_core(cuda, cuda_gh, hshape);
        }
    }

    Array<float> G_input = cuda.detach(cuda_gh, "conv::G_input");

    return G_input;
}
