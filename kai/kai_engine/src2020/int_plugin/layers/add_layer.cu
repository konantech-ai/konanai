/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "add_layer.cuh"
#include "batch_norm_layer.cuh"

AddLayer::AddLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine) : ComplexLayer(options, layer_subnet, shape, seq, engine), m_pBNLayers(NULL) {
    m_add_x = get_option("x", true);

    List pms;

    int64 max_chn = 0;
    m_b_cnn = shape.size() == 3; // for CNN, MAX, AVG layers...: 채널수 차이 지원, 차원 달라질 수도... 옵션 정보로 처리?
    
    if (m_b_cnn) {
        m_stride = get_2d_option("stride", 1);

        if (m_stride[0] != 1 || m_stride[1] != 1) {
            assert(shape.size() == 3);
            assert(m_stride[0] > 0 || m_stride[1] > 0);
            //assert(m_output_shape[0] % m_stride[0] == 0);
            //assert(m_output_shape[1] % m_stride[1] == 0);

            m_output_shape[0] /= m_stride[0];
            m_output_shape[1] /= m_stride[1];

            //m_output_shape[0] = (m_output_shape[0] - 1) / m_stride[0] + 1;
            //m_output_shape[1] = (m_output_shape[1] - 1) / m_stride[1] + 1;
        }

        max_chn = m_output_shape[2];
        m_output_shape[2] = 0;  // temporal to compare with bshape
    }

    for (List::iterator it = layer_subnet.begin(); it != layer_subnet.end(); it++) {
        Shape bshape = m_input_shape;
        Layer* pPlayer = Layer::CreateLayer(*it, bshape, seq, engine);
        m_layers.push_back(pPlayer);
        pms.push_back(pPlayer->m_param);
        if (m_b_cnn) {
            assert(max_chn % bshape[2] == 0 || bshape[2] % max_chn == 0);
            if (max_chn < bshape[2]) max_chn = bshape[2];
            bshape[2] = 0;  // temporal to compare
        }
        assert(bshape == m_output_shape);
    }

    if (m_b_cnn) {
        m_output_shape[2] = max_chn;
    }

    for (int n = 0; n < (int) m_actions.length(); n++) {
        if (m_actions[n] == 'B') {
            Dict bn_options;
            bn_options["rescale"] = false;
            m_pBNLayers = new BatchNormalLayer(bn_options, m_output_shape, seq, engine);
            m_pBNLayers->m_layer_name = "(batchnorm)";
            m_param["batch_norm"] = m_pBNLayers->m_param;
        }
    }

    m_param["pms"] = pms;

    shape = m_output_shape;
}

AddLayer::~AddLayer() {
    delete m_pBNLayers;
}

Array<float> AddLayer::m_forward_farr(Array<float> hidden) {
    //hidden.print("add in");

    if (m_trace) hidden.print("hidden in");

    int64 mb_size = hidden.axis_size(0);
    int64 ychn = m_output_shape[-1];
    
    Array<float> added = kmath->zeros(m_output_shape.add_front(mb_size));
    if (m_trace) added.print("added");
    Shape ashape = added.shape();

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        Array<float> bhidden = pLayer->forward_subnet(hidden);
        if (m_trace) bhidden.print("bhidden");
        if (m_b_cnn && bhidden.axis_size(-1) != ychn) {
            int ratio = (int)(ychn / bhidden.axis_size(-1));
            bhidden = bhidden.tile(ratio);
            bhidden.reshape(ashape);
            if (m_trace) bhidden.print("bhidden reshaped");
        }

        //bhidden.print("bhidden");

        added += bhidden;
        if (m_trace) added.print("added");
    }

    //added.print("after subnet");

    if (m_add_x) {
        if (m_b_cnn) {
            if (m_stride != Shape(1, 1)) {
                Shape yshape(m_output_shape[0], m_output_shape[1]);
                int64 xchn = hidden.axis_size(-1);

                hidden = hidden.reshape(Shape(mb_size, yshape[0], m_stride[0], yshape[1], m_stride[1], xchn));
                hidden = hidden.transpose(Idx(0, 1, 3, 5, 2, 4));
                hidden = hidden.reshape(Shape(-1, m_stride.total_size()));
                hidden = hidden.avg(0);
                hidden = hidden.reshape(Shape(mb_size, yshape[0], yshape[1], xchn));
            }

            if (hidden.axis_size(-1) != ychn) {
                int ratio = (int)(ychn / hidden.axis_size(-1));
                hidden = hidden.tile(ratio);
                hidden = hidden.reshape(ashape);
            }
        }
        added += hidden;
        if (m_trace) added.print("added after add x");
    }

    if (m_pBNLayers) {
        added = m_pBNLayers->forward_subnet(added);
        if (m_trace) added.print("added after bn");
    }

    //added.print("add out");
    //throw KaiException(KERR_ASSERT);

    return added;
}

Array<float> AddLayer::m_backprop_farr(Array<float> G_hidden) {
    if (m_trace) G_hidden.print("AddLayer::m_backprop::G_hidden in");
    if (m_pBNLayers) {
        G_hidden = m_pBNLayers->backprop_subnet(G_hidden);
        if (m_trace) G_hidden.print("AddLayer::m_backprop::G_hidden bn");
    }

    int64 mb_size = G_hidden.axis_size(0);
    int64 xchn = m_input_shape[-1];

    Array<float> G_input = kmath->zeros(m_input_shape.add_front(mb_size));
    if (m_trace) G_input.print("AddLayer::m_backprop::G_input");

    if (m_add_x) {
        Array<float> G_residual = G_hidden;

        if (m_b_cnn) {
            if (G_residual.axis_size(-1) != xchn) {
                int ratio = (int) (G_residual.axis_size(-1) / xchn);
                G_residual = G_residual.untile(ratio);
            }

            if (m_stride != Shape(1, 1)) {
                Shape yshape(m_output_shape[0], m_output_shape[1]);
                int tile_size = (int) m_stride.total_size();

                G_residual = G_residual.flatten() / (float)tile_size;
                Array<float> G_temp = kmath->zeros(Shape(G_residual.total_size(), tile_size));
                for (int n = 0; n < tile_size; n++) {
                    G_temp[Axis(_all_, n)] = G_residual.reshape(Shape(-1, 1));
                }
                G_residual = G_temp.reshape(Shape(mb_size, yshape[0], yshape[1], xchn, m_stride[0], m_stride[1]));
                G_residual = G_residual.transpose(Idx(0, 1, 4, 2, 5, 3));
                G_residual = G_residual.reshape(G_input.shape());
            }
        }

        G_input = G_residual;
        if (m_trace) G_input.print("AddLayer::m_backprop::G_input residual");
    }

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        Array<float> G_bhidden = pLayer->backprop_subnet(G_hidden);
        if (m_b_cnn) {
            if (G_bhidden.axis_size(-1) != xchn) {
                int ratio = (int) (G_bhidden.axis_size(-1) / xchn);
                G_bhidden = G_bhidden.untile(ratio);
            }
        }
        if (m_trace) G_bhidden.print("AddLayer::m_backprop::G_bhidden");
        G_input += G_bhidden;
        if (m_trace) G_input.print("AddLayer::m_backprop::G_input step");
    }
    if (m_trace) G_input.print("AddLayer::m_backprop::G_input final");

    return G_input;
}

Array<float> AddLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    m_trace = false;

    if (m_trace) hidden.print_shape("AddLayer::m_forward: hidden in");

    //int64 mb_size = hidden.axis_size(0);
    int64 xchn = m_input_shape[-1];
    int64 ychn = m_output_shape[-1];

    Shape xshape = hidden.shape();
    Shape yshape = xshape.replace_end(m_output_shape);
    int64 ysize = yshape.total_size();

    float* cuda_y = cuda.alloc_float_mem(yshape, "out:add(forward)");

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        Array<float> bhidden = pLayer->forward_subnet(hidden);

        float* cuda_b = bhidden.data_ptr();

        if (m_b_cnn && bhidden.axis_size(-1) != ychn) {
            float* cuda_t = cuda.alloc_float_mem(yshape, "tile:add(forward)");
            cu_call(ker_tile_chn, ysize, (ysize, cuda_t, cuda_b, ychn, bhidden.axis_size(-1)));
            cuda_b = cuda_t;
        }

        if (m_trace) cuda.DumpShape(cuda_b, "AddLayer::m_forward: bhidden");

        cu_call(ker_add_on, ysize, (ysize, cuda_y, cuda_b));
    }

    if (m_trace) cuda.DumpShape(cuda_y, "AddLayer::m_forward: subnet");

    if (m_add_x) {
        float* cuda_x = hidden.data_ptr();

        if (m_b_cnn) {
            if (m_stride != Shape(1, 1)) {
                Shape tshape = yshape.replace_nth(-1, xchn);
                int64 tsize = tshape.total_size();

                float* cuda_t = cuda.alloc_float_mem(tshape, "x_resize:add(forward)");

                cu_call(ker_avg_exact, tsize, (tsize, cuda_t, cuda_x, xshape[-3], xshape[-2], xchn, m_stride[0], m_stride[1]));

                cuda_x = cuda_t;
                if (m_trace) cuda.DumpShape(cuda_x, "AddLayer::m_forward: after stride");
            }

            if (hidden.axis_size(-1) != ychn) {
                float* cuda_t = cuda.alloc_float_mem(yshape, "tile:add(forward)");
                cu_call(ker_tile_chn, ysize, (ysize, cuda_t, cuda_x, ychn, hidden.axis_size(-1)));
                cuda_x = cuda_t;
                if (m_trace) cuda.DumpShape(cuda_x, "AddLayer::m_forward: tile");
            }
        }

        cu_call(ker_add_on, ysize, (ysize, cuda_y, cuda_x));

        if (m_trace) cuda.DumpShape(cuda_y, "AddLayer::m_forward: add x");
    }

    if (m_pBNLayers) {
        m_pBNLayers->m_forward_bn_core(cuda, cuda_y, yshape);
        if (m_trace) cuda.DumpShape(cuda_y, "AddLayer::m_forward: normalize");
    }

    Array<float> output = cuda.detach(cuda_y, "#netout#");

    if (m_trace) output.print_shape("add out");

    return output;
}

Array<float> AddLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    m_trace = false;

    if (m_trace) G_hidden.print_shape("AddLayer::backprop: G_input");

    //int64 mb_size = G_hidden.axis_size(0);
    int64 xchn = m_input_shape[-1];

    Shape hshape = G_hidden.shape();
    Shape xshape = hshape.replace_end(m_input_shape); // m_input_shape.add_front(mb_size);
    int64 xsize = xshape.total_size();

    float* cuda_gh = cuda.copy(G_hidden, "G_hidden::parallel(backprop)");
    float* cuda_gx = NULL;

    if (m_pBNLayers) {
        m_pBNLayers->m_backprop_bn_core(cuda, cuda_gh, hshape);
        if (m_trace) cuda.DumpShape(cuda_gh, "AddLayer::backprop: normalize");
    }

    if (m_add_x) {
        Array<float> G_residual = G_hidden;

        if (m_b_cnn) {
            if (G_residual.axis_size(-1) != xchn) {
                Shape tshape = G_residual.shape().replace_nth(-1, xchn);
                int64 tsize = tshape.total_size();
                float* cuda_t = cuda.alloc_float_mem(tshape, "untile-y:add(backprop)");
                cu_call(ker_untile_chn, tsize, (tsize, cuda_t, cuda_gh, G_residual.axis_size(-1), xchn));
                cuda_gh = cuda_t;
                if (m_trace) cuda.DumpShape(cuda_gh, "AddLayer::backprop: untile");
            }

            if (m_stride != Shape(1, 1)) {
                float* cuda_t = cuda.alloc_float_mem(xshape, "x_resize:add(backprop)");
                cu_call(ker_avg_exact_derv, xsize, (xsize, cuda_t, cuda_gh, xshape[-3], xshape[-2], xchn, m_stride[0], m_stride[1]));
                cuda_gh = cuda_t;
                if (m_trace) cuda.DumpShape(cuda_gh, "AddLayer::backprop: atride");
            }
        }

        cuda_gx = cuda_gh; // cuda.copy(G_residual, "residual:add(backprop)");
        if (m_trace) cuda.DumpShape(cuda_gx, "AddLayer::backprop: add x");
    }
    else {
        cuda_gx = cuda.alloc_float_mem(xshape, "out:parallel(backprop)");
        if (m_trace) cuda.DumpShape(cuda_gx, "AddLayer::backprop: add x bypass");
    }

    if (m_trace) cuda.DumpShape(cuda_gx, "AddLayer::backprop: before subnet");

    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        Array<float> G_bhidden = pLayer->backprop_subnet(G_hidden);
        float* cuda_gb = G_bhidden.data_ptr();

        if (m_trace) cuda.DumpShape(cuda_gb, "gb1");

        if (m_b_cnn) {
            if (G_bhidden.axis_size(-1) != xchn) {
                float* cuda_t = cuda.alloc_float_mem(xshape, "untile-b:add(backprop)");
                cu_call(ker_untile_chn, xsize, (xsize, cuda_t, cuda_gb, G_bhidden.axis_size(-1), xchn));
                cuda_gb = cuda_t;
            }
        }

        if (m_trace) cuda.DumpShape(cuda_gb, "gb2");
        if (m_trace) cuda.DumpShape(cuda_gb, "AddLayer::backprop: branch");

        cu_call(ker_add_on, xsize, (xsize, cuda_gx, cuda_gb));
        if (m_trace) cuda.DumpShape(cuda_gx, "AddLayer::backprop: add branch");
    }

    Array<float> G_input = cuda.detach(cuda_gx, "add::G_input");

    if (m_trace) G_input.print_shape("add:G_input");

    return G_input;
}
