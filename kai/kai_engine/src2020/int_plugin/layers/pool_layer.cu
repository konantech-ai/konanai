/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "pool_layer.cuh"

PoolLayer::PoolLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : ConvPoolLayer(options, shape, seq, engine) {
    m_stride = get_2d_option("stride", 1);
    m_ksize = get_2d_option("ksize", m_stride);

    m_ychn = m_xchn;

    m_is_simple = (m_stride == m_ksize) && (m_xshape.mod(m_stride) == 0) && (m_padding == "SAME");

    if (m_padding == "VALID") {
        m_xshape = m_xshape.sub(m_ksize) + 1;
    }

    m_output_shape = m_xshape.div(m_stride).append(m_ychn);

    shape = m_output_shape;
}

PoolLayer::~PoolLayer() {
}

MaxLayer::MaxLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : PoolLayer(options, shape, seq, engine) {
}

MaxLayer::~MaxLayer() {
}

Array<float> MaxLayer::m_forward_farr(Array<float> hidden) {
    int64 mb_size = hidden.shape()[0];
    Array<int64> arg;
    m_aux["xshape"] = hidden.shape();

    if (m_is_simple) {
        Shape yshape = m_xshape.div(m_stride);
        hidden = hidden.reshape(Shape(mb_size, yshape[0], m_stride[0], yshape[1], m_stride[1], m_xchn));
        hidden = hidden.transpose(Idx(0, 1, 3, 5, 2, 4));
        m_aux["hshape"] = hidden.shape();
        hidden = hidden.reshape(Shape(-1, m_stride[0] * m_stride[1]));
        hidden = hidden.maxarg(0, arg);
        hidden = hidden.reshape(Shape(mb_size, yshape[0], yshape[1], m_xchn));
    }
    else {
        hidden = m_get_ext_regions(hidden, -FLT_MAX);
        hidden = hidden.transpose(Idx(2, 5, 0, 1, 3, 4));
        
        //m_aux["hshape"] = hidden.shape(); // 잘못 저장되고 오류 발생, 확인 필요, 바로 확인 시 무한 루프 의심 현상 발생, backprop에서 확인시 바뀐 값 나옴
        m_hshape = hidden.shape(); // 일단 로컬 변수 두어 위의 처리를 대신해봄
        hidden = hidden.reshape(Shape(-1, m_ksize.total_size()));
        hidden = hidden.maxarg(0, arg);
        hidden = hidden.reshape(Shape(mb_size, m_xchn, m_xshape[0], m_xshape[1]));
        hidden = hidden.transpose(Idx(0, 2, 3, 1));
        hidden = m_stride_filter(hidden);
    }

    m_aux["arg"] = arg;

    //arg.print("arg forward");

    if (m_engine.lookup_option("need_maps")) {
        m_engine.add_map(hidden);
    }

    return hidden;
}

Array<float> MaxLayer::m_backprop_farr(Array<float> G_hidden) {
    Array<int64> arg = m_aux["arg"];
    Shape xshape = m_aux["xshape"];

    //arg.print("arg backprop");

    if (m_is_simple) {
        Shape hshape = m_aux["hshape"];

        G_hidden = G_hidden.flatten();

        Array<float> G_tiles = kmath->zeros(Shape(G_hidden.total_size(), m_stride.total_size()));
        G_tiles.setarg(arg, G_hidden);
        G_tiles = G_tiles.reshape(hshape);
        G_tiles = G_tiles.transpose(Idx(0, 1, 4, 2, 5, 3));
        G_hidden = G_tiles.reshape(xshape);
    }
    else {
        G_hidden = m_stride_filter_derv(G_hidden);

        G_hidden = G_hidden.transpose(Idx(0, 3, 1, 2));
        G_hidden = G_hidden.flatten();

        Array<float> G_h_flat = kmath->zeros(Shape(G_hidden.total_size(), m_ksize.total_size()));
        G_h_flat.setarg(arg, G_hidden);
        G_h_flat = G_h_flat.reshape(m_hshape);
        G_h_flat = G_h_flat.transpose(Idx(2, 3, 0, 4, 5, 1));
        G_hidden = m_undo_ext_regions(G_h_flat);
    }

    return G_hidden;
}

Array<float> MaxLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    Shape hshape = hidden.shape();

    int64 xh = hshape[-3], xw = hshape[-2], xchn = hshape[-1];
    int64 kh = m_ksize[0], kw = m_ksize[1];
    int64 xs = m_stride[0], ys = m_stride[1];

    int64 hsize = hshape.total_size();

    float* cuda_h = CudaConn::GetCudaMem(hidden, "h:Max(forward)");
    float* cuda_y = cuda.alloc_float_mem(hshape, "y:Max(forward)");

    int64* cuda_n = cuda.alloc_int64_mem(hshape, "n:Max(forward)");

    cu_call(ker_max, hsize, (hsize, cuda_y, cuda_n, cuda_h, xh, xw, xchn, kh, kw));

    Array<float> output;
    bool valid_padding = m_padding == "VALID";

    //output = cuda.array_from_dev(cuda_y, hshape);
    //output.print("max layer output before stride filter");

    if (xs != 1 || ys != 1 || valid_padding) {
        Shape sshape = hshape.replace_end(m_output_shape);
        int64 ssize = sshape.total_size();
        float* cuda_s = cuda.alloc_float_mem(sshape, "cuda_s::Max(forward)");
        int64 sh = sshape[-3], sw = sshape[-2];
        cu_call(ker_stride, ssize, (ssize, cuda_s, cuda_h, xh, xw, sh, sw, xchn, kh, kw, xs, ys, valid_padding));
        output = cuda.detach(cuda_s, "cuda_s=>output::Max(forward)");
    }
    else {
        output = cuda.detach(cuda_y, "output::Max(forward)");
    }

    if (m_engine.lookup_option("need_maps")) {
        m_engine.add_map(output);
    }

    m_aux["arg"] = cuda.detach(cuda_n, "n:Max(forward)");

    return output;
}

Array<float> MaxLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    int64 xs = m_stride[0], ys = m_stride[1];

    Shape hshape = G_hidden.shape();
    Shape xshape = hshape.replace_end(m_input_shape);

    int64 xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
    int64 kh = m_ksize[0], kw = m_ksize[1];
    int64 xsize = xshape.total_size();

    float* cuda_gh = CudaConn::GetCudaMem(G_hidden, "gh::Max(backprop)");
    bool valid_padding = m_padding == "VALID";

    if (xs != 1 || ys != 1 || valid_padding) {
        Shape sshape = hshape.replace_end(m_output_shape);
        int64 sh = sshape[-3], sw = sshape[-2];

        float* cuda_gx = cuda.alloc_float_mem(xshape, "gx for stride::Max(backprop)");

        cu_call(ker_stride_derv, xsize, (xsize, cuda_gx, cuda_gh, xh, xw, sh, sw, xchn, kh, kw, xs, ys, valid_padding));
        cuda_gh = cuda_gx;
    }

    float* cuda_gx = cuda.alloc_float_mem(xshape, "gx for max::Max(backprop)");
    int64* cuda_n = cuda.attach_int64(m_aux["arg"], "arg");

    cu_call(ker_max_derv, xsize, (xsize, cuda_gx, cuda_n, cuda_gh, xh, xw, xchn, kh, kw));

    Array<float> G_input = cuda.detach(cuda_gx, "G_input::Max(backprop)");

    return G_input;
}

AvgLayer::AvgLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : PoolLayer(options, shape, seq, engine) {
    if (!m_is_simple && !CudaConn::UsingCuda()) {
        m_mask = kmath->ones(m_input_shape.add_front(1));
        m_mask = m_get_ext_regions(m_mask, 0);
        m_mask = m_mask.transpose(Idx(2, 5, 0, 1, 3, 4));
        m_mask = m_mask.reshape(Shape(-1, m_ksize.total_size()));
        m_mask = kmath->sum(m_mask, 0);
    }
}

AvgLayer::~AvgLayer() {
}

Array<float> AvgLayer::m_forward_farr(Array<float> hidden) {
    int64 mb_size = hidden.shape()[0];
    int64 tile_size = m_ksize.total_size();

    if (m_is_simple) {
        Shape yshape = m_xshape.div(m_stride);
        hidden = hidden.reshape(Shape(mb_size, yshape[0], m_stride[0], yshape[1], m_stride[1], m_xchn));
        hidden = hidden.transpose(Idx(0, 1, 3, 5, 2, 4));
        hidden = hidden.reshape(Shape(-1, tile_size));
        hidden = hidden.avg(0);
        hidden = hidden.reshape(Shape(mb_size, yshape[0], yshape[1], m_xchn));
    }
    else {
        hidden = m_get_ext_regions(hidden, 0);
        hidden = hidden.transpose(Idx(2, 5, 0, 1, 3, 4));
        hidden = hidden.reshape(Shape(hidden.total_size() / tile_size, tile_size));
        hidden = hidden.sum(0);
        hidden = hidden.reshape(Shape(mb_size, -1));
        hidden = hidden / m_mask;
        hidden = hidden.reshape(Shape(mb_size, m_ychn, m_xshape[0], m_xshape[1]));
        hidden = hidden.transpose(Idx(0, 2, 3, 1));
        hidden = m_stride_filter(hidden);
    }

    if (m_engine.lookup_option("need_maps")) {
        m_engine.add_map(hidden);
    }

    return hidden;
}

Array<float> AvgLayer::m_backprop_farr(Array<float> G_hidden) {
    Array<float> G_input;

    int64 mb_size = G_hidden.shape()[0];
    int64 tile_size = m_ksize.total_size();

    if (m_is_simple) {
        Shape yshape = m_xshape.div(m_stride);
        G_hidden = G_hidden.flatten() / (float)tile_size;
        G_input = kmath->zeros(Shape(mb_size*yshape[0]*yshape[1]*m_xchn, tile_size));
        for (int n = 0; n < tile_size; n++) {
            G_input[Axis(_all_, n)] = G_hidden.reshape(Shape(-1,1));
        }
        G_input = G_input.reshape(Shape(mb_size, yshape[0], yshape[1], m_xchn, m_stride[0], m_stride[1]));
        G_input = G_input.transpose(Idx(0, 1, 4, 2, 5, 3));
        G_input = G_input.reshape(Shape(mb_size, m_xshape[0], m_xshape[1], m_xchn));

        return G_input;
    }
    else {
        G_hidden = m_stride_filter_derv(G_hidden);
        G_hidden = G_hidden.transpose(Idx(0, 3, 1, 2));
        G_hidden = G_hidden.reshape(Shape(mb_size, -1));
        G_hidden = G_hidden / m_mask;
        G_hidden = G_hidden.tile((int)(m_ksize[0] * m_ksize[1]));
        G_hidden = G_hidden.reshape(Shape(mb_size, m_xchn, m_xshape[0], m_xshape[1], m_ksize[0], m_ksize[1]));
        G_hidden = G_hidden.transpose(Idx(2, 3, 0, 4, 5, 1));
        G_hidden = m_undo_ext_regions(G_hidden);

        return G_hidden;
    }
}

Array<float> AvgLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    Shape hshape = hidden.shape();

    int64 xh = hshape[-3], xw = hshape[-2], xchn = hshape[-1];
    int64 kh = m_ksize[0], kw = m_ksize[1];
    int64 xs = m_stride[0], ys = m_stride[1];

    int64 hsize = hshape.total_size();

    float* cuda_h = CudaConn::GetCudaMem(hidden, "h:Avg(forward)");
    float* cuda_y = cuda.alloc_float_mem(hshape, "y:Avg(forward)");

    int64* cuda_n = cuda.alloc_int64_mem(hshape, "n:Avg(forward)");

    cu_call(ker_avg, hsize, (hsize, cuda_y, cuda_n, cuda_h, xh, xw, xchn, kh, kw));

    m_aux["arg"] = cuda.detach(cuda_n, "n:Avg(forward)");

    Array<float> output;
    bool valid_padding = m_padding == "VALID";

    if (xs != 1 || ys != 1 || valid_padding) {
        Shape sshape = hshape.replace_end(m_output_shape);
        int64 ssize = sshape.total_size();
        float* cuda_s = cuda.alloc_float_mem(sshape, "cuda_s::Avg(forward)");
        int64 sh = sshape[-3], sw = sshape[-2];
        cu_call(ker_stride, ssize, (ssize, cuda_s, cuda_h, xh, xw, sh, sw, xchn, kh, kw, xs, ys, valid_padding));
        output = cuda.detach(cuda_s, "cuda_s=>output::Avg(forward)");
    }
    else {
        output = cuda.detach(cuda_y, "output::Avg(forward)");
    }

    return output;
}

Array<float> AvgLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    CudaConn cuda("backprop", this);

    int64 xs = m_stride[0], ys = m_stride[1];

    Shape hshape = G_hidden.shape();
    Shape xshape = hshape.replace_end(m_input_shape);

    int64 xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
    int64 kh = m_ksize[0], kw = m_ksize[1];
    int64 xsize = xshape.total_size();

    float* cuda_gh = CudaConn::GetCudaMem(G_hidden, "gh::Avg(backprop)");
    bool valid_padding = m_padding == "VALID";

    if (xs != 1 || ys != 1 || valid_padding) {
        Shape sshape = hshape.replace_end(m_output_shape);
        int64 sh = sshape[-3], sw = sshape[-2];

        float* cuda_gx = cuda.alloc_float_mem(xshape, "gx for stride::Avg(backprop)");

        cu_call(ker_stride_derv, xsize, (xsize, cuda_gx, cuda_gh, xh, xw, sh, sw, xchn, kh, kw, xs, ys, valid_padding));
        cuda_gh = cuda_gx;
    }

    float* cuda_gx = cuda.alloc_float_mem(xshape, "gx for avg::Avg(backprop)");
    int64* cuda_n = cuda.attach_int64(m_aux["arg"], "arg");

    cu_call(ker_avg_derv, xsize, (xsize, cuda_gx, cuda_n, cuda_gh, xh, xw, xchn, kh, kw));

    Array<float> G_input = cuda.detach(cuda_gx, "G_input::Avg(backprop)");

    return G_input;
}

GlobalAvgLayer::GlobalAvgLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : AvgLayer(options, shape, seq, engine) {
}

GlobalAvgLayer::~GlobalAvgLayer() {
}
