/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "rnn_layer.cuh"

static bool bRnnTrace = false;
static bool bCrossTesting = false;

SeqLayer::SeqLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
    m_inseq = m_seq;
    seq = m_outseq = get_option("outseq", true);
}

SeqLayer::~SeqLayer() {
}

RnnLayer::RnnLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : SeqLayer(options, shape, seq, engine) {
    assert(m_inseq || m_outseq);

    if (! m_inseq) m_timesteps = get_option("timesteps");

    m_lstm = get_option("lstm", true);
    m_use_state = get_option("use_state", false);

    m_recur_size = get_option("recur_size");
    m_timefeats = (int) m_input_shape.total_size();
    m_ex_inp_dim = m_timefeats + m_recur_size;

    if (m_lstm) {
        m_param = alloc_affine_param(Shape(m_ex_inp_dim, 4*m_recur_size), true);
        Array<float> bias = m_fetch_bias(m_param);
        bias[Axis(Ax(0, m_recur_size), _all_)] = 1.0;
    }
    else {
        m_param = alloc_affine_param(Shape(m_ex_inp_dim, m_recur_size), true);
        set_activate((string)get_option("actfunc", "relu"));
    }

    shape = m_output_shape = Shape(m_recur_size);
 }

RnnLayer::~RnnLayer() {
}

Array<float> RnnLayer::m_forward_farr(Array<float> hidden) {
    if (bCrossTesting) hidden.print("nocuda forward input");
    if (m_inseq) m_timesteps = (int) hidden.axis_size(1);

    m_aux["xshape"] = hidden.shape();

    int64 mb_size = hidden.axis_size(0);

    Array<float> recurrent = hmath.zeros(Shape(mb_size, m_recur_size));
    Array<float> state = hmath.zeros(Shape(mb_size, m_recur_size));
    Array<float> outputs = hmath.zeros(Shape(mb_size, m_timesteps, m_recur_size));

    //Array<float> weight = m_pm["w"], bias = m_pm["b"];

    Array<float> x_slice = hmath.zeros(Shape(mb_size, m_timefeats));
    Array<float> x_slice_buf, output_buf;

    if (!m_inseq) x_slice = hidden;

    assert(m_rnn_haux.size() == 0);

    for (int tn = 0; tn < m_timesteps; tn++) {
        if (m_inseq) {
            x_slice_buf = hidden[Axis(_all_, tn, _all_)];
            x_slice = x_slice_buf.reshape(Shape(mb_size, m_timefeats));
        }

        Dict aux_step;

        Array<float> ex_inp = hmath.hstack(x_slice, recurrent);

        Array<float> affine = forward_affine(m_param, ex_inp);
        //Array<float> affine = hmath.matmul(ex_inp, weight) + bias;

        aux_step["ex_inp"] = ex_inp;

        if (m_lstm) {
            Array<float> forget_gate, input_gate, output_gate, block_input;

            forget_gate = affine[Axis(_all_, Ax(0 * m_recur_size, 1 * m_recur_size))];
            input_gate = affine[Axis(_all_, Ax(1 * m_recur_size, 2 * m_recur_size))];
            output_gate = affine[Axis(_all_, Ax(2 * m_recur_size, 3 * m_recur_size))];
            block_input = affine[Axis(_all_, Ax(3 * m_recur_size, 4 * m_recur_size))];

            forget_gate = hmath.sigmoid(forget_gate);
            input_gate = hmath.sigmoid(input_gate);
            output_gate = hmath.sigmoid(output_gate);
            block_input = hmath.tanh(block_input);

            aux_step["forget_gate"] = forget_gate;
            aux_step["input_gate"] = input_gate;
            aux_step["output_gate"] = output_gate;
            aux_step["block_input"] = block_input;

            aux_step["state_tmp"] = state;
            state = state * forget_gate + block_input * input_gate;

            recurrent = hmath.tanh(state);
            aux_step["recur_tmp"] = recurrent;
            recurrent = recurrent * output_gate;

            output_buf = m_use_state ? state : recurrent;
        }
        else {
            recurrent = activate(affine);
            output_buf = recurrent;

            aux_step["affine"] = affine;
            aux_step["recurrent"] = recurrent;
        }

        if (m_engine.m_is_training) m_rnn_haux.push_back(aux_step);

        outputs[Axis(_all_, tn, _all_)] = output_buf.reshape(Shape(mb_size, 1, m_recur_size));
    }

    Array<float> output = outputs;
    if (!m_outseq) {
        output = hmath.select_rnn_last_col(outputs);
    }
    else {
        output = outputs;
    }

    if (bCrossTesting) output.print("nocuda forward output");

    return output;
}

Array<float> RnnLayer::m_backprop_farr(Array<float> G_hidden) {
    if (bCrossTesting) G_hidden.print("nocuda backprop input");

    Array<float> weight = m_fetch_weight(m_param);
    Array<float> bias = m_fetch_bias(m_param);

    if (bCrossTesting) {
        weight = weight.to_host();
        bias = bias.to_host();
    }

    Array<float> G_weight = hmath.zeros(weight.shape());
    Array<float> G_bias = hmath.zeros(bias.shape());

    int64 mb_size = G_hidden.axis_size(0);

    Shape xshape = m_aux["xshape"];

    Array<float> G_xs = hmath.zeros(xshape);
    Array<float> G_recurrent = hmath.zeros(Shape(mb_size, m_recur_size));
    Array<float> G_state = hmath.zeros(Shape(mb_size, m_recur_size));

    Array<float> G_y_slice, G_affine, G_ex_inp;

    if (!m_outseq) {
        if (!m_lstm || !m_use_state)
            G_recurrent.copy(G_hidden);
        else
            G_state.copy(G_hidden);
    }

    for (int tn = m_timesteps - 1; tn >= 0; tn--) {
        Dict aux_step = m_rnn_haux.back();
        m_rnn_haux.pop_back();

        if (m_outseq) {
            G_y_slice = G_hidden[Axis(_all_, tn, _all_)];
            if (!m_lstm || !m_use_state) G_recurrent += G_y_slice.reshape(Shape(mb_size, m_recur_size));
            else G_state += G_y_slice.reshape(Shape(mb_size, m_recur_size));
        }

        if (m_lstm) {
            Array<float> forget_gate, input_gate, output_gate, block_input, recur_tmp, state_tmp;

            forget_gate = aux_step["forget_gate"];
            input_gate = aux_step["input_gate"];
            output_gate = aux_step["output_gate"];
            block_input = aux_step["block_input"];
            recur_tmp = aux_step["recur_tmp"];
            state_tmp = aux_step["state_tmp"];

            Array<float> G_recur_tmp = G_recurrent * output_gate;
            Array<float> G_output_gate = G_recurrent * recur_tmp;

            G_state += hmath.tanh_derv(recur_tmp) * G_recur_tmp;

            Array<float> G_input_gate = G_state * block_input;
            Array<float> G_block_input = G_state * input_gate;

            Array<float> G_forget_gate = G_state * state_tmp;
            G_state = G_state * forget_gate;

            G_affine = hmath.zeros(Shape(mb_size, 4 * m_recur_size));

            G_affine[Axis(_all_, Ax(0 * m_recur_size, 1 * m_recur_size))] = hmath.sigmoid_derv(forget_gate) * G_forget_gate;
            G_affine[Axis(_all_, Ax(1 * m_recur_size, 2 * m_recur_size))] = hmath.sigmoid_derv(input_gate) * G_input_gate;
            G_affine[Axis(_all_, Ax(2 * m_recur_size, 3 * m_recur_size))] = hmath.sigmoid_derv(output_gate) * G_output_gate;
            G_affine[Axis(_all_, Ax(3 * m_recur_size, 4 * m_recur_size))] = hmath.tanh_derv(block_input) * G_block_input;
        }
        else {
            Array<float> affine = aux_step["affine"];
            Array<float> recurrent = aux_step["recurrent"];

            G_affine = activate_derv(G_recurrent, affine, recurrent);
        }

        Array<float> ex_inp = aux_step["ex_inp"];

        Array<float> g_affine_weight = ex_inp.transpose();
        Array<float> g_affine_input = weight.transpose();

        G_weight += hmath.matmul(g_affine_weight, G_affine);
        G_bias += hmath.sum(G_affine, -1);
        G_ex_inp = hmath.matmul(G_affine, g_affine_input);

        Array<float> piece1, piece2;

        piece1 = G_ex_inp[Axis(_all_, Ax(0, m_timefeats))];
        piece2 = G_ex_inp[Axis(_all_, Ax(m_timefeats, m_ex_inp_dim))];

        if (m_inseq)
            G_xs[Axis(_all_, tn, _all_)] = piece1.reshape(Shape(mb_size, 1, m_timefeats));
        else
            G_xs += piece1;

        G_recurrent = piece2.reshape(Shape(mb_size, m_recur_size));
    }

    if (bCrossTesting) {
        G_weight.print("nocuda backprop G_weight");
        G_bias.print("nocuda backprop G_bias");
    }

    m_update_weight(m_param, G_weight);
    m_update_bias(m_param, G_bias);

    assert(m_rnn_haux.size() == 0);

    if (bCrossTesting) G_xs.print("nocuda backprop output");

    return G_xs;
}

int64 RnnLayer::dump_structure(int64 depth) {
    int64 param_cnt;
    string pm_desc = m_get_affine_param_desc(m_param, &param_cnt);

    logger.Print("%*s%s: %s(%d) : %s => %s : %s => %lld pms",
        depth * 2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str(),
        pm_desc.c_str(), param_cnt);
    return param_cnt;
}

Array<float> RnnLayer::m_forward_cuda_farr(Array<float> hidden) {
    CudaConn cuda("forward", this);

    if (bCrossTesting) m_forward_farr(hidden.to_host());
    if (bCrossTesting) hidden.print("cuda forward input");

    if (m_inseq) m_timesteps = (int)hidden.axis_size(-m_input_shape.size()-1);

    //int64 mb_size = hidden.axis_size(0);

    if (bRnnTrace) hidden.print("hidden");

    float* cuda_h = cuda.copy_to_buffer(hidden, "x:rnn(for)");

    Array<float> weight = m_fetch_weight(m_param);
    Array<float> bias = m_fetch_bias(m_param);

    if (bRnnTrace) weight.print("weight");
    if (bRnnTrace) bias.print("bias");

    int64 ex_inp_size = m_timefeats + m_recur_size;
    int64 ncol = weight.axis_size(-1);

    Shape hshape = hidden.shape();
    Shape rshape = hshape.replace_tail(m_input_shape.size(), m_output_shape);
    if (m_inseq) rshape = hshape.replace_tail(m_input_shape.size() + 1, m_output_shape);
    Shape eshape = rshape.replace_end(ex_inp_size);
    Shape ashape = rshape.replace_end(ncol);

    int64 mb_size = ashape.total_size() / ncol;

    Shape fshape = hshape.remove_tail(m_input_shape.size() + (m_inseq ? 1 : 0));
    
    //if (m_outseq) fshape = fshape.append(m_timesteps);
    fshape = fshape.append(m_timesteps);
    // 출력이 시계열이 아닐 경우 굳이 시간대별 출력을 모을 필요가 없으므로 처리 과정을 손질하여 위 줄을 주석 처리한 그 위 줄로 대치하는 방안을 고려할 것
    // 기존 접근에서는 데이터별 마지막 시간대 출력을 모으느라 시갖대별 출력을 보존했으나 전체적으로 마지막 출력을 보내면 굳이 그럴 필요 없음
    // 단, 시간대별 마지막 출력 대신 전체 마지막 시간대 출력을 사용하는 현재의 접근에 성능 영향이 없는지 검증이 필요하므로 적용은 일단 보류

    fshape = fshape.append(m_output_shape);

    m_aux["xshape"] = hshape;
    m_aux["rshape"] = rshape;

    /*
    Shape old_rshape(mb_size, m_recur_size);
    Shape old_fshape(mb_size, m_timesteps, m_recur_size);
    Shape old_eshape(mb_size, ex_inp_size);
    Shape old_ashape(mb_size, ncol);
    */

    int64 rsize = rshape.total_size();
    int64 esize = eshape.total_size();
    int64 asize = ashape.total_size();

    float* cuda_w = CudaConn::GetCudaMem(weight, "w:rnn(for)");
    float* cuda_b = CudaConn::GetCudaMem(bias, "b:rnn(for)");

    float* cuda_rec = cuda.alloc_float_mem(rshape, "r:rnn(for)");
    float* cuda_state = cuda.alloc_float_mem(rshape, "s:rnn(for)");
    float* cuda_outputs = cuda.alloc_float_mem(fshape, "ys:rnn(for)");

    float* cuda_ex_inp = cuda.alloc_float_mem(eshape, "ex_inp:rnn(for)");
    float* cuda_affine = cuda.alloc_float_mem(ashape, "affine:rnn(for)");

    float* cuda_fgate = NULL;
    float* cuda_igate = NULL;
    float* cuda_ogate = NULL;
    float* cuda_block = NULL;

    if (m_lstm) {
        cuda_fgate = cuda.alloc_float_mem(rshape, "forget_gate:rnn(for)");
        cuda_igate = cuda.alloc_float_mem(rshape, "input_gate:rnn(for)");
        cuda_ogate = cuda.alloc_float_mem(rshape, "output_gate:rnn(for)");
        cuda_block = cuda.alloc_float_mem(rshape, "input_block:rnn(for)");
    }

    for (int tn = 0; tn < m_timesteps; tn++) {
        Dict aux_step;

        if (bRnnTrace) CudaConn::DumpArr(cuda_rec, "cuda_rec");

        cu_call(ker_rnn_combine_ex_inp, esize, (esize, cuda_ex_inp, cuda_h, cuda_rec, m_timesteps, m_timefeats, m_recur_size, m_inseq, tn));

        if (bRnnTrace) CudaConn::DumpArr(cuda_ex_inp, "cuda_ex_inp");

        if (0 && tn == 1) {
            CudaConn::DumpArr(cuda_h, "cuda_h");
            CudaConn::DumpArr(cuda_rec, "cuda_rec");
            CudaConn::DumpArr(cuda_ex_inp, "cuda_ex_inp");
        }

        cu_call(ker_matmul, asize, (asize, cuda_affine, cuda_ex_inp, cuda_w, mb_size, ex_inp_size, ncol));
        cu_call(ker_add_bias, asize, (asize, cuda_affine, cuda_b, mb_size, ex_inp_size, ncol));

        if (0 && tn == 1) {
            CudaConn::DumpArr(cuda_w, "cuda_w");
            CudaConn::DumpArr(cuda_b, "cuda_b");
            CudaConn::DumpArr(cuda_affine, "cuda_affine");
        }

        if (bRnnTrace) CudaConn::DumpArr(cuda_affine, "cuda_affine");

        aux_step["ex_inp"] = CudaConn::Copy(cuda_ex_inp, "ex_inp:rnn(for)");

        if (m_lstm) {
            // 아래 다섯 줄은 하나의 커널에서 한꺼번에 처리하도록 하여 가속화 가능할 것으로 보임
            cu_call(ker_lstm_split_affine, asize, (asize, cuda_fgate, cuda_igate, cuda_ogate, cuda_block, cuda_affine, m_recur_size));

            if (bRnnTrace) CudaConn::DumpArr(cuda_fgate, "cuda_fgate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_igate, "cuda_igate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_ogate, "cuda_ogate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_block, "cuda_block");

            cu_call(ker_sigmoid_on, rsize, (rsize, cuda_fgate));
            cu_call(ker_sigmoid_on, rsize, (rsize, cuda_igate));
            cu_call(ker_sigmoid_on, rsize, (rsize, cuda_ogate));
            cu_call(ker_tanh_on, rsize, (rsize, cuda_block));

            if (bRnnTrace) CudaConn::DumpArr(cuda_fgate, "cuda_fgate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_igate, "cuda_igate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_ogate, "cuda_ogate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_block, "cuda_block");

            aux_step["pre_state"] = CudaConn::Copy(cuda_state, "state:rnn(for)");

            float* cuda_new_state = cuda.alloc_float_mem(rshape, "ns:rnn(for)");

            cu_call(ker_lstm_new_state, rsize, (rsize, cuda_new_state, cuda_state, cuda_fgate, cuda_block, cuda_igate)); // state = state * forget_gate + block_input * input_gate;

            aux_step["state"] = cuda.detach(cuda_state, "state:rnn(for)");

            cuda_state = cuda_new_state;
            if (bRnnTrace) CudaConn::DumpArr(cuda_state, "cuda_state");
            if (bRnnTrace) CudaConn::DumpArr(cuda_rec, "cuda_rec");

            cu_call(ker_tanh_to, rsize, (rsize, cuda_rec, cuda_state));
            if (bRnnTrace) CudaConn::DumpArr(cuda_state, "cuda_state");
            if (bRnnTrace) CudaConn::DumpArr(cuda_rec, "cuda_rec");

            aux_step["recur_pre"] = CudaConn::Copy(cuda_rec, "recur_pre:rnn(for)");

            cu_call(ker_mult_on, rsize, (rsize, cuda_rec, cuda_ogate));
            if (bRnnTrace) CudaConn::DumpArr(cuda_rec, "cuda_rec");

            aux_step["forget_gate"] = CudaConn::Copy(cuda_fgate, "fg:rnn(for)");
            aux_step["input_gate"] = CudaConn::Copy(cuda_igate, "ig:rnn(for)");
            aux_step["output_gate"] = CudaConn::Copy(cuda_ogate, "og:rnn(for)");
            aux_step["block_input"] = CudaConn::Copy(cuda_block, "ib:rnn(for)");

            //cu_call(ker_rnn_fill_output_slice, rsize, (rsize, (m_use_state ? cuda_state : cuda_outputs), cuda_rec, m_timesteps, m_recur_size, tn)); // 오류 확인 필요
            cu_call(ker_rnn_fill_output_slice, rsize, (rsize, cuda_outputs, (m_use_state ? cuda_state : cuda_rec), m_timesteps, m_recur_size, tn)); // 
            if (bRnnTrace) CudaConn::DumpArr(cuda_outputs, "cuda_outputs");
        }
        else {
            cu_call(ker_activate, rsize, (rsize, cuda_rec, cuda_affine, m_nActFunc, m_leaky_alpha));
            cu_call(ker_rnn_fill_output_slice, rsize, (rsize, cuda_outputs, cuda_rec, m_timesteps, m_recur_size, tn));
        }

        aux_step["affine"] = CudaConn::Copy(cuda_affine, "affine:rnn(for)");
        aux_step["recur"] = CudaConn::Copy(cuda_rec, "recurrent:rnn(for)");

        if (m_engine.m_is_training) m_rnn_caux.push_back(aux_step);
    }

    Array<float> output;

    if (!m_outseq) {
        float* cuda_output = cuda.alloc_float_mem(rshape, "output:rnn(for)");
        cu_call(ker_rnn_select_last_vecs, rsize, (rsize, cuda_output, cuda_outputs, m_timesteps, m_recur_size));
        output = cuda.detach(cuda_output, "output:rnn(for)");
    }
    else {
        output = cuda.detach(cuda_outputs, "outputs:rnn(for)");
    }

    if (bCrossTesting) output.print("cuda forward output");

    return output;
}

Array<float> RnnLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    if (bCrossTesting) m_backprop_farr(G_hidden.to_host());

    if (bCrossTesting) 
        G_hidden.print("cuda backprop input");

    CudaConn cuda("backprop", this);

    //int64 mb_size = G_hidden.axis_size(0);

    if (bRnnTrace) G_hidden.print("G_hidden");

    float* cuda_gh = cuda.copy_to_buffer(G_hidden, "gy:rnn(back)");

    Array<float> weight = m_fetch_weight(m_param);
    Array<float> bias = m_fetch_bias(m_param);

    int64 ncol = weight.axis_size(-1);
    int64 ex_inp_size = m_timefeats + m_recur_size;

    Shape rshape = m_aux["rshape"];
    Shape xshape = m_aux["xshape"];
    Shape eshape = rshape.replace_end(ex_inp_size);
    Shape ashape = rshape.replace_end(ncol);

    int64 mb_size = ashape.total_size() / ncol;

    //Shape old_rshape(mb_size, m_recur_size);
    //Shape old_ashape(mb_size, ncol);
    //Shape old_eshape(mb_size, ex_inp_size);
    //Shape old_xshape = m_inseq ? Shape(mb_size, m_timesteps, m_timefeats) : Shape(mb_size, m_timefeats);

    Shape wshape = weight.shape();
    Shape bshape = bias.shape();

    int64 rsize = rshape.total_size();
    int64 asize = ashape.total_size();
    int64 esize = eshape.total_size();
    int64 wsize = wshape.total_size();
    int64 bsize = bshape.total_size();

    float* cuda_w = CudaConn::GetCudaMem(weight, "w:rnn(back)");

    float* cuda_gw = cuda.alloc_float_mem(wshape, "gw:rnn(back)");
    float* cuda_gb = cuda.alloc_float_mem(bshape, "gb:rnn(back)");

    float* cuda_gw_slice = cuda.alloc_float_mem(wshape, "gw:rnn(back)");
    float* cuda_gb_slice = cuda.alloc_float_mem(bshape, "gb:rnn(back)");

    float* cuda_grec = cuda.alloc_float_mem(rshape, "gr:rnn(back)");
    float* cuda_gstate = cuda.alloc_float_mem(rshape, "gs:rnn(back)");
    float* cuda_gx = cuda.alloc_float_mem(xshape, "gx:rnn(back)");
    float* cuda_gaffine = cuda.alloc_float_mem(ashape, "gaffine:rnn(back)");
    float* cuda_gex_inp = cuda.alloc_float_mem(eshape, "gex_inp:rnn(back)");

    float* cuda_gfgate = NULL;
    float* cuda_gigate = NULL;
    float* cuda_gogate = NULL;
    float* cuda_gblock = NULL;

    if (m_lstm) {
        cuda_gfgate = cuda.alloc_float_mem(rshape, "forget_gate:rnn(for)");
        cuda_gigate = cuda.alloc_float_mem(rshape, "input_gate:rnn(for)");
        cuda_gogate = cuda.alloc_float_mem(rshape, "output_gate:rnn(for)");
        cuda_gblock = cuda.alloc_float_mem(rshape, "input_block:rnn(for)");
    }


    if (!m_outseq) {
        cu_call(ker_rnn_copy_last_grad, rsize, (rsize, cuda_grec, cuda_gh, m_timesteps, m_recur_size));
    }

    for (int tn = m_timesteps - 1; tn >= 0; tn--) {
        CudaConn cuda_step("rnn-step", this);

        Dict aux_step = m_rnn_caux.back();
        m_rnn_caux.pop_back();

        if (m_outseq) {
            cu_call(ker_rnn_add_time_slice, rsize, (rsize, (m_lstm && m_use_state) ? cuda_gstate : cuda_grec, cuda_gh, m_timesteps, m_recur_size, tn));
            if (bRnnTrace) CudaConn::DumpArr(cuda_grec, "cuda_grec");
            if (bRnnTrace) CudaConn::DumpArr(cuda_gstate, "cuda_gstate");
        }

        if (m_lstm) {
            float* cuda_fgate = cuda_step.attach(aux_step["forget_gate"], "fg:rnn(back)");
            float* cuda_igate = cuda_step.attach(aux_step["input_gate"], "ig:rnn(back)");
            float* cuda_ogate = cuda_step.attach(aux_step["output_gate"], "og:rnn(back)");
            float* cuda_block = cuda_step.attach(aux_step["block_input"], "ib:rnn(back)");

            if (bRnnTrace) CudaConn::DumpArr(cuda_fgate, "cuda_fgate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_igate, "cuda_igate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_ogate, "cuda_ogate");
            if (bRnnTrace) CudaConn::DumpArr(cuda_block, "cuda_block");

            //float* cuda_rec = cuda_step.attach(aux_step["recur"], "rec:rnn(back)");
            float* cuda_recur_pre = cuda_step.attach(aux_step["recur_pre"], "rec:rnn(back)");
            float* cuda_pre_state = cuda_step.attach(aux_step["pre_state"], "state:rnn(back)");

            if (bRnnTrace) CudaConn::DumpArr(cuda_recur_pre, "cuda_recur_pre");
            if (bRnnTrace) CudaConn::DumpArr(cuda_pre_state, "cuda_pre_state");

            cu_call(ker_mult_to, rsize, (rsize, cuda_gogate, cuda_grec, cuda_recur_pre));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gogate, "cuda_gogate");
            cu_call(ker_lstm_state_derv, rsize, (rsize, cuda_gstate, cuda_grec, cuda_ogate, cuda_recur_pre)); // G_recur_tmp = G_recurrent * output_gate; G_state += kmath->tanh_derv(recur_tmp) * G_recur_tmp;
            if (bRnnTrace) CudaConn::DumpArr(cuda_gstate, "cuda_gstate");

            cu_call(ker_mult_to, rsize, (rsize, cuda_gigate, cuda_gstate, cuda_block));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gigate, "cuda_gigate");
            cu_call(ker_mult_to, rsize, (rsize, cuda_gblock, cuda_gstate, cuda_igate));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gblock, "cuda_gblock");

            cu_call(ker_mult_to, rsize, (rsize, cuda_gfgate, cuda_gstate, cuda_pre_state));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gfgate, "cuda_gfgate");
            cu_call(ker_mult_on, rsize, (rsize, cuda_gstate, cuda_fgate));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gstate, "cuda_gstate");

            cu_call(ker_sigmoid_derv_on, rsize, (rsize, cuda_gfgate, cuda_fgate));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gfgate, "cuda_gfgate");
            cu_call(ker_sigmoid_derv_on, rsize, (rsize, cuda_gigate, cuda_igate));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gigate, "cuda_gigate");
            cu_call(ker_sigmoid_derv_on, rsize, (rsize, cuda_gogate, cuda_ogate));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gogate, "cuda_gogate");
            cu_call(ker_tanh_derv_on, rsize, (rsize, cuda_gblock, cuda_block));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gblock, "cuda_gblock");

            cu_call(ker_lstm_combine_affine, asize, (asize, cuda_gaffine, cuda_gfgate, cuda_gigate, cuda_gogate, cuda_gblock, m_recur_size));
            if (bRnnTrace) CudaConn::DumpArr(cuda_gaffine, "cuda_gaffine");
        }
        else {
            float* cuda_rec = cuda_step.attach(aux_step["recur"], "rec:rnn(back)");
            float* cuda_affine = cuda_step.attach(aux_step["affine"], "affine:rnn(back)");

            cu_call(ker_activate_derv, rsize, (rsize, cuda_gaffine, cuda_grec, cuda_affine, cuda_rec, m_nActFunc, m_leaky_alpha));
        }

        float* cuda_ex_inp = cuda_step.attach(aux_step["ex_inp"], "ex_inp:rnn(back)");
        if (bRnnTrace) CudaConn::DumpArr(cuda_ex_inp, "cuda_ex_inp");

        cu_call(ker_matmul_derv_x, esize, (esize, cuda_gex_inp, cuda_gaffine, cuda_w, mb_size, ex_inp_size, ncol));
        if (bRnnTrace) CudaConn::DumpArr(cuda_gex_inp, "cuda_gex_inp");
        cu_call(ker_matmul_derv_w, wsize, (wsize, cuda_gw_slice, cuda_gaffine, cuda_ex_inp, mb_size, ex_inp_size, ncol));
        if (bRnnTrace) CudaConn::DumpArr(cuda_gw_slice, "cuda_gw_slice");
        cu_call(ker_add_bias_derv, bsize, (bsize, cuda_gb_slice, cuda_gaffine, mb_size, ex_inp_size, ncol));
        if (bRnnTrace) CudaConn::DumpArr(cuda_gb_slice, "cuda_gb_slice");

        cu_call(ker_add_on, wsize, (wsize, cuda_gw, cuda_gw_slice));
        if (bRnnTrace) CudaConn::DumpArr(cuda_gw, "cuda_gw");
        cu_call(ker_add_on, bsize, (bsize, cuda_gb, cuda_gb_slice));
        if (bRnnTrace) CudaConn::DumpArr(cuda_gb, "cuda_gb");

        cu_call(ker_rnn_split_ex_inp, esize, (esize, cuda_gx, cuda_grec, cuda_gex_inp, m_timesteps, m_timefeats, m_recur_size, m_inseq, tn));
        if (bRnnTrace) CudaConn::DumpArr(cuda_gx, "cuda_gx");
        if (bRnnTrace) CudaConn::DumpArr(cuda_grec, "cuda_grec");
    }

    Array<float> G_weight = cuda.detach(cuda_gw);
    Array<float> G_bias = cuda.detach(cuda_gb);

    if (bCrossTesting) {
        G_weight.print("cuda backprop G_weight");
        G_bias.print("cuda backprop G_bias");
    }

    m_update_weight(m_param, G_weight);
    m_update_bias(m_param, G_bias);
    //cuda.optimize(m_pm, "w", m_engine, cuda_gw);
    //cuda.optimize(m_pm, "b", m_engine, cuda_gb);

    if (bCrossTesting) {
        Array<float> weight = m_fetch_weight(m_param);
        Array<float> bias = m_fetch_bias(m_param);

        weight.print("cuda backprop weight");
        bias.print("cuda backprop bias");
    }

    assert(m_rnn_caux.size() == 0);

    Array<float> G_input = cuda.detach(cuda_gx, "G_input:rnn(back)");

    if (bCrossTesting) G_input.print("cuda backprop output");
    if (bCrossTesting) throw KaiException(KERR_ASSERT);

    return G_input;
}
