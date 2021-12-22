/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "adam_optimizer.cuh"


//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif

AdamOptimizer::AdamOptimizer(Dict kwArgs) : Optimizer("adam", kwArgs) {
	m_ro1 = (float)Value::seek_option(kwArgs, "adam_ro1", 0.9f);
	m_ro2 = (float)Value::seek_option(kwArgs, "adam_ro2", 0.999f);
	m_epsilon = (float)Value::seek_option(kwArgs, "adam_epsilon", 1.0e-8f);
}

AdamOptimizer::~AdamOptimizer() {
}

void AdamOptimizer::setup(Dict kwArgs) {
	m_ro1 = (float)Value::seek_option(kwArgs, "adam_ro1", m_ro1);
	m_ro2 = (float)Value::seek_option(kwArgs, "adam_ro2", m_ro2);
	m_epsilon = (float)Value::seek_option(kwArgs, "adam_epsilon", m_epsilon);
}

string AdamOptimizer::introduce_extra() {
	char buffer[128];
	sprintf_s(buffer, 128,", ro1:%f, ro2=%f, epsilon:%f", m_ro1, m_ro2, m_epsilon);
	return  (string)buffer;
}

void AdamOptimizer::m_alloc_affine_param(Dict& param, Shape shape, bool use_cuda, Dict kwArgs) {
	Dict pm_w = param["w"];

	pm_w["s"] = kmath->zeros(shape);
	pm_w["t"] = kmath->zeros(shape);
	pm_w["n"] = 0.0f;

	param["w"] = pm_w;

	if (use_cuda) {
		Shape bias_shape = shape[-1];
		Dict pm_b = param["b"];

		pm_b["s"] = kmath->zeros(bias_shape);
		pm_b["t"] = kmath->zeros(bias_shape);
		pm_b["n"] = 0.0f;

		param["b"] = pm_b;
	}
}

void AdamOptimizer::m_alloc_embed_param(Dict& param, vector<int64> voc_sizes, int64 vec_size, Dict kwArgs) {
	Array<float> dic_pm = param["_pm_"];

	Shape shape = dic_pm.shape();

	param["s"] = kmath->zeros(shape);
	param["t"] = kmath->zeros(shape);
	param["n"] = kmath->zeros(Shape(shape[0]));
}

Array<float> AdamOptimizer::m_eval_adam_delta(Dict param, Array<float> G_param, float n, int64 row) {
	Array<float> pm_s = param["s"];
	Array<float> pm_t = param["t"];

	Array<float> s = pm_s;
	Array<float> t = pm_t;

	if (row != -1) {
		s = pm_s.get_row(row);
		t = pm_t.get_row(row);
	}

	s = s * m_ro1 + G_param * (1 - m_ro1);
	t = t * m_ro2 + (G_param * G_param) * (1 - m_ro2);

	if (row != -1) {
		pm_s.set_row(row, s);
		pm_t.set_row(row, t);
	}
	else {
		param["s"] = s;
		param["t"] = t;
	}

	s = s / (1.0f - kmath->power(m_ro1, n));
	t = t / (1.0f - kmath->power(m_ro2, n));

	return s / (kmath->sqrt(t) + m_epsilon);
}

void AdamOptimizer::update_weight(Dict pm_w, Array<float> G_weight) {
	Array<float> weight = pm_w["_pm_"];
	float n = (float)pm_w["n"] + 1;
	pm_w["n"] = n;

	G_weight = G_weight + m_eval_adam_delta(pm_w, G_weight, n) + m_eval_decay_delta(weight);

	weight -= G_weight * m_learning_rate;
}

void AdamOptimizer::update_bias(Dict pm_b, Array<float> G_bias) {
	Array<float> bias = pm_b["_pm_"];

	float n = (float)pm_b["n"] + 1;
	pm_b["n"] = n;

	G_bias = G_bias + m_eval_adam_delta(pm_b, G_bias, n);

	bias -= G_bias * m_learning_rate;
}

void AdamOptimizer::update_embed(Dict param, Array<float> G_words, Array<int64> selector) {
	Array<float> pm_dic = param["_pm_"];
	Array<int64> voc_start = param["voc_start"];

	int64 dic_cnt = voc_start.total_size();
	int64 vec_size = pm_dic.axis_size(1);

	Array<float> G_words_flat = G_words.reshape(Shape(-1, vec_size)); // G_words는 mb_size를 별도의 축으로 하는 등 다양한 형상을 가질 수 있다

	Array<int64> flat_selector = selector.reshape(Shape(-1, dic_cnt)); // selector 역시 mb_size를 별도의 축으로 하는 등 다양한 형상을 가질 수 있다
	Array<int64> mark = Array<int64>::zeros(flat_selector.shape());

	int64 word_cnt = flat_selector.axis_size(0);

	assert(G_words_flat.axis_size(0) == word_cnt);

	Array<float> narr = param["n"];

	for (int64 nw = 0; nw < word_cnt; nw++) {
		for (int64 nd = 0; nd < dic_cnt; nd++) {
			if (mark[Idx(nw, nd)]) continue;

			int64 wid = flat_selector[Idx(nw, nd)];
			int64 dpos = wid + voc_start[Idx(nd)];

			Array<float> word_vec = pm_dic.get_row(dpos);
			Array<float> G_wvec = G_words_flat.get_row(nw);

			for (int64 m = nw + 1; m < word_cnt; m++) {
				if (flat_selector[Idx(m, nd)] == wid) {
					mark[Idx(m, nd)] = 1;
					G_wvec += G_words_flat.get_row(m);
				}
			}

			float n = narr[Idx(dpos)] + 1;
			narr[Idx(dpos)] = n;

			G_wvec += m_eval_adam_delta(param, G_wvec, n, dpos) + m_eval_decay_delta(word_vec);

			pm_dic.sub_row(dpos, G_wvec * m_learning_rate);
		}
	}
}

__global__ void kernel_adam_update_weight(int64 size, float* w_inout, float* g_in, float* s_inout, float* t_inout, float nth, float learning_rate, float l2_decay, float l1_decay, float ro1, float ro2, float epsilon);
__global__ void kernel_adam_update_bias(int64 size, float* b_inout, float* g_in, float* s_inout, float* t_inout, float nth, float learning_rate, float ro1, float ro2, float epsilon);
__global__ void kernel_adam_update_embed_step1(int64 size, float* h_out, int64* m_out, float* g_in, int64* w_in, int64* v_in, float* n_inout, int64 word_cnt, int64 vec_size, int64 dic_cnt);
__global__ void kernel_adam_update_embed_step2(int64 size, float* d_inout, float* h_in, int64* m_in, float* s_inout, float* t_inout, float* n_in, int64 vec_size, int64 dic_cnt, float learning_rate, float l2_decay, float l1_decay, float ro1, float ro2, float epsilon);

void AdamOptimizer::update_weight_cuda(Dict param, Array<float> G_weight) {
	Array<float> weight = param["_pm_"];

	float nth = (float)param["n"] + 1;
	param["n"] = nth;

	Array<float> pm_s = param["s"];
	Array<float> pm_t = param["t"];

	int64 wsize = weight.total_size();

	float* cuda_w = weight.data_ptr();
	float* cuda_s = pm_s.data_ptr();
	float* cuda_t = pm_t.data_ptr();
	float* cuda_g = G_weight.data_ptr();

	cu_call_no_lock(kernel_adam_update_weight, wsize, (wsize, cuda_w, cuda_g, cuda_s, cuda_t, nth, m_learning_rate, m_l2_decay, m_l1_decay, m_ro1, m_ro2, m_epsilon));
}

void AdamOptimizer::update_bias_cuda(Dict param, Array<float> G_bias) {
	Array<float> bias = param["_pm_"];

	float nth = (float)param["n"] + 1;
	param["n"] = nth;

	Array<float> pm_s = param["s"];
	Array<float> pm_t = param["t"];

	int64 bsize = bias.total_size();

	float* cuda_b = bias.data_ptr();
	float* cuda_s = pm_s.data_ptr();
	float* cuda_t = pm_t.data_ptr();
	float* cuda_g = G_bias.data_ptr();

	cu_call_no_lock(kernel_adam_update_bias, bsize, (bsize, cuda_b, cuda_g, cuda_s, cuda_t, nth, m_learning_rate, m_ro1, m_ro2, m_epsilon));
}

void AdamOptimizer::update_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector) {
	CudaConn conn("AdamOptimizer::update_embed_cuda", NULL);

	Array<float> pm_mdic = param["_pm_"];
	Array<int64> voc_start = param["voc_start"];
	Array<float> pm_n = param["n"];
	Array<float> pm_s = param["s"];
	Array<float> pm_t = param["t"];

	int64 dic_cnt = voc_start.axis_size(0);
	int64 dic_size = pm_mdic.axis_size(0);
	int64 word_cnt = selector.total_size() / dic_cnt;
	int64 vec_size = pm_mdic.axis_size(1);

	float* cuda_g = G_words.data_ptr();
	float* cuda_d = pm_mdic.data_ptr();
	float* cuda_s = pm_s.data_ptr();
	float* cuda_t = pm_t.data_ptr();
	float* cuda_n = pm_n.data_ptr();

	int64* cuda_w = selector.data_ptr();
	int64* cuda_v = voc_start.data_ptr();

	int64 gsize = word_cnt * vec_size * dic_cnt;

	int64* cuda_m = conn.alloc_int64_mem(Shape(word_cnt, dic_cnt), "mark_to_work");
	float* cuda_h = conn.alloc_float_mem(Shape(word_cnt, vec_size, dic_cnt), "grad_sum_to_apply"); // 여러번 출현하는 단어에 대해 첫 단어 위치에 경사도 합을 집계

	//m_need_debugging_here();

	cu_call_no_lock(kernel_adam_update_embed_step1, gsize, (gsize, cuda_h, cuda_m, cuda_g, cuda_w, cuda_v, cuda_n, word_cnt, vec_size, dic_cnt));
	cu_call_no_lock(kernel_adam_update_embed_step2, gsize, (gsize, cuda_d, cuda_h, cuda_m, cuda_s, cuda_t, cuda_n, vec_size, dic_cnt, m_learning_rate, m_l2_decay, m_l1_decay, m_ro1, m_ro2, m_epsilon));
}

__device__ float dev_adam_eval_decay(float pm, float l2_decay, float l1_decay) {
	float delta = 0;
	if (l2_decay > 0) delta += pm * l2_decay;
	if (l1_decay > 0) {
		if (pm > 0) delta += l1_decay;
		else if (pm < 0) delta -= l1_decay;
	}
	return delta;
}

__device__ float dev_adam_eval_grad(float grad, float* s, float* t, int64 idx, float nth, float ro1, float ro2, float epsilon) {
	s[idx] = s[idx] * ro1 + (1.0f - ro1) * grad;
	t[idx] = t[idx] * ro2 + (1.0f - ro2) * grad * grad;

	float sm = s[idx] / (1.0f - __powf(ro1, nth));
	float tm = t[idx] / (1.0f - __powf(ro2, nth));

	return sm / (__fsqrt_rn(tm) + epsilon);
}

__global__ void kernel_adam_update_weight(int64 size, float* w_inout, float* g_in, float* s_inout, float* t_inout, float nth, float learning_rate, float l2_decay, float l1_decay, float ro1, float ro2, float epsilon) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float grad = g_in[idx];

		grad = dev_adam_eval_grad(grad, s_inout, t_inout, idx, nth, ro1, ro2, epsilon);
		grad += dev_adam_eval_decay(w_inout[idx], l2_decay, l1_decay);

		w_inout[idx] -= grad * learning_rate;
	}
}

__global__ void kernel_adam_update_bias(int64 size, float* b_inout, float* g_in, float* s_inout, float* t_inout, float nth, float learning_rate, float ro1, float ro2, float epsilon) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float grad = g_in[idx];

		grad = dev_adam_eval_grad(grad, s_inout, t_inout, idx, nth, ro1, ro2, epsilon);

		b_inout[idx] -= grad * learning_rate;
	}
}

__global__ void kernel_adam_update_embed_step1(int64 size, float* h_out, int64* m_out, float* g_in, int64* w_in, int64* v_in, float* n_inout, int64 word_cnt, int64 vec_size, int64 dic_cnt) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		int64 nw = idx / (vec_size * dic_cnt);
		int64 nv = idx / dic_cnt % vec_size;
		int64 nd = idx % dic_cnt;

		int64 wid = w_in[nw * dic_cnt + nd];

		float grad = 0;

		for (int64 n = 0; n < word_cnt; n++) {
			if (w_in[n * dic_cnt + nd] != wid) continue;
			if (n < nw) return;
			grad += g_in[n * vec_size + nv];
		}

		if (nv == 0) {
			int64 dwid = wid + v_in[nd];
			n_inout[dwid] = n_inout[dwid] + 1.0f;
			m_out[nw * dic_cnt + nd] = dwid + 1;
		}

		h_out[idx] = grad;
	}
}

__global__ void kernel_adam_update_embed_step2(int64 size, float* d_inout, float* h_in, int64* m_in, float* s_inout, float* t_inout, float* n_in, int64 vec_size, int64 dic_cnt, float learning_rate, float l2_decay, float l1_decay, float ro1, float ro2, float epsilon) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		int64 nw = idx / (vec_size * dic_cnt);
		int64 nv = idx / dic_cnt % vec_size;
		int64 nd = idx % dic_cnt;

		int64 dwid = m_in[nw * dic_cnt + nd] - 1;

		if (dwid < 0) return;

		float grad = h_in[idx];
		int64 dpos = dwid * vec_size + nv;

		float nth = n_in[dwid];

		grad = dev_adam_eval_grad(grad, s_inout, t_inout, dpos, nth, ro1, ro2, epsilon);
		grad += dev_adam_eval_decay(d_inout[dpos], l2_decay, l1_decay);

		d_inout[dpos] -= grad * learning_rate;
	}
}
