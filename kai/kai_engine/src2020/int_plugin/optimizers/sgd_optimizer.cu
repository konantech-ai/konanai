/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "sgd_optimizer.cuh"

SgdOptimizer::SgdOptimizer(Dict kwArgs) : Optimizer("sgd", kwArgs) {
}

SgdOptimizer::~SgdOptimizer() {
}

void SgdOptimizer::setup(Dict kwArgs) {
}

void SgdOptimizer::update_weight(Dict pm_w, Array<float> G_weight) {
	Array<float> weight = pm_w["_pm_"];
	G_weight += m_eval_decay_delta(weight);
	weight -= G_weight * m_learning_rate;
}

void SgdOptimizer::update_bias(Dict pm_b, Array<float> G_bias) {
	Array<float> bias = pm_b["_pm_"];
	bias -= G_bias * m_learning_rate;
}

void SgdOptimizer::update_embed(Dict param, Array<float> G_words, Array<int64> selector) {
	Array<float> pm_dic = param["_pm_"];
	Array<int64> voc_start = param["voc_start"];

	int64 dic_cnt = voc_start.total_size();
	int64 vec_size = pm_dic.axis_size(1);

	Array<float> G_words_flat = G_words.reshape(Shape(-1, vec_size)); // G_words는 mb_size를 별도의 축으로 하는 등 다양한 형상을 가질 수 있다

	Array<int64> flat_selector = selector.reshape(Shape(-1, dic_cnt)); // selector 역시 mb_size를 별도의 축으로 하는 등 다양한 형상을 가질 수 있다
	Array<int64> mark = Array<int64>::zeros(flat_selector.shape());

	int64 word_cnt = flat_selector.axis_size(0);

	assert(G_words_flat.axis_size(0) == word_cnt);

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

			G_wvec += m_eval_decay_delta(word_vec);

			pm_dic.sub_row(dpos, G_wvec * m_learning_rate);
		}
	}
}

__global__ void kernel_sgd_update_weight(int64 size, float* p_inout, float* g_in, float learning_rate, float l2_decay, float l1_decay);
__global__ void kernel_sgd_update_bias(int64 size, float* b_inout, float* g_in, float learning_rate);
__global__ void kernel_sgd_update_embed(int64 size, float* d_inout, float* g_in, int64* w_in, int64* v_in, int64 dic_cnt, int64 word_cnt, int64 vec_size, float learning_rate, float l2_decay, float l1_decay);

void SgdOptimizer::update_weight_cuda(Dict param, Array<float> G_weight) {
	Array<float> weight = param["_pm_"];
	int64 wsize = weight.total_size();

	float* cuda_w = weight.data_ptr();
	float* cuda_g = G_weight.data_ptr();

	cu_call_no_lock(kernel_sgd_update_weight, wsize, (wsize, cuda_w, cuda_g, m_learning_rate, m_l2_decay, m_l1_decay));
}

void SgdOptimizer::update_bias_cuda(Dict param, Array<float> G_bias) {
	Array<float> bias = param["_pm_"];
	int64 bsize = bias.total_size();

	float* cuda_b = bias.data_ptr();
	float* cuda_g = G_bias.data_ptr();

	cu_call_no_lock(kernel_sgd_update_bias, bsize, (bsize, cuda_b, cuda_g, m_learning_rate));
}

void SgdOptimizer::update_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector) {
	Array<float> pm_dic = param["_pm_"];
	Array<int64> voc_start = param["voc_start"];

	float* cuda_g = G_words.data_ptr();
	float* cuda_d = pm_dic.data_ptr();
	int64* cuda_s = selector.data_ptr();
	int64* cuda_v = voc_start.data_ptr();

	int64 dic_cnt = selector.axis_size(-1);
	int64 word_cnt = selector.total_size() / dic_cnt;
	int64 vec_size = pm_dic.axis_size(1);
	int64 psize = dic_cnt * word_cnt * vec_size;

	cu_call_no_lock(kernel_sgd_update_embed, psize, (psize, cuda_d, cuda_g, cuda_s, cuda_v, dic_cnt, word_cnt, vec_size, m_learning_rate, m_l2_decay, m_l1_decay));
}

__device__ float dev_sgd_eval_decay(float pm, float l2_decay, float l1_decay) {
	float delta = 0;
	if (l2_decay > 0) delta += pm * l2_decay;
	if (l1_decay > 0) {
		if (pm > 0) delta += l1_decay;
		else if (pm < 0) delta -= l1_decay;
	}
	return delta;
}

__global__ void kernel_sgd_update_weight(int64 size, float* w_inout, float* g_in, float learning_rate, float l2_decay, float l1_decay) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float grad = g_in[idx] + dev_sgd_eval_decay(w_inout[idx], l2_decay, l1_decay);
		w_inout[idx] -= grad * learning_rate;
	}
}

__global__ void kernel_sgd_update_bias(int64 size, float* b_inout, float* g_in, float learning_rate) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		float grad = g_in[idx];
		b_inout[idx] -= grad * learning_rate;
	}
}

__global__ void kernel_sgd_update_embed(int64 size, float* d_inout, float* g_in, int64* w_in, int64* v_in, int64 dic_cnt, int64 word_cnt, int64 vec_size, float learning_rate, float l2_decay, float l1_decay) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		int64 nd = idx / (word_cnt * vec_size);
		int64 nw = idx / vec_size % word_cnt;
		int64 nv = idx % vec_size;

		int64 wid = w_in[nw * dic_cnt + nd];

		float grad = 0;

		for (int64 n = 0; n < word_cnt; n++) {
			if (w_in[n * dic_cnt + nd] != wid) continue;
			if (n < nw) return;
			grad += g_in[n * vec_size + nv];
		}

		int64 dpos = (wid + v_in[nd]) * vec_size + nv;

		grad += dev_sgd_eval_decay(d_inout[dpos], l2_decay, l1_decay);

		d_inout[dpos] -= grad * learning_rate;
	}
}
