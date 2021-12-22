/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "optimizer.cuh"

#include "optimizers/sgd_optimizer.cuh"
#include "optimizers/adam_optimizer.cuh"

#include "../core/engine.h"


//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif

Optimizer* Optimizer::ms_curr_instance = NULL;

Optimizer* Optimizer::create_instance(string component_name, Dict kwArgs) {
	Optimizer* new_instance = NULL;

	if (component_name == "sgd") new_instance = new SgdOptimizer(kwArgs);
	else if (component_name == "adam") new_instance = new AdamOptimizer(kwArgs);

	return new_instance;
}

Optimizer::Optimizer(string name, Dict kwArgs) {
	m_ref_count = 1;

	m_name = name;

	m_learning_rate = (float)Value::seek_option(kwArgs, "learning_rate", 0.001f);
	m_l2_decay = (float)Value::seek_option(kwArgs, "l2_decay", 0.0f);
	m_l1_decay = (float)Value::seek_option(kwArgs, "l1_decay", 0.0f);
}

Optimizer::~Optimizer() {
}

void Optimizer::setup(Dict kwArgs) {
	m_learning_rate = (float)Value::seek_option(kwArgs, "learning_rate", m_learning_rate);
	m_l2_decay = (float)Value::seek_option(kwArgs, "l2_decay", m_l2_decay);
	m_l1_decay = (float)Value::seek_option(kwArgs, "l1_decay", m_l1_decay);
}

Optimizer* Optimizer::check_in_curr_instance(Dict kwArgs) {
	if (ms_curr_instance != NULL) {
		ms_curr_instance->m_ref_count++;
		ms_curr_instance->setup(kwArgs);
		return ms_curr_instance;
	}
	else {
		return NULL;
	}
}

Optimizer* Optimizer::check_in_named_instance(string component_name, Dict kwArgs) {
	if (ms_curr_instance->m_name == component_name) {
		ms_curr_instance->m_ref_count++;
		ms_curr_instance->setup(kwArgs);
		return ms_curr_instance;
	}
	else {
		Optimizer* new_instance = create_instance(component_name, kwArgs);
		return new_instance;
	}
}

void Optimizer::check_out() {
	if (this && --m_ref_count <= 0) delete this;
}

string Optimizer::introduce_curr_instance() {
	if (ms_curr_instance != NULL) {
		return ms_curr_instance->introduce();
	}

	throw KaiException(KERR_NO_OPTIMZER_SET_DEFAULT_ADAM_IN_USING);
}

void Optimizer::set_plugin_component(string component_name) {
	Dict kwArgs = Engine::get_default_options();

	Optimizer* new_instance = create_instance(component_name, kwArgs);

	if (new_instance == NULL) throw KaiException(KERR_UNKNOWN_OPTIMIZER_NAME);

	if (ms_curr_instance) ms_curr_instance->check_out();
	ms_curr_instance = new_instance;
}

Optimizer* Optimizer::get_curr_instance() {
	return ms_curr_instance;
}

string Optimizer::introduce() {
	string extra = introduce_extra();

	char buffer[128];

	string l2_extra;
	string l1_extra;

	if (m_l2_decay > 0) {
		//hs.cho
		sprintf_s(buffer,128, ", l2_decay:%f", m_l2_decay);
		l2_extra = (string)buffer;
	}

	if (m_l1_decay > 0) {
		//hs.cho
		sprintf_s(buffer, 128,", l1_decay:%f", m_l1_decay);
		l1_extra = (string)buffer;
	}
	//hs.cho
	sprintf_s(buffer,128, "%s optimizer(learning_rate:%f%s%s%s)", m_name.c_str(), m_learning_rate, l2_extra.c_str(), l1_extra.c_str(), extra.c_str());

	return  (string)buffer;
}

string Optimizer::introduce_extra() {
	return "";
}

Array<float> Optimizer::fetch_core(Dict param) {
	return param["_pm_"];
}

string Optimizer::get_affine_param_desc(Dict param) {
	Dict pm_w = param["w"];
	Array<float> weight = pm_w["_pm_"];
	string shape_str;
	string delimeter = "";
	Shape wshape = weight.shape();
	for (int64 n = 0; n < wshape.size(); n++) {
		int64 size = wshape[n];
		shape_str += delimeter + std::to_string(size);
		delimeter = "*";
	}
	if (param.find("b") != param.end()) {
		shape_str += "+" + std::to_string(wshape[-1]);
	}
	return shape_str;
}

string Optimizer::get_embed_param_desc(Dict param) {
	Array<float> dic = param["_pm_"];
	Array<int64> voc_start = param["voc_start"];

	Shape dshape = dic.shape();

	string shape_str;

	int64 voc_count = voc_start.axis_size(0);
	if (voc_count == 1) shape_str = std::to_string(dshape[0]);
	else {
		string delimeter = "(";
		for (int64 n = 0; n < voc_count - 1; n++) {
			shape_str += std::to_string(voc_start[Idx(n + 1)] - voc_start[Idx(n)]) + "+";
		}
		shape_str += std::to_string(dshape[0] - voc_start[Idx(voc_count - 1)]) + ")";
	}

	shape_str += "*" + std::to_string(dshape[1]);

	return shape_str;
}

Dict Optimizer::alloc_affine_param(Shape shape, bool use_bias, param_init init_type, Dict kwArgs) {
	Dict param, pm_w, pm_b;

	pm_w["_pm_"] = m_init_weight(shape, init_type, kwArgs);
	param["w"] = pm_w;

	if (use_bias) {
		pm_b["_pm_"] = kmath->zeros(shape[-1]);
		param["b"] = pm_b;
	}

	m_alloc_affine_param(param, shape, use_bias, kwArgs);

	return param;
}

Dict Optimizer::alloc_embed_param(vector<int64> voc_sizes, int64 vec_size, Dict kwArgs) {
	Dict param;

	Array<int64> voc_start(voc_sizes.size());

	int64 acc_size = 0;

	for (int64 n = 0; n < (int64) voc_sizes.size(); n++) {
		voc_start[Idx(n)] = acc_size;
		acc_size += voc_sizes[n];
	}

	if (CudaConn::UsingCuda()) voc_start = CudaConn::ToCudaArray(voc_start);

	param["_pm_"] = m_init_weight(Shape(acc_size, vec_size), param_init::gauss, kwArgs);
	param["voc_start"] = voc_start;

	m_alloc_embed_param(param, voc_sizes, vec_size, kwArgs);

	return param;
}

Array<float> Optimizer::forward_affine(Dict param, Array<float> x) {
	bool bCrossTesting = CudaConn::UsingCuda() && !x.is_cuda();
	Dict pm_w = param["w"];
	Array<float> w = pm_w["_pm_"];
	if (bCrossTesting) w = w.to_host(); // 비교 실험 등의 경우
	//x.prod_check();
	//w.prod_check();
	Array<float> affine = bCrossTesting ? hmath.matmul(x, w) : kmath->matmul(x, w);
	affine.prod_check();
	if (param.find("b") != param.end()) {
		Dict pm_b = param["b"];
		Array<float> b = pm_b["_pm_"];
		if (bCrossTesting) b = b.to_host(); // 비교 실험 등의 경우
		b.prod_check();
		affine = affine + b;
		affine.prod_check();
	}
	//affine.prod_check();

	m_forward_affine(param, x, affine);

	return affine;
}

Array<float> Optimizer::backprop_affine(Dict param, Array<float> x, Array<float> G_affine) {
	Dict pm_w = param["w"];
	Array<float> w = pm_w["_pm_"];
	Array<float> g_affine_weight = x.transpose();
	Array<float> g_affine_input = w.transpose();

	Array<float> G_weight = kmath->matmul(g_affine_weight, G_affine);
	Array<float> G_input = kmath->matmul(G_affine, g_affine_input);

	regist_update_weight(pm_w, G_weight);

	if (param.find("b") != param.end()) {
		Dict pm_b = param["b"];
		Array<float> b = pm_b["_pm_"];
		Array<float> G_bias = kmath->sum(G_affine, -1);
		regist_update_bias(pm_b, G_bias);
	}

	m_backprop_affine(param, x, G_affine, G_input);

	return G_input;
}

Array<float> Optimizer::forward_embed(Dict param, Array<int64> selector) {
	Array<float> dic = param["_pm_"];
	Array<int64> voc_start = param["voc_start"];

	int64 dic_cnt = voc_start.total_size();
	int64 vec_size = dic.axis_size(1);

	Array<int64> words = selector.reshape(Shape(-1, dic_cnt));
	Array<float> word_vecs(words.shape().replace_end(vec_size));

	int64 word_cnt = words.axis_size(0);

	int64* sp = words.data_ptr();
	int64* vp = voc_start.data_ptr();
	float* dp = dic.data_ptr();
	float* yp = word_vecs.data_ptr();

	for (int64 n = 0; n < word_cnt; n++) {
		for (int64 m = 0; m < dic_cnt; m++) {
			float* dpos = dp + (vp[m] + *sp++) * vec_size;;
			for (int64 k = 0; k < vec_size; k++) {
				yp[k] = dpos[k];
			}
		}
		yp += vec_size;
	}

	Array<float> output = word_vecs.reshape(selector.shape().replace_end(vec_size));

	m_forward_embed(param, selector, output);

	return output;
}

__global__ void kernel_opt_forward_embed(int64 size, float* y_out, float* d_in, int64* w_in, int64* v_in, int64 dic_cnt, int64 word_cnt, int64 vec_size) {
	int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		int64 nw = idx / vec_size % word_cnt;
		int64 nv = idx % vec_size;

		float sum = 0;

		for (int64 nd = 0; nd < dic_cnt; nd++) {
			int64 wid = w_in[nw * dic_cnt + nd];
			int64 dpos = (wid + v_in[nd]) * vec_size + nv;

			sum += d_in[dpos];
		}

		y_out[idx] = sum;
	}
}

void Optimizer::forward_embed_cuda(Dict param, Array<float> word_vecs, Array<int64> selector) {
	Array<float> pm_dic = param["_pm_"];
	Array<int64> voc_start = param["voc_start"];

	float* cuda_d = pm_dic.data_ptr();
	float* cuda_y = word_vecs.data_ptr();
	int64* cuda_s = selector.data_ptr();
	int64* cuda_v = voc_start.data_ptr();

	int64 dic_cnt = selector.axis_size(-1);
	int64 word_cnt = selector.total_size() / dic_cnt;
	int64 vec_size = pm_dic.axis_size(1);
	int64 psize = word_cnt * vec_size;

	assert(word_vecs.total_size() == psize);
	assert(voc_start.total_size() == dic_cnt);

	cu_call_no_lock(kernel_opt_forward_embed, psize, (psize, cuda_y, cuda_d, cuda_s, cuda_v, dic_cnt, word_cnt, vec_size));

	m_forward_embed_cuda(param, word_vecs, selector);
}

Array<float> Optimizer::m_init_weight(Shape shape, param_init init_type, Dict kwArgs) {
	switch (init_type) {
	case param_init::zeros:
		return kmath->zeros(shape);
	case param_init::ones:
		return kmath->ones(shape);
	case param_init::uniform:
		return kmath->random_uniform(shape);
	case param_init::gauss:
	default:
		float rand_std = (float)Value::seek_option(kwArgs, "rand_std", 0.030f);
		return kmath->random_normal(0, rand_std, shape);
	}
}

Array<float> Optimizer::m_eval_decay_delta(Array<float> pm) {
	Array<float> delta = kmath->zeros(pm.shape());

	if (m_l2_decay > 0) delta += pm * m_l2_decay;
	if (m_l1_decay > 0) delta += kmath->sign(pm) * m_l1_decay;

	return delta;
}

void Optimizer::backprop_embed(Dict param, Array<float> G_words, Array<int64> selector) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void Optimizer::backprop_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void Optimizer::regist_update_weight(Dict pm, Array<float> Grad) {
	_UpdateInfo update_info;

	update_info.m_paramType = 'w';
	update_info.m_param = pm;
	update_info.m_grad = Grad;

	m_updateInfo.push_back(update_info);
}

void Optimizer::regist_update_bias(Dict pm, Array<float> Grad) {
	_UpdateInfo update_info;

	update_info.m_paramType = 'b';
	update_info.m_param = pm;
	update_info.m_grad = Grad;

	m_updateInfo.push_back(update_info);
}

void Optimizer::regist_update_embed(Dict pm, Array<float> Grad, Array<int64> selector) {
	_UpdateInfo update_info;

	update_info.m_paramType = 'e';
	update_info.m_param = pm;
	update_info.m_grad = Grad;
	update_info.m_wids = selector;

	m_updateInfo.push_back(update_info);
}

void Optimizer::regist_update_weight_cuda(Dict pm, Array<float> Grad) {
	_UpdateInfo update_info;

	update_info.m_paramType = 'W';
	update_info.m_param = pm;
	update_info.m_grad = Grad;

	m_updateInfo.push_back(update_info);
}

void Optimizer::regist_update_bias_cuda(Dict pm, Array<float> Grad) {
	_UpdateInfo update_info;

	update_info.m_paramType = 'B';
	update_info.m_param = pm;
	update_info.m_grad = Grad;

	m_updateInfo.push_back(update_info);
}

void Optimizer::regist_update_embed_cuda(Dict pm, Array<float> Grad, Array<int64> selector) {
	_UpdateInfo update_info;

	update_info.m_paramType = 'E';
	update_info.m_param = pm;
	update_info.m_grad = Grad;
	update_info.m_wids = selector;

	m_updateInfo.push_back(update_info);
}

float Optimizer::flush_update(bool trace_grad_norm, float clip_grad_threshold) {
	float grad_norm = 0;

	if (trace_grad_norm || clip_grad_threshold > 0) {
		for (auto it = m_updateInfo.begin(); it != m_updateInfo.end(); it++) {
			_UpdateInfo update_info = *it;
			Array<float> pm = update_info.m_grad;
			grad_norm += kmath->square_sum(pm);
		}
	}

	float clip_ratio = 1.0f;

	if (clip_grad_threshold > 0 && grad_norm > clip_grad_threshold) clip_ratio = ::sqrt(clip_grad_threshold / grad_norm);

	for (auto it = m_updateInfo.begin(); it != m_updateInfo.end(); it++) {
		_UpdateInfo update_info = *it;
		Array<float> pm = update_info.m_grad;
		
		if (clip_ratio != 1.0f) {
			pm = pm * clip_ratio;
		}

		switch (update_info.m_paramType) {
		case 'w':
			update_weight(update_info.m_param, pm);
			break;
		case 'b':
			update_bias(update_info.m_param, pm);
			break;
		case 'e':
			update_embed(update_info.m_param, pm, update_info.m_wids);
			break;
		case 'W':
			update_weight_cuda(update_info.m_param, pm);
			break;
		case 'B':
			update_bias_cuda(update_info.m_param, pm);
			break;
		case 'E':
			update_embed_cuda(update_info.m_param, pm, update_info.m_wids);
			break;
		default:
			throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "Optimizer::flush_update()");
		}
	}

	m_updateInfo.clear();

	return grad_norm;
}

void Optimizer::m_alloc_affine_param(Dict& param, Shape shape, bool use_bias, Dict kwArgs) {
}

void Optimizer::m_alloc_embed_param(Dict& param, vector<int64> voc_sizes, int64 vec_size, Dict kwArgs) {
}

void Optimizer::m_forward_affine(Dict param, Array<float> x, Array<float>& output) {
}

void Optimizer::m_backprop_affine(Dict param, Array<float> x, Array<float> G_affine, Array<float>& G_input) {
}

void Optimizer::m_forward_embed(Dict param, Array<int64> selector, Array<float>& output) {
}

void Optimizer::m_forward_embed_cuda(Dict param, Array<float> word_vecs, Array<int64> selector) {
}

void Optimizer::m_backprop_embed(Dict param, Array<float> G_words, Array<int64> selector) {
}

void Optimizer::m_backprop_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector) {
}
