/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/common.h"
#include "../cuda/cuda_conn.cuh"
#include "../cuda/cuda_math.h"

#include "../int_plugin/internal_plugin.h"

struct _UpdateInfo {
	char m_paramType;
	Dict m_param;
	Array<float> m_grad;
	Array<int64> m_wids;
};

class Optimizer : public InternalPluginComponent {
public:	// functions
	Optimizer(string name, Dict kwArgs);
	virtual ~Optimizer();

	static string introduce_curr_instance();
	static Optimizer* get_curr_instance();
	static Optimizer* create_instance(string component_name, Dict kwArgs);
	static void set_plugin_component(string component_name);

	static Optimizer* check_in_curr_instance(Dict kwArgs);
	static Optimizer* check_in_named_instance(string component_name, Dict kwArgs);
	void check_out();

	virtual void setup(Dict kwArgs);

	virtual string introduce();
	virtual string introduce_extra();

	Array<float> fetch_core(Dict param);

	string get_affine_param_desc(Dict param);
	string get_embed_param_desc(Dict param);

	Dict alloc_affine_param(Shape shape, bool use_bias, param_init init_type, Dict kwArgs);
	Dict alloc_embed_param(vector<int64> voc_sizes, int64 vec_size, Dict kwArgs);

	Array<float> forward_affine(Dict param, Array<float> x);
	Array<float> backprop_affine(Dict param, Array<float> x, Array<float> G_affine);
	
	Array<float> forward_embed(Dict param, Array<int64> selector);
	void forward_embed_cuda(Dict param, Array<float> word_vecs, Array<int64> selector);

	void backprop_embed(Dict param, Array<float> G_words, Array<int64> selector);
	void backprop_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector);

	void regist_update_weight(Dict pm, Array<float> Grad);
	void regist_update_bias(Dict pm, Array<float> Grad);
	void regist_update_embed(Dict pm, Array<float> Grad, Array<int64> selector);

	void regist_update_weight_cuda(Dict pm, Array<float> Grad);
	void regist_update_bias_cuda(Dict pm, Array<float> Grad);
	void regist_update_embed_cuda(Dict pm, Array<float> Grad, Array<int64> selector);

	float flush_update(bool trace_grad_norm, float clip_grad_threshold);

protected: // functions
	virtual void m_alloc_affine_param(Dict& param, Shape shape, bool use_bias, Dict kwArgs);
	virtual void m_alloc_embed_param(Dict& param, vector<int64> voc_sizes, int64 vec_size, Dict kwArgs);

	virtual void m_forward_affine(Dict param, Array<float> x, Array<float>& output);
	virtual void m_backprop_affine(Dict param, Array<float> x, Array<float> G_affine, Array<float>& G_input);

	virtual void m_forward_embed(Dict param, Array<int64> selector, Array<float>& output);
	virtual void m_forward_embed_cuda(Dict param, Array<float> word_vecs, Array<int64> selector);

	virtual void m_backprop_embed(Dict param, Array<float> G_words, Array<int64> selector);
	virtual void m_backprop_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector);

	virtual void update_weight(Dict param, Array<float> G_weight) = 0;
	virtual void update_bias(Dict param, Array<float> G_bias) = 0;
	virtual void update_embed(Dict param, Array<float> G_words, Array<int64> selector) = 0;

	virtual void update_weight_cuda(Dict param, Array<float> G_weight) = 0;
	virtual void update_bias_cuda(Dict param, Array<float> G_bias) = 0;
	virtual void update_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector) = 0;

protected: // functions
	virtual Array<float> m_init_weight(Shape shape, param_init init_type, Dict kwArgs);
	virtual Array<float> m_eval_decay_delta(Array<float> param);
	
public: // variables

protected: // variables
	static Optimizer* ms_curr_instance;

	int64 m_ref_count;

	string m_name;

	float m_learning_rate;
	float m_l2_decay;
	float m_l1_decay;

	vector<_UpdateInfo> m_updateInfo;
};

