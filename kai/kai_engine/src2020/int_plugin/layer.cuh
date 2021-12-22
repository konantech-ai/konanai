/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/array.h"
#include "../core/dim.h"
#include "../core/shape.h"
#include "../core/host_math.h"
#include "../core/engine.h"
#include "../core/value.h"
#include "../core/func_timer.h"
#include "../core/log.h"

#include "../cuda/cuda_conn.cuh"
#include "../cuda/cuda_math.h"
#include "../cuda/cuda_kernels.h"

#include "../int_plugin/optimizer.cuh"

#include <stdio.h>
#include <assert.h>
#include <float.h>

class Engine;

#define ACTFUNC_NONE		0
#define ACTFUNC_RELU		1
#define ACTFUNC_SIGMOID		2
#define ACTFUNC_TANH		3
#define ACTFUNC_LEAKY_RELU	4
#define ACTFUNC_GELU		5

class CudaConn;

class Layer {
public:
	Layer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~Layer();

	static Layer* CreateLayer(Value hconfig, Shape& shape, bool& seq, Engine& engine);
	
	Dict forward(Dict xs);
	Dict backprop(Dict G_hidden);

	virtual void forward_cuda_core(CudaConn& cuda, float* cuda_h, Shape hshape);
	virtual void backprop_cuda_core(CudaConn& cuda, float* cuda_gh, Shape hshape);

	virtual Array<float> forward_subnet(Array<float> hidden);
	virtual Array<float> backprop_subnet(Array<float> G_hidden);

	//virtual Array<float> forward_cuda_subnet(Array<float> hidden);
	//virtual Array<float> backprop_cuda_subnet(Array<float> G_hidden);

	virtual int64 dump_structure(int64 depth);

	virtual bool seek_wvec_param(string layer_name, Dict& param);
	virtual bool seek_named_param_set(string param_path, Dict& param_set);

	virtual Layer* seek_named_layer(string name);

	void copyParam(Layer* pLayerSrc);

	string desc();

	Dict m_param;

	string m_layer_name;
	string m_name;

	string m_get;
	string m_set;

	int64 m_id;

protected:
	virtual bool seq_layer() { return false; }

	bool m_seq;

	Dict m_options;
	Dict m_aux;

	Shape m_input_shape;	// inshape 옵션이나 get 옵션에 의해 입력이 대치되는 경우 대치 이후의 입력 형상, 다른 경우에는 m_org_input_shape와 동일
	Shape m_output_shape;
	Shape m_org_input_shape;  // 레이어 연결 관계에 따라 이전 레이어 출력으로부터 유추한 입력 형상, 첫 레이어는 데이터셋이 정하는 입력 형상

	int64 m_nActFunc;
	float m_leaky_alpha;

	Engine& m_engine;

	bool m_trace;
	static int64 ms_called;

protected:
	friend class AddLayer;
	friend class ConvLayer;

	Value get_option(string key, Value def = None);
	string get_option_string(string key, string def = "") { return get_option(key, def); }

	Shape get_2d_option(string key, Value def = None);

	Dict alloc_affine_param(Shape shape, bool use_bias, param_init init_type= param_init::gauss);
	Dict alloc_embed_param(vector<int64> voc_sizes, int64 vec_size);

	virtual Dict m_forward_main(Dict hidden);
	virtual Dict m_backprop_main(Dict G_hidden);

	virtual Dict m_forward_cuda_main(Dict hidden);
	virtual Dict m_backprop_cuda_main(Dict G_hidden);

	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);

	Array<float> forward_affine(Dict param, Array<float> x);
	Array<float> backprop_affine(Dict param, Array<float> x, Array<float> G_affine);

	void forward_affine_cuda(Dict param, bool use_bias, float* a_out, float* x_in, int64 nrow, int64 nvec, int64 ncol);
	void backprop_affine_cuda(Dict param, bool use_bias, float* gx_out, float* gy_in, float* x_in, int64 nrow, int64 nvec, int64 ncol);

	Array<float> forward_embed(Dict param, Array<int64> wids);
	void backprop_embed(Dict param, Array<int64> wids, Array<float> G_words);

	void forward_embed_cuda(Dict param, Array<float> word_vecs_out, Array<int64> wids);
	void backprop_embed_cuda(Dict param, Array<float> G_words, Array<int64> wids);

	void set_activate(string func);

	Array<float> activate(Array<float> hidden, int64 nFunc = -1);
	Array<float> activate_derv(Array<float> G_hidden, Array<float> affine, Array<float> y, int64 nFunc = -1);

	//Array<float> activate_cuda(Array<float> hidden, int64 nFunc = -1);
	//Array<float> activate_derv_cuda(Array<float> G_hidden, Array<float> affine, Array<float> y, int64 nFunc = -1);

	/*
	void update_param(Dict& pm, string key, Array<float> G_param);

	void update_param_select(Dict& pm, string key, Array<int64> selector, Array<float> G_param);

	void update_by_select_idx(Dict& pm, string key, Array<int64> selector, Array<float> G_param, int64 dic_count, int64* voc_counts);
	//void update_by_select_idx_cuda(Dict& pm, string key, Array<int64> selector, Array<float> G_param, int64 dic_count, int64* voc_counts);
	//void update_by_select_idx_no_cuda(Dict& pm, string key, Array<int64> selector, Array<float> G_param, int64 dic_count, int64* voc_counts);

	Array<float> eval_adam_delta(Dict& pm, string key, Array<float> G_param);
	Array<float> eval_adam_nth_delta(Dict& pm, string key, int64 nth, Array<float> G_param);
	*/

	bool is_weight_param(string key) { return key == "w" || key == "k"; }
	bool m_check_shape(Dict hidden, Shape shape, bool check_x);
	
	virtual Dict m_get_wvec_param();

	bool m_seek_named_param_set(Dict pm, string param_path, Dict& param_set);

	void m_update_weight(Dict param, Array<float> G_weight);
	void m_update_bias(Dict param, Array<float> G_bias);
	void m_update_embed(Dict param, Array<float> G_words, Array<int64> wids);

	Array<float> m_fetch_weight(Dict param);
	Array<float> m_fetch_bias(Dict param);
	Array<float> m_fetch_embed(Dict param);

	float* m_fetch_weight_ptr(Dict param);
	float* m_fetch_bias_ptr(Dict param);
	float* m_fetch_embed_ptr(Dict param);

	string m_get_affine_param_desc(Dict param, int64* pm_cnt);
	string m_get_embed_param_desc(Dict param, int64* pm_cnt);

	int64 m_get_param_count(Dict param);

	//Array<float> m_fetch_embed_arr(Dict param);
	//float* m_fetch_affine_ptr(Dict param, string key);
	//float* m_fetch_embed_ptr(Dict param);

	/*
	string m_get_affine_param_desc(Dict param, int64* pm_cnt);
	string m_get_embed_param_desc(Dict param, int64* pm_cnt);

	int64 m_get_allocated_param_size(Dict param);
	*/
};

class ComplexLayer : public Layer {
public:
	ComplexLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine);
	virtual ~ComplexLayer();

	virtual int64 dump_structure(int64 depth);
	virtual bool seek_wvec_param(string layer_name, Dict& param);
	virtual bool seek_named_param_set(string param_path, Dict& param_set);

	virtual Layer* seek_named_layer(string name);

protected:
	Layers m_layers;
};

