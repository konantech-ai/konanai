/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "layer.cuh"

#include "../int_plugin/layers/full_layer.cuh"
#include "../int_plugin/layers/conv_layer.cuh"
#include "../int_plugin/layers/deconv_layer.cuh"
#include "../int_plugin/layers/pool_layer.cuh"
#include "../int_plugin/layers/dropout_layer.cuh"
#include "../int_plugin/layers/noise_layer.cuh"
#include "../int_plugin/layers/batch_norm_layer.cuh"
#include "../int_plugin/layers/custom_layer.cuh"
#include "../int_plugin/layers/serial_layer.cuh"
#include "../int_plugin/layers/parallel_layer.cuh"
#include "../int_plugin/layers/add_layer.cuh"
#include "../int_plugin/layers/rnn_layer.cuh"
#include "../int_plugin/layers/embedding_layer.cuh"
#include "../int_plugin/layers/merge_layer.cuh"
#include "../int_plugin/layers/embed_layer.cuh"
#include "../int_plugin/layers/attention_layer.cuh"
#include "../int_plugin/layers/pass_layer.cuh"
#include "../int_plugin/layers/extract_layer.cuh"
#include "../int_plugin/layers/expand_layer.cuh"
#include "../int_plugin/layers/reshape_layer.cuh"

#include "../cuda/cuda_conn.cuh"
#include "../cuda/cuda_math.h"

#include "../int_plugin/optimizer.cuh"

#include <stdio.h>
#include <assert.h>

int64 Layer::ms_called;

bool bTraceShapeDump = false;

Layer::Layer(Dict options, Shape& shape, bool& seq, Engine& engine) : m_engine(engine) {
	m_options = options;

	m_id = engine.get_next_layer_id();
	m_name = (string)get_option("name", "");

	m_set = (string)get_option("set", "");
	m_get = (string)get_option("get", "");

	m_org_input_shape = shape;

	if (options.find("inshape") != options.end()) {
		List option_shape = options["inshape"];
		Shape inshape(option_shape);
		assert(shape.total_size() == inshape.total_size());
		shape = inshape;
	}

	if (m_get != "") {
		engine.get_saved_shape(m_get, shape, seq);
	}

	m_seq = seq;

	m_input_shape = shape;
	m_output_shape = shape;
}

Layer::~Layer() {
}

string Layer::desc() {
	if (this == NULL) return "NULL";
	string layer_desc = m_layer_name;
	if (m_name != "") layer_desc += "-" + m_name;
	layer_desc += "(" + to_string(m_id) + ")";
	return layer_desc;
}

Layer* Layer::CreateLayer(Value hconfig, Shape& shape, bool& seq, Engine& engine) {
	Dict layer_options;

	if (hconfig.type() == vt::kint) {
		layer_options["width"] = hconfig;
		return new FullLayer(layer_options, shape, seq, engine);
	}
	
	List config = hconfig;

	int64 subnet_from = 1;

	if (config.size() > 1 && config[1].type() == vt::dict) {
		layer_options = config[1];
		subnet_from = 2;
	}

	assert(layer_options.find("load") == layer_options.end());
	assert(layer_options.find("save") == layer_options.end());  // save 대신 set 이름으로 처리하기로, 레이어 기본 생성자에서 속성 처리, 전처리 후 결과 저장

	List layer_subnet(config.begin()+subnet_from, config.end());

	string layer_name = config[0];

	Layer* pLayer = NULL;

	if (layer_name == "full") {
		assert(layer_subnet.size() == 0);
		pLayer = new FullLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "conv") {
		assert(layer_subnet.size() == 0);
		pLayer = new ConvLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "deconv") {
		assert(layer_subnet.size() == 0);
		pLayer = new DeconvLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "max") {
		assert(layer_subnet.size() == 0);
		pLayer = new MaxLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "avg") {
		assert(layer_subnet.size() == 0);
		pLayer = new AvgLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "globalavg") {
		assert(layer_subnet.size() == 0);
		/*
		char buffer[32];
		snprintf(buffer, 32, "[%lld, %lld]", shape[0], shape[1]);
		layer_options["stride"] = (string) buffer;
		*/
		List resolution;
		resolution.push_back(shape[0]);
		resolution.push_back(shape[1]);
		layer_options["stride"] = resolution;
		pLayer = new GlobalAvgLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "dropout") {
		assert(layer_subnet.size() == 0);
		pLayer = new DropoutLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "noise") {
		assert(layer_subnet.size() == 0);
		pLayer = new NoiseLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "batch_normal") {
		assert(layer_subnet.size() == 0);
		pLayer = new BatchNormalLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "custom") {
		assert(layer_subnet.size() == 0);
		pLayer = new CustomLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "serial") {
		assert(layer_subnet.size() > 0);
		pLayer = new SerialLayer(layer_options, layer_subnet, shape, seq, engine);
	}
	else if (layer_name == "parallel") {
		assert(layer_subnet.size() > 0);
		pLayer = new ParallelLayer(layer_options, layer_subnet, shape, seq, engine);
	}
	else if (layer_name == "add") {
		assert(layer_subnet.size() > 0);
		pLayer = new AddLayer(layer_options, layer_subnet, shape, seq, engine);
	}
	else if (layer_name == "rnn") {
		assert(layer_subnet.size() == 0);
		pLayer = new RnnLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "embedding") {
		assert(layer_subnet.size() > 0);
		pLayer = new EmbeddingLayer(layer_options, layer_subnet, shape, seq, engine);
	}
	else if (layer_name == "merge") {
		assert(layer_subnet.size() == 0);
		pLayer = new MergeLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "embed") {
		assert(layer_subnet.size() == 0);
		pLayer = new EmbedLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "attention") {
		assert(layer_subnet.size() == 0);
		pLayer = new AttentionLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "pass") {
		//assert(layer_subnet.size() >= 0);
		pLayer = new PassLayer(layer_options, layer_subnet, shape, seq, engine);
	}
	else if (layer_name == "extract") {
		assert(layer_subnet.size() == 0);
		pLayer = new ExtractLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "expand") {
		assert(layer_subnet.size() == 0);
		pLayer = new ExpandLayer(layer_options, shape, seq, engine);
	}
	else if (layer_name == "reshape") {
		assert(layer_subnet.size() == 0);
		pLayer = new ReshapeLayer(layer_options, shape, seq, engine);
	}
	else if (engine.in_macro(layer_name)) {
		Dict custom_options;
		custom_options["name"] = layer_name;
		custom_options["args"] = layer_options;
		pLayer = new CustomLayer(custom_options, shape, seq, engine);
		layer_name = "custom";
	}
	else {
		throw KaiException(KERR_ASSERT);
		return NULL;
	}

	if (pLayer->m_set != "") {
		engine.set_saved_shape(pLayer->m_set, shape, seq);
	}

	pLayer->m_layer_name = layer_name;

	return pLayer;
}

void Layer::forward_cuda_core(CudaConn& cuda, float* cuda_h, Shape hshape) {
	throw KaiException(KERR_ASSERT);
}

void Layer::backprop_cuda_core(CudaConn& cuda, float* cuda_gh, Shape hshape) {
	throw KaiException(KERR_ASSERT);
}

Array<float> Layer::forward_subnet(Array<float> hidden) {
	Dict hd_in = Value::wrap_dict("data", hidden);
	Dict hd_out = forward(hd_in);
	return hd_out["data"];
}

Array<float> Layer::backprop_subnet(Array<float> G_hidden) {
	Dict G_hd_in = Value::wrap_dict("data", G_hidden);
	Dict G_hd_out = backprop(G_hd_in);
	return G_hd_out["data"];
}

int64 Layer::dump_structure(int64 depth) {
	logger.Print("%*s%s: %s(%lld) : %s => %s", depth*2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str());
	return 0;
}

Dict Layer::alloc_affine_param(Shape shape, bool use_bias, param_init init_type) {
	Optimizer* optimizer = m_engine.get_optimizer();
	if (optimizer == NULL) throw KaiException(KERR_ASSERT);
	Dict kwArgs = m_engine.get_options();
	return optimizer->alloc_affine_param(shape, use_bias, init_type, kwArgs);
}

Dict Layer::alloc_embed_param(vector<int64> voc_sizes, int64 vec_size) {
	Optimizer* optimizer = m_engine.get_optimizer();
	if (optimizer == NULL) throw KaiException(KERR_ASSERT);
	Dict kwArgs = m_engine.get_options();
	return optimizer->alloc_embed_param(voc_sizes, vec_size, kwArgs);
}

Dict Layer::forward(Dict hidden) {
	FuncTimer func_timer(m_layer_name + "_layer (forward)");

	if (bTraceShapeDump) {
		Array<float> input_data = hidden["data"];
		logger.Print("%s forward: input shape is %s (m_input_shape: %s)", m_layer_name.c_str(), input_data.shape().desc().c_str(), m_input_shape.desc().c_str());
	}

	m_trace = false;
	ms_called++;

	//int64 mb_size = 0;
	Array<int64> length;

	if (!m_check_shape(hidden, m_org_input_shape, true)) {
		Array<float> data = hidden["data"];
		logger.Print("%s forward: hidden shape is %s (m_input_shape: %s, seq: %d)", m_layer_name.c_str(), data.shape().desc().c_str(), m_input_shape.desc().c_str(), (!seq_layer() && m_seq));
		throw KaiException(KERR_ASSERT);
	}

	if (m_get != "") {
		hidden = m_engine.get_saved_data(m_get, false);
	}

	if (!m_check_shape(hidden, m_input_shape, true)) {
		Array<float> data = hidden["data"];
		int64 mb_size = data.axis_size(0);
		if (data.shape().total_size() == mb_size * m_input_shape.total_size()) {
			Shape shape = m_input_shape.add_front(mb_size);
			hidden["data"] = data.reshape(shape);
		}
		else {
			logger.Print("%s forward: hidden shape is %s (m_input_shape: %s, seq: %d)", m_layer_name.c_str(), data.shape().desc().c_str(), m_input_shape.desc().c_str(), (!seq_layer() && m_seq));
			throw KaiException(KERR_ASSERT);
		}
	}


	if (CudaConn::UsingCuda()) {
		Dict hidden_cuda = m_forward_cuda_main(hidden);
		hidden = hidden_cuda;
}
	else {
		hidden = m_forward_main(hidden);
	}


	if (!m_check_shape(hidden, m_output_shape, false)) {
		Array<float> data = hidden["data"];
		logger.Print("%s forward: hidden output shape is %s (m_output_shape: %s, seq: %d)", m_layer_name.c_str(), data.shape().desc().c_str(), m_output_shape.desc().c_str(), (!seq_layer() && m_seq));
		throw KaiException(KERR_ASSERT);
	}

	if (m_set != "") {
		m_engine.set_saved_data(m_set, hidden, false);
	}

	if (!m_engine.m_is_training) {
		m_aux.clear();
	}

	if (bTraceShapeDump) {
		Array<float> output_data = hidden["data"];
		logger.Print("%s forward: output shape is %s (m_output_shape: %s)", m_layer_name.c_str(), output_data.shape().desc().c_str(), m_output_shape.desc().c_str());
	}

	return hidden;
}

Dict Layer::backprop(Dict G_hidden) {
	FuncTimer func_timer(m_layer_name + "_layer (backprop)");

	if (bTraceShapeDump) {
		Array<float> h_data = G_hidden["data"];
		logger.Print("%s backprop: h_data shape is %s (m_output_shape: %s)", m_layer_name.c_str(), h_data.shape().desc().c_str(), m_output_shape.desc().c_str());
	}

	m_trace = false;
	ms_called++;

	if (m_set != "") {
		G_hidden = m_engine.get_saved_derv(m_set, false);
	}

	int64 mb_size = 0;
	Array<int64> length;

	if (!m_check_shape(G_hidden, m_output_shape, false)) {
		Array<float> data = G_hidden["data"];
		throw KaiException(KERR_ASSERT);
	}

	if (CudaConn::UsingCuda()) {
		Dict G_hidden_cuda = m_backprop_cuda_main(G_hidden);
		G_hidden = G_hidden_cuda;
	}
	else {
		G_hidden = m_backprop_main(G_hidden);
	}

	/*
	if (!seq_layer() && m_seq) {
		m_split_time_axis(G_hidden, mb_size, length, false);
	}
	*/

	if (!m_check_shape(G_hidden, m_input_shape, true)) {
		Array<float> data = G_hidden["data"];
		logger.Print("backprop of %s(%lld) layer should produce %s output with mb_size, but %s was made.", m_layer_name.c_str(), m_id, m_input_shape.desc().c_str(), data.shape().desc().c_str());
		throw KaiException(KERR_ASSERT);
	}

	if (m_get != "") {
		Array<float> data = G_hidden["data"];
		mb_size = data.axis_size(0);
		m_engine.set_saved_derv(m_get, G_hidden, false);
		Array<float> dummy = kmath->zeros(m_org_input_shape.add_front(mb_size));
		G_hidden = Value::wrap_dict("data", dummy);
	}

	if (!m_check_shape(G_hidden, m_org_input_shape, true)) {
		Array<float> data = G_hidden["data"];
		int64 mb_size = data.axis_size(0);
		if (data.total_size() == mb_size * m_org_input_shape.total_size()) {
			Shape shape = m_org_input_shape.add_front(mb_size);
			G_hidden["data"] = data.reshape(shape);
		}
		else {
			logger.Print("backprop of %s(%lld) layer should produce %s output with mb_size, but %s was made.", m_layer_name.c_str(), m_id, m_input_shape.desc().c_str(), data.shape().desc().c_str());
			throw KaiException(KERR_ASSERT);
		}
	}

	if (bTraceShapeDump) {
		Array<float> in_grad = G_hidden["data"];
		logger.Print("%s backprop: in_grad shape is %s (m_input_shape: %s)", m_layer_name.c_str(), in_grad.shape().desc().c_str(), m_input_shape.desc().c_str());
	}

	return G_hidden;
}

// default-data 이외의 성분을 전처리 과정에 포함시키려면 아래 메서드를 재정의할 것
Dict Layer::m_forward_main(Dict hin) {
	Array<float> x = hin["data"];

	Dict hout;
	Array<float> y = m_forward_farr(x);
	hout["data"] = y;

	return hout;
}

// default-data 이외의 성분을 전처리 과정에 포함시키려면 아래 메서드를 재정의할 것
Dict Layer::m_backprop_main(Dict G_hin) {
	Array<float> gy = G_hin["data"];
	Dict G_hout;
	Array<float> gx = m_backprop_farr(gy);
	G_hout["data"] = gx;
	return G_hout;
}

Array<float> Layer::m_forward_farr(Array<float> hidden) {
	logger.Print("m_forward_farr(hidden) for %s layer is not implemented yet.", m_layer_name.c_str());
	throw KaiException(KERR_ASSERT);
	return hidden;
}

Array<float> Layer::m_backprop_farr(Array<float> G_hidden) {
	logger.Print("m_backprop_farr(hidden) for %s layer is not implemented yet.", m_layer_name.c_str());
	throw KaiException(KERR_ASSERT);
	return G_hidden;
}

// default-data 이외의 성분을 전처리 과정에 포함시키려면 아래 메서드를 재정의할 것
Dict Layer::m_forward_cuda_main(Dict hin) {
	Array<float> x = hin["data"];
	Dict hout;
	Array<float> y = m_forward_cuda_farr(x);
	hout["data"] = y;
	return hout;
}

// default-data 이외의 성분을 전처리 과정에 포함시키려면 아래 메서드를 재정의할 것
Dict Layer::m_backprop_cuda_main(Dict G_hin) {
	Array<float> gy = G_hin["data"];
	Dict G_hout;
	Array<float> gx = m_backprop_cuda_farr(gy);
	G_hout["data"] = gx;
	return G_hout;
}

Array<float> Layer::m_forward_cuda_farr(Array<float> hidden) {
	logger.Print("m_forward_cuda_farr(hidden) for %s layer is not implemented yet.", m_layer_name.c_str());
	throw KaiException(KERR_ASSERT);
	return hidden;
}

Array<float> Layer::m_backprop_cuda_farr(Array<float> G_hidden) {
	logger.Print("m_backprop_cuda_farr(hidden) for %s layer is not implemented yet.", m_layer_name.c_str());
	throw KaiException(KERR_ASSERT);
	return G_hidden;
}

Array<float> Layer::forward_affine(Dict pm, Array<float> x) {
	Optimizer* optimizer = m_engine.get_optimizer();
	if (optimizer == NULL) throw KaiException(KERR_ASSERT);
	return optimizer->forward_affine(pm, x);
}

void Layer::forward_affine_cuda(Dict param, bool use_bias, float* a_out, float* x_in, int64 nrow, int64 nvec, int64 ncol) {
	int64 asize = nrow * ncol;

	float* weight = m_fetch_weight_ptr(param);
	cu_call(ker_matmul, asize, (asize, a_out, x_in, weight, nrow, nvec, ncol));

	if (use_bias) {
		float* bias = m_fetch_bias_ptr(param);
		cu_call(ker_add_bias, asize, (asize, a_out, bias, nrow, nvec, ncol));
	}
}

void Layer::backprop_affine_cuda(Dict param, bool use_bias, float* cuda_gx, float* cuda_ga, float* cuda_x, int64 nrow, int64 nvec, int64 ncol) {
	CudaConn cuda("Layer::backprop_affine_cuda", NULL);

	Optimizer* optimizer = m_engine.get_optimizer();
	if (optimizer == NULL) throw KaiException(KERR_ASSERT);

	float* cuda_w = m_fetch_weight_ptr(param);
	float* cuda_gw = cuda.alloc_float_mem(CudaConn::GetShape(cuda_w));

	int64 xsize = nrow * nvec;
	int64 wsize = nvec * ncol;

	cu_call(ker_matmul_derv_x, xsize, (xsize, cuda_gx, cuda_ga, cuda_w, nrow, nvec, ncol));

	if (!m_engine.m_block_update) {
		cu_call(ker_matmul_derv_w, wsize, (wsize, cuda_gw, cuda_ga, cuda_x, nrow, nvec, ncol));

		Array<float> G_weight = cuda.detach(cuda_gw);
		optimizer->regist_update_weight_cuda(param["w"], G_weight);

		if (use_bias) {
			float* cuda_b = m_fetch_bias_ptr(param);
			float* cuda_gb = cuda.alloc_float_mem(CudaConn::GetShape(cuda_b));
			int64 bsize = ncol;

			cu_call(ker_add_bias_derv, bsize, (bsize, cuda_gb, cuda_ga, nrow, nvec, ncol));

			Array<float> G_bias = cuda.detach(cuda_gb);
			optimizer->regist_update_bias_cuda(param["b"], G_bias);
		}
	}
}

Array<float> Layer::backprop_affine(Dict pm, Array<float> x, Array<float> G_affine) {
	if (!m_engine.m_block_update) {
		Optimizer* optimizer = m_engine.get_optimizer();
		if (optimizer == NULL) throw KaiException(KERR_ASSERT);
		return optimizer->backprop_affine(pm, x, G_affine);
	}
	else { // 파라미터 업데이트는 하지 않고 입력에 대한 기울기만 계산
		Dict pm_w = pm["w"];
		Array<float> w = pm_w["_pm_"];
		Array<float> g_affine_input = w.transpose();
		return kmath->matmul(G_affine, g_affine_input);
	}
}

Array<float> Layer::forward_embed(Dict param, Array<int64> wids) {
	Optimizer* optimizer = m_engine.get_optimizer();
	if (optimizer == NULL) throw KaiException(KERR_ASSERT);
	return optimizer->forward_embed(param, wids);
}

void Layer::backprop_embed(Dict param, Array<int64> wids, Array<float> G_words) {
	throw KaiException(KERR_ASSERT);
}

void Layer::forward_embed_cuda(Dict param, Array<float> word_vecs_out, Array<int64> wids) {
	Optimizer* optimizer = m_engine.get_optimizer();
	if (optimizer == NULL) throw KaiException(KERR_ASSERT);
	optimizer->forward_embed_cuda(param, word_vecs_out, wids);
}

void Layer::backprop_embed_cuda(Dict param, Array<float> G_words, Array<int64> wids) {
	if (!m_engine.m_block_update) {
		Optimizer* optimizer = m_engine.get_optimizer();
		if (optimizer == NULL) throw KaiException(KERR_ASSERT);
		assert(G_words.shape().remove_end() == wids.shape().remove_end());
		optimizer->regist_update_embed_cuda(param, G_words, wids);
	}
}

void Layer::set_activate(string func) {
	if (func == "none") m_nActFunc = ACTFUNC_NONE;
	else if (func == "relu") m_nActFunc = ACTFUNC_RELU;
	else if (func == "sigmoid") m_nActFunc = ACTFUNC_SIGMOID;
	else if (func == "tanh") m_nActFunc = ACTFUNC_TANH;
	else if (func == "leaky_relu") {
		m_nActFunc = ACTFUNC_LEAKY_RELU;
		m_leaky_alpha = get_option("leaky_alpha", 0.0f);
	}
	else if (func == "gelu") m_nActFunc = ACTFUNC_GELU;
	else throw KaiException(KERR_ASSERT);
}

Array<float> Layer::activate(Array<float> hidden, int64 nFunc) {
	if (nFunc < 0) nFunc = m_nActFunc;

	switch (nFunc) {
	case ACTFUNC_NONE:
		return hidden;
	case ACTFUNC_RELU:
		return kmath->relu(hidden);
	case ACTFUNC_SIGMOID:
		return kmath->sigmoid(hidden);
	case ACTFUNC_TANH:
		return kmath->tanh(hidden);
	case ACTFUNC_LEAKY_RELU:
		return kmath->leaky_relu(hidden, m_leaky_alpha);
	case ACTFUNC_GELU:
		return kmath->gelu(hidden);
	default:
		throw KaiException(KERR_ASSERT);
	}

	return hidden;
}

Array<float> Layer::activate_derv(Array<float> G_hidden, Array<float> x, Array<float> y, int64 nFunc) {
	if (nFunc < 0) nFunc = m_nActFunc;

	switch (nFunc) {
	case ACTFUNC_NONE:
		return G_hidden;
	case ACTFUNC_RELU:
		return kmath->relu_derv(y) * G_hidden;
	case ACTFUNC_SIGMOID:
		return kmath->sigmoid_derv(y) * G_hidden;
	case ACTFUNC_TANH:
		return kmath->tanh_derv(y) * G_hidden;
	case ACTFUNC_LEAKY_RELU:
		return kmath->leaky_relu_derv(y, m_leaky_alpha) * G_hidden;
	case ACTFUNC_GELU:
		return kmath->gelu_derv(x) * G_hidden;
	default:
		throw KaiException(KERR_ASSERT);
	}

	return G_hidden;
}

Shape Layer::get_2d_option(string key, Value def) {
	Value value = get_option(key, def);
	if (value.type() == vt::kint) return Shape(value, value);
	else if (value.type() == vt::list) {
		List list = value;
		assert(list.size() == 2);
		return Shape(list[0], list[1]);
	}
	else if (value.type() == vt::shape) return value;
	else {
		throw KaiException(KERR_ASSERT);
	}
	return value;
}

Value Layer::get_option(string key, Value def) {
	Value value = Value::seek_option(m_options, key, def);
	if (value.type() == vt::string) {
		string str = value;
		if (str.substr(0, 8) == "dataset.") {
			return m_engine.get_dataset_ext_param(str.substr(8));
		}
	}
	return value;
}

bool Layer::m_check_shape(Dict dict, Shape shape, bool check_x) {
	if (m_engine.use_custom_data_format()) return true;

	Array<float> arr = dict["data"];
	Shape arr_shape = arr.shape();

	bool seq = m_seq;

	if (seq_layer()) {
		SeqLayer* pSeqlayer = (SeqLayer*)this;
		seq = check_x ? pSeqlayer->m_inseq : pSeqlayer->m_outseq;
	}
	
	if (seq) {
		int64 timesteps = arr_shape[1];
		shape = shape.add_front(timesteps);
	}
	int64 mb_size = arr_shape[0];
	return arr_shape == shape.add_front(mb_size);
}

bool Layer::seek_wvec_param(string layer_name, Dict& param) {
	if (layer_name != m_name) return false;
	param = m_get_wvec_param();
	return true;
}

Dict Layer::m_get_wvec_param() {
	throw KaiException(KERR_ASSERT);	// word vector는 embed, embedding layer에서만 재정의되어 값이 반환되고 다른 layer들은 요청 자체를 오류 처리한다.
	return Dict();
}

bool Layer::seek_named_param_set(string param_path, Dict& param) {
	if (m_name == "") return false;
	int64 length = m_name.length();
	if (param_path.substr(0, length) != m_name) return false;
	if (param_path[length] != '.') return false;

	string rest = param_path.substr(length + 1);
	return m_seek_named_param_set(m_param, rest, param);
}

Layer* Layer::seek_named_layer(string name) {
	return (m_name == name) ? this : NULL;
}

void Layer::copyParam(Layer* pLayerSrc) {
	m_param = Value::copy(m_param);
}

bool Layer::m_seek_named_param_set(Dict pm, string param_path, Dict& param_set) {
	for (Dict::iterator it = pm.begin(); it != pm.end(); it++) {
		if (it->first == param_path) {
			param_set = it->second;
			if (param_set.find("_pm_") == param_set.end()) return false;
			return true;
		}
		int64 length = it->first.length();
		if (param_path.substr(0, length) != it->first) continue;
		if (param_path[length] != '.') continue;

		if (it->second.type() != vt::dict) return false;
		string rest = param_path.substr(length + 1);
		return m_seek_named_param_set(it->second, rest, param_set);
	}

	return false;
}

void Layer::m_update_weight(Dict param, Array<float> G_weight) {
	if (m_engine.m_block_update) return;

	if (param.find("w") == param.end()) throw KaiException(KERR_ASSERT);

	if (G_weight.is_cuda()) {
		m_engine.get_optimizer()->regist_update_weight_cuda(param["w"], G_weight);
	}
	else {
		m_engine.get_optimizer()->regist_update_weight(param["w"], G_weight);
	}
}

void Layer::m_update_bias(Dict param, Array<float> G_bias) {
	if (m_engine.m_block_update) return;

	if (param.find("b") == param.end()) throw KaiException(KERR_ASSERT);

	if (G_bias.is_cuda()) {
		m_engine.get_optimizer()->regist_update_bias_cuda(param["b"], G_bias);
	}
	else {
		m_engine.get_optimizer()->regist_update_bias(param["b"], G_bias);
	}
}

void Layer::m_update_embed(Dict param, Array<float> G_words, Array<int64> wids) {
	if (m_engine.m_block_update) return;

	if (G_words.is_cuda()) {
		m_engine.get_optimizer()->regist_update_embed_cuda(param, G_words, wids);
	}
	else {
		m_engine.get_optimizer()->regist_update_embed(param, G_words, wids);
	}
}

Array<float> Layer::m_fetch_weight(Dict param) {
	return m_engine.get_optimizer()->fetch_core(param["w"]);
}

Array<float> Layer::m_fetch_bias(Dict param) {
	if (param.find("b") == param.end()) throw KaiException(KERR_ASSERT);
	return m_engine.get_optimizer()->fetch_core(param["b"]);
}

Array<float> Layer::m_fetch_embed(Dict param) {
	return m_engine.get_optimizer()->fetch_core(param);
}

float* Layer::m_fetch_weight_ptr(Dict param) {
	Array<float> weight = m_engine.get_optimizer()->fetch_core(param["w"]);
	if (weight.is_cuda()) {
		return CudaConn::GetCudaMem(weight);
	}
	else {
		return weight.data_ptr();
	}
}

float* Layer::m_fetch_bias_ptr(Dict param) {
	if (param.find("b") == param.end()) throw KaiException(KERR_ASSERT);
	Array<float> bias = m_engine.get_optimizer()->fetch_core(param["b"]);
	if (bias.is_cuda()) {
		return CudaConn::GetCudaMem(bias);
	}
	else {
		return bias.data_ptr();
	}
}

float* Layer::m_fetch_embed_ptr(Dict param) {
	Array<float> embed = m_engine.get_optimizer()->fetch_core(param);
	if (embed.is_cuda()) {
		return CudaConn::GetCudaMem(embed);
	}
	else {
		return embed.data_ptr();
	}
}

string Layer::m_get_affine_param_desc(Dict param, int64* pm_cnt) {
	*pm_cnt = m_get_param_count(param);
	return m_engine.get_optimizer()->get_affine_param_desc(param);
}
string Layer::m_get_embed_param_desc(Dict param, int64* pm_cnt) {
	*pm_cnt = m_get_param_count(param);
	return m_engine.get_optimizer()->get_embed_param_desc(param);
}

int64 Layer::m_get_param_count(Dict param) {
	int64 pm_cnt = 0;

	for (Dict::iterator it = param.begin(); it != param.end(); it++) {
		Value value = it->second;
		if (value.type() == vt::farray) {
			Array<float> pm = value;
			pm_cnt += pm.total_size();
		}
		else if (value.type() == vt::dict) {
			Dict sub_pm = value;
			pm_cnt += m_get_param_count(sub_pm);
		}
	}

	return pm_cnt;
}

ComplexLayer::ComplexLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine) : Layer(options, shape, seq, engine) {
}

ComplexLayer::~ComplexLayer() {
	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		delete (*it);
	}
}

int64 ComplexLayer::dump_structure(int64 depth) {
	int64 param_cnt = 0;
	logger.Print("%*s%s: %s(%lld) : %s => %s", depth*2, "", m_layer_name.c_str(), m_name.c_str(), m_id, m_input_shape.desc().c_str(), m_output_shape.desc().c_str());
	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		Layer* pLayer = *it;
		param_cnt += pLayer->dump_structure(depth + 1);
	}
	logger.Print("%*s%s: %s(%lld) : %lld pms", depth*2, "", m_layer_name.c_str(), m_name.c_str(), m_id, param_cnt);
	return param_cnt;
}

bool ComplexLayer::seek_wvec_param(string layer_name, Dict& param) {
	if (Layer::seek_wvec_param(layer_name, param)) return true;

	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		Layer* pLayer = *it;
		if (pLayer->seek_wvec_param(layer_name, param)) return true;
	}
	return false;
}

bool ComplexLayer::seek_named_param_set(string param_path, Dict& param_set) {
	if (Layer::seek_named_param_set(param_path, param_set)) return true;

	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		Layer* pLayer = *it;
		if (pLayer->seek_named_param_set(param_path, param_set)) return true;
	}
	return false;
}

Layer* ComplexLayer::seek_named_layer(string name) {
	if (m_name == name) return this;

	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		Layer* pLayer = (*it)->seek_named_layer(name);
		if (pLayer) return pLayer;
	}
	return NULL;
}
