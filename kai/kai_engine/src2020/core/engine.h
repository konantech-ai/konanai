/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "dataset.h"
#include "macro_pack.h"
#include "../int_plugin/layer.cuh"

class Optimizer;

class Engine {
public:
	Engine(const char* name, Dataset& dataset, const char* conf, const char* options = NULL, MacroPack* pMacros=NULL);
	virtual ~Engine();

	virtual void exec_all(const char* args="", bool skip_test=false);

	void build_hidden_net(List hconfigs, Shape& shape, bool &seq, Layers& layers, List& pms);
	void build_output_net(Shape& shape, bool& seq, Layers& layers, List& pms);

	static Dict get_default_options();
	Dict get_options();
	Value lookup_option(string key, Value def = None);

	virtual void train();
	virtual void test();
	virtual void visualize(int64 count);
	void show_param_dist();

	virtual void save_param(string name = "");
	virtual void load_param(string name = "");
	virtual void load_param(int64 epoch, int64 batch);

	virtual void forward_postproc(Dict xs, Dict ys, Dict outs, string mode, Dict& losses, Dict& accs, Dict& losses_batch, Dict& accs_batch);
	virtual Dict backprop_postproc(Dict xs, Dict ys, Dict outs);

	void valid_report(Dict& losses, Dict& accs, Dict acc, int64 nepoch, int64 nbatch);
	void log_train(Dict& losses, Dict& accs, int64 nepoch, int64 nbatch);
	void train_report(string title);
	void test_report(Dict acc);

	virtual void forward_extra_cost(Dict& loss, List pms, Dict xs, Dict ys, Dict outs);
	virtual void backprop_extra_cost();

	virtual Dict m_evaluate_loss(Dict xs, Dict ys, Dict outs, string mode = "default");
	virtual Dict m_eval_accuracy(Dict xs, Dict ys, Dict outs = Dict(), string mode = "default");

	void regist_kernel(Array<float> kernel) { m_kernels.push_back(kernel); }
	void add_map(Array<float> map) { throw KaiException(KERR_ASSERT); }

	List get_macro(string name, Dict args) { return m_pMacros->get_macro(name, args); }
	bool in_macro(string name) { return m_pMacros->in_macro(name); }

	int64 get_next_layer_id() { return m_next_layer_id++; }

	void save_named_param_set(string param_path, string dic_path);
	void save_named_wvec_param(string layer_name, string dic_path);

	//void load_named_param(string dic_path, Array<float>& arr);
	void load_named_param_set(string dic_path, Dict& pm, string key, bool isAdam, bool nsingle);

	Value get_dataset_ext_param(string key) { return m_dataset.get_ext_param(key); }
	
	void get_saved_shape(string name, Shape& shape, bool& seq) { shape = m_saved_shape[name]; seq = m_saved_seq[name]; }
	void set_saved_shape(string name, Shape shape, bool seq) { m_saved_shape[name] = shape; m_saved_seq[name] = seq; }

	Dict get_saved_data(string name, bool from_ext = true);
	Dict get_saved_derv(string name, bool from_ext = true);

	void set_saved_data(string name, Dict data, bool from_ext = true);
	void set_saved_derv(string name, Dict data, bool from_ext = true);

	void copy_params(Engine& srcEngine, string pairs);

	Optimizer* get_optimizer() { return m_optimizer; }

	virtual bool use_custom_data_format();

	virtual Layer* seek_named_layer(string name);

	//static Dict get_options() { return Value::parse_dict(ms_default_option); }

protected:
	virtual Dict forward_neuralnet(Dict xs, Layers& layers);
	virtual Dict backprop_neuralnet(Dict G_outs, Layers& layers);

	void m_show_n_flush_grad_norm(int64 epoch);

public:
	bool m_is_training;
	bool m_block_update;
	
	int64 m_next_layer_id;

	float l2_decay;
	float l1_decay;
	float learning_rate;

protected:
	static const char* ms_default_option;
	//static const char* ms_default_args;

	string m_name;
	
	Dict m_engine_options;
	Dict m_exec_options;

	Dict m_saved_shape;
	Dict m_saved_seq;
	Dict m_saved_data;
	Dict m_saved_derv;

	Dataset& m_dataset;

	int64 m_acc_epoch;

	int64 m_acc_time;

	time_t m_time1;
	time_t m_time2;

	List m_kernels;
	//List saved_data;

	int64 m_epoch_count;
	int64 m_batch_count;
	int64 m_valid_count;
	int64 m_batch_size;
	int64 m_batch_report;
	int64 m_temp_batch_count;
	int64 m_report;
	int64 m_epoch_visualize;
	int64 m_epoch_save;
	int64 m_epoch_show_grad_norm;
	int64 m_in_batch_valid;
	int64 m_in_batch_visualize;
	int64 m_in_batch_save;
	int64 m_start_epoch;
	int64 m_start_batch;
	int64 m_show_cnt;

	bool m_rand_sync;
	bool m_show_params;

	float m_clip_grad;

	enum data_channel m_train_channel;

	Layers m_layers;
	List m_pms;
	List m_grad_norm;

	Optimizer* m_optimizer;

	bool m_reporter_cont;

	queue<Dict> m_report_Queue;
	mutex* m_mu_reporter;

	thread* m_reporter;
	int64 m_reporter_seed;

	static void ms_report(void* aux);
	void m_report_loop();
	void m_sleep();

	//Dict m_pm_name_map;

	MacroPack* m_pMacros;

	void m_build_neuralnet(const char* conf);
	
	int64 m_dump_structure(Layers& layers, const char* name = "", int64 indent = 0);

	virtual void m_set_options(const char* args);

	virtual void m_train_step(Dict xs, Dict ys);

	virtual void m_invoke_forward_postproc(Dict xs, Dict ys, Dict outs, string mode="default");	// 비동기적으로 loss, acc 게산 및 보고
	virtual void m_invoke_validate(int64 nepoch, int64 nbatch);
	virtual void m_invoke_log_train(int64 nepoch, int64 nbatch=-1);
	virtual void m_invoke_train_report(string title);
	virtual void m_invoke_test_report(Dict acc);

	virtual Dict m_strap_default_forward(Dict dat);
	virtual Dict m_unstrap_default_forward(Dict dat);
	virtual Dict m_strap_default_backprop(Dict dat);
	virtual Dict m_unstrap_default_backprop(Dict dat);

	Dict m_seek_wvec_param(string layer_name);
	Dict m_seek_named_param_set(string param_path);

	void m_collect_params_dist(float& mu, float& sigma, int64& param_cnt, int64& near_zero_cnt, float threshold);
	void m_collect_params_info(Value value, int64& param_cnt, int64& near_zero_cnt, float threshold, float& param_sum, float& param_sq_sum);
	void m_collect_params_cost(Value value, float& abs_sum, float& sq_sum);

	virtual void m_train_prepare() {}
};
