/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/engine.h"
#include "../core/dataset.h"
#include "../core/macro_pack.h"

#include "../int_plugin/layer.cuh"

class Autoencoder : public Engine {
public:
	Autoencoder(const char* name, AutoencodeDataset& dataset, const char* conf, const char* options = NULL, MacroPack* pMacros = NULL);
	virtual ~Autoencoder();

	void exec_all(const char* args="", bool skip_test=false);
	void exec_autoencode(const char* args="");
	void exec_suprtvised(const char* args="", bool skip_test=false);
	//void autoencode(const char* args = "");

	virtual void test();

	virtual void visualize(int64 count);
	virtual void visualize_autoencode(int64 count);

	virtual void save_param(string name = "");
	virtual void load_param(string name = "");

	virtual Layer* seek_named_layer(string name);

protected:
	bool m_fix_encoder;

	Layers m_elayers;
	Layers m_dlayers;

	List m_epms;
	List m_dpms;

protected:
	AutoencodeDataset* ADP() { return (AutoencodeDataset*)&m_dataset; }

	void m_build_neuralnet(const char* conf);
	void m_dump_structure();

	virtual Dict forward_neuralnet(Dict xs, Layers& layers);
	virtual Dict backprop_neuralnet(Dict G_outs, Layers& layers);

	//virtual int m_get_batch_count(int batch_size);
	//virtual void m_get_train_data(Dict& xs, Dict& ys);
	//virtual void m_train_step(Dict xs, Dict ys, int nepoch, int nbatch);

	virtual Dict m_evaluate_loss(Dict xs, Dict ys, Dict outs, string mode = "default");
	virtual Dict m_eval_accuracy(Dict xs, Dict ys, Dict outs = Dict(), string mode = "default");

	virtual void m_train_prepare();

	virtual void forward_extra_cost(Dict& loss, Dict xs, Dict ys, Dict outs);
	virtual void backprop_autoencode_extra_cost() {}

	//virtual void m_invoke_forward_postproc(Dict xs, Dict ys, Dict outs);	// 비동기적으로 loss, acc 게산 및 보고

	//virtual void forward_postproc(Dict xs, Dict ys, Dict outs, Dict& losses, Dict& accs);
	virtual Dict backprop_postproc(Dict xs, Dict ys, Dict outs);

	virtual void m_invoke_validate(int nepoch, int nbatch);
};

class AutoencoderHash : public Autoencoder {
public:
	AutoencoderHash(const char* name, AutoencodeDataset& dataset, const char* conf, const char* options = NULL, MacroPack* pMacros = NULL);
	virtual ~AutoencoderHash();

	void semantic_hasing_index(int64 batch_size=-1);
	void semantic_hasing_search(int64 count=3, int64 max_rank =5);

protected:
	Array<float> m_index_code;
	Array<int64> m_label;
	Array<float> m_xs_dat;
#ifdef DEBUG_HASH
	map<int64, vector<int64>> m_hash_tab;
#else
	//Array<int64> m_hash;
#endif
};
