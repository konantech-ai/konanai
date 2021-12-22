/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/engine.h"
#include "../core/dataset.h"
#include "../core/macro_pack.h"

#include "../int_plugin/layer.cuh"

class Gan : public Engine {
public:
	Gan(const char* name, GanDataset& dataset, const char* conf, const char* options = NULL, MacroPack* pMacros = NULL);
	virtual ~Gan();

	virtual void exec_all(const char* args);
	virtual void test();
	virtual void visualize(int64 count);

	virtual void save_param(string name = "");
	virtual void load_param(int64 epoch, string name = "");

	virtual Layer* seek_named_layer(string name);

protected:
	Shape m_seed_shape;

	Layers m_glayers;
	Layers m_dlayers;
	List m_gpms;
	List m_dpms;

	string m_train_mode;

protected:
	int64 m_pre_train_epoch;

	GanDataset* GDP() { return (GanDataset*)&m_dataset; }

	void m_build_neuralnet(const char* conf);
	void m_dump_structure();

	virtual void train();
	//virtual void m_train_step(Dict xs, Dict ys, int64 nepoch, int64 nbatch);

	virtual void m_invoke_validate(int64 nepoch, int64 nbatch);

	virtual Dict m_evaluate_loss(Dict xs, Dict ys, Dict outs, string mode = "default");
	virtual Dict m_eval_accuracy(Dict xs, Dict ys, Dict outs = Dict(), string mode = "default");

protected:
	void m_train_discriminor(Dict real_xs, Dict real_ys, bool training);
	void m_train_generator(bool training);

	int64 m_generate_mixed_data(Dict& mixed_xs, Dict& mixed_ys, enum data_channel channel, bool training);
	int64 m_generate_fake_data(Dict& fake_xs, Dict& fake_ys, int64 size, bool training, bool to_mix);

	Dict m_forward_process(enum data_channel channel);
};
