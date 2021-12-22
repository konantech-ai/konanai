/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "gan.h"
#include "../core/func_timer.h"
#include "../core/log.h"
#include "../core/random.h"
#include "../cuda/cuda_math.h"

Gan::Gan(const char* name, GanDataset& dataset, const char* conf, const char* options, MacroPack* pMacros)
: Engine(name, dataset, "(none)", options, pMacros) {
	m_build_neuralnet(conf);

	if ((bool)lookup_option("dump_structure")) {
		m_dump_structure();
	}
}

Gan::~Gan() {
	for (vector<Layer*>::iterator it = m_dlayers.begin(); it != m_dlayers.end(); it++) {
		delete (*it);
	}

	for (vector<Layer*>::iterator it = m_glayers.begin(); it != m_glayers.end(); it++) {
		delete (*it);
	}
}

Layer* Gan::seek_named_layer(string name) {
	for (vector<Layer*>::iterator it = m_glayers.begin(); it != m_glayers.end(); it++) {
		Layer* pLayer = (*it)->seek_named_layer(name);
		if (pLayer) return pLayer;
	}

	for (vector<Layer*>::iterator it = m_dlayers.begin(); it != m_dlayers.end(); it++) {
		Layer* pLayer = (*it)->seek_named_layer(name);
		if (pLayer) return pLayer;
	}

	throw KaiException(KERR_ASSERT);
	return NULL;
}

void Gan::m_build_neuralnet(const char* conf) {
	Dict hconfigs = Value::parse_dict(conf);

	int64 seed_size = lookup_option("seed_size", 1);
	m_seed_shape = Shape(seed_size);

	Shape shape = m_seed_shape;
	bool seq = false;

	build_hidden_net(hconfigs["generator"], shape, seq, m_glayers, m_gpms);

	assert(shape == m_dataset.input_shape);
	assert(seq == m_dataset.input_seq());

	shape = m_dataset.input_shape;

	build_hidden_net(hconfigs["discriminor"], shape, seq, m_dlayers, m_dpms);

	if ((bool)lookup_option("use_output_layer")) {
		build_output_net(shape, seq, m_dlayers, m_dpms);
	}

	assert(shape.total_size() == 1);
}

void Gan::m_dump_structure() {
	logger.Print("Gan structure");

	int64 pm_count = 0;

	pm_count += Engine::m_dump_structure(m_glayers, "Generator", 1);
	pm_count += Engine::m_dump_structure(m_dlayers, "Discriminor", 1);

	logger.Print("Total parameter count: %lld pms", pm_count);
}

int64 Gan::m_generate_mixed_data(Dict& mixed_xs, Dict& mixed_ys, enum data_channel channel, bool training) {
	Dict real_xs, real_ys, fake_xs, fake_ys;

	m_dataset.get_data(channel, real_xs, real_ys);

	Dict real_xs_def = real_xs["default"], real_ys_def = real_ys["default"];

	Array<float> rxs = real_xs_def["data"], rys = real_ys_def["data"];

	int64 size = rxs.axis_size(0);

	m_generate_fake_data(fake_xs, fake_ys, size, training, true);

	mixed_xs = Value::wrap_dict("data", kmath->vstack(rxs, fake_xs["data"]));
	mixed_ys = Value::wrap_dict("data", kmath->vstack(rys, fake_ys["data"]));

	return size;
}

int64 Gan::m_generate_fake_data(Dict& fake_xs, Dict& fake_ys, int64 size, bool training, bool to_mix) {
	Array<float> seed = Random::normal(0, 1, m_seed_shape.add_front(size));
	Dict seed_dict = Value::wrap_dict("data", seed);

	fake_xs = forward_neuralnet(seed_dict, m_glayers);
	fake_ys = Value::wrap_dict("data", kmath->ones(Shape(size, 1), to_mix ? 0.0f : 1.0f));

	return size;
}

void Gan::exec_all(const char* args) {
	FuncTimer::init();

	m_set_options(args);

	m_pre_train_epoch = lookup_option("pre_train_epoch");

	train();

	if (!lookup_option("skip_test")) test();
	if (m_show_cnt > 0) visualize(m_show_cnt);
	if (m_show_params) show_param_dist();
	save_param();
	FuncTimer::dump();
}

void Gan::train() {
	m_time1 = m_time2 = time(NULL);

	string title = "GAN";

	if (m_report != 0) logger.Print("Model %s %s started", m_name.c_str(), title.c_str());

	if (m_rand_sync) {
		throw KaiException(KERR_ASSERT); // 랜덤 함수 사용 요소: 데이터 생성 과정, dropout 마스크...
		// 랜덤 함수를 래핑하고 사용 횟수를 카운트하는 방안?
		// curand 사용 문제도 있음...
	}

	int64 total_epochs = m_epoch_count + m_pre_train_epoch;

	m_batch_count = m_dataset.open_data_channel_epochs(m_train_channel, total_epochs, m_batch_size);

	if (m_temp_batch_count > 0) m_batch_count = m_temp_batch_count;

	logger.Print("Engine::train::batch_count = %d (total: %d)", m_batch_count, total_epochs);

	m_train_prepare();

	for (int64 epoch = 0; epoch < m_pre_train_epoch; epoch++) {
		Dict xs, ys;

		for (int64 nth = 0; nth < m_batch_count; nth++) {
			m_train_discriminor(xs, ys, true);
		}

		m_invoke_log_train(epoch);

		m_show_n_flush_grad_norm(epoch);
	}

	for (int64 epoch = 0; epoch < m_epoch_count; epoch++) {
		Dict xs, ys;

		for (int64 nth = 0; nth < m_batch_count; nth++) {
			m_train_generator(true);
			m_train_discriminor(xs, ys, true);
		}

		m_acc_epoch++;

		if (m_epoch_save > 0 && m_acc_epoch % m_epoch_save == 0) {
			save_param();
		}

		if (m_report > 0 && m_acc_epoch % m_report == 0) {
			m_invoke_log_train(m_acc_epoch);
			//m_invoke_validate(epoch + 1, 0);
			//m_invoke_log_train_batch(epoch + 1, 0);
		}

		if (m_epoch_visualize > 0 && m_acc_epoch % m_epoch_visualize == 0) {
			visualize(5);
		}

		if (m_epoch_show_grad_norm > 0 && m_acc_epoch % m_epoch_show_grad_norm == 0) {
			m_show_n_flush_grad_norm(m_epoch_show_grad_norm);
		}
	}

	m_dataset.close_data_channel(m_train_channel);
}

void Gan::m_train_discriminor(Dict real_xs, Dict real_ys, bool training) {
	Dict mixed_xs, mixed_ys;

	m_generate_mixed_data(mixed_xs, mixed_ys, m_train_channel, training);

	m_is_training = training;

	Dict mixed_out = forward_neuralnet(mixed_xs, m_dlayers);

	Dict xs = m_unstrap_default_forward(mixed_xs);
	Dict ys = m_unstrap_default_forward(mixed_ys);
	Dict outs = m_unstrap_default_forward(mixed_out);

	m_invoke_forward_postproc(xs, ys, outs, "discriminor");

	Dict G_outs = backprop_postproc(xs, ys, outs);
	Dict G_def_outs = m_strap_default_backprop(G_outs);

	backprop_neuralnet(G_def_outs, m_dlayers);

	float grad_norm = m_optimizer->flush_update(m_epoch_show_grad_norm > 0, m_clip_grad);
	if (m_epoch_show_grad_norm > 0) m_grad_norm.push_back(grad_norm);

	m_is_training = false;
}

void Gan::m_train_generator(bool training) {
	Dict fake_xs, fake_ys;

	m_is_training = true;

	m_generate_fake_data(fake_xs, fake_ys, 2*m_batch_size, training, false);

	Dict fake_out = forward_neuralnet(fake_xs, m_dlayers);

	Dict xs = m_unstrap_default_forward(fake_xs);
	Dict ys = m_unstrap_default_forward(fake_ys);
	Dict outs = m_unstrap_default_forward(fake_out);

	m_invoke_forward_postproc(xs, ys, outs, "generator");

	Dict G_outs = backprop_postproc(xs, ys, outs);
	Dict G_def_outs = m_strap_default_backprop(G_outs);

	m_block_update = true;

	Dict G_fake_xs = backprop_neuralnet(G_def_outs, m_dlayers);

	m_block_update = false;

	backprop_neuralnet(G_fake_xs, m_glayers);

	float grad_norm = m_optimizer->flush_update(m_epoch_show_grad_norm > 0, m_clip_grad);
	if (m_epoch_show_grad_norm > 0) m_grad_norm.push_back(grad_norm);

	m_is_training = false;
}

void m_dump_dict_keys(Dict xs) {
	logger.PrintWait("keys(xs):");
	for (Dict::iterator it = xs.begin(); it != xs.end(); it++) logger.PrintWait(" %s", it->first.c_str());
	logger.Print("");
}

Dict Gan::m_evaluate_loss(Dict xs, Dict ys, Dict outs, string mode) {
	if (mode != "discriminor" && mode != "generator") {
		logger.Print("strange mode = %s in GAN", mode.c_str());
		throw KaiException(KERR_ASSERT);
	}
		
	Dict loss = m_dataset.forward_postproc_sys(xs, ys, outs);
	forward_extra_cost(loss, m_pms, xs, ys, outs);
	loss[mode] = loss["default"];
	loss.erase("default");

	return loss;
}

Dict Gan::m_eval_accuracy(Dict xs, Dict ys, Dict outs, string mode) {
	if (outs.size() == 0) {
		logger.Print("m_eval_accuracy(mode=%s)", mode.c_str());
		throw KaiException(KERR_ASSERT);
		Dict def_xs = m_strap_default_forward(xs);
		Dict hidden = forward_neuralnet(def_xs, m_layers);
		outs = m_unstrap_default_forward(hidden);
	}

	if (mode != "discriminor" && mode != "generator") {
		logger.Print("m_eval_accuracy(mode=%s)", mode.c_str());
		throw KaiException(KERR_ASSERT);
	}

	Dict acc = m_dataset.eval_accuracy_sys(xs, ys, outs);
	acc[mode] = acc["default"];
	acc.erase("default");
	return acc;
}

Dict Gan::m_forward_process(enum data_channel channel) {
	Dict mixed_xs, mixed_ys;
	
	int64 size = m_generate_mixed_data(mixed_xs, mixed_ys, channel, false);

	Dict mixed_out = forward_neuralnet(mixed_xs, m_dlayers);

	Dict xs = m_unstrap_default_forward(mixed_xs);
	Dict ys = m_unstrap_default_forward(mixed_ys);
	Dict outs = m_unstrap_default_forward(mixed_out);

	Dict acc_discriminor = m_eval_accuracy(xs, ys, outs, "discriminor");

	Dict fake_xs, fake_ys;

	m_generate_fake_data(fake_xs, fake_ys, size, false, false);

	Dict fake_out = forward_neuralnet(fake_xs, m_dlayers);

	xs = m_unstrap_default_forward(fake_xs);
	ys = m_unstrap_default_forward(fake_ys);
	outs = m_unstrap_default_forward(fake_out);

	Dict acc_generator = m_eval_accuracy(xs, ys, outs, "generator");

	Dict acc;
	acc["discriminor"] = acc_discriminor["discriminor"];
	acc["generator"] = acc_generator["generator"];

	return acc;
}

void Gan::m_invoke_validate(int64 nepoch, int64 nbatch) {
	Dict acc = m_forward_process(data_channel::validate);

	Dict proc_data;

	proc_data["job"] = "valid_report";
	proc_data["acc"] = acc;
	proc_data["nepoch"] = nepoch;
	proc_data["nbatch"] = nbatch;

	m_mu_reporter->lock();
	m_report_Queue.push(proc_data);
	m_mu_reporter->unlock();
}

void Gan::test() {
	m_time2 = time(NULL);

	m_dataset.open_data_channel_all(data_channel::test);

	Dict acc = m_forward_process(data_channel::test);

	m_invoke_test_report(acc);

	m_dataset.close_data_channel(data_channel::test);
}

void Gan::visualize(int64 count) {
	logger.Print("Model %s Visualization", m_name.c_str());

	if (count > 0) m_dataset.open_data_channel_once(data_channel::visualize, count);

	Dict mixed_xs, mixed_ys;

	m_generate_mixed_data(mixed_xs, mixed_ys, data_channel::visualize, false);

	GDP()->visualize(this, mixed_xs, mixed_ys);

	if (count > 0) m_dataset.close_data_channel(data_channel::visualize);
}

void Gan::save_param(string name) {
	string dir = KArgs::param_root;
	Util::mkdir(dir.c_str());

	// 복합 레이어와 숨어 있는 배치정규화 레이어 잘 처리되는지 확인 필요
	char filepath[1024];
	snprintf(filepath, 1024, "%s%s%s_%lld.pmk", dir.c_str(), m_name.c_str(), name.c_str(), m_acc_epoch);
	FILE* fid = Util::fopen(filepath, "wb");
	Value::serial_save(fid, m_gpms);
	Value::serial_save(fid, m_dpms);
	fclose(fid);
}

void Gan::load_param(int64 epoch, string name) {
	string dir = KArgs::param_root;
	Util::mkdir(dir.c_str());

	m_acc_epoch = epoch;

	char filepath[1024];
	snprintf(filepath, 1024, "%s%s%s_%lld.pmk", dir.c_str(), m_name.c_str(), name.c_str(), m_acc_epoch);
	FILE* fid = Util::fopen(filepath, "rb");
	Value::serial_load(fid, m_gpms);
	Value::serial_load(fid, m_dpms);
	fclose(fid);
}
