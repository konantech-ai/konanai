/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "autoencoder.h"
#include "../core/log.h"
#include "../cuda/cuda_math.h"

Autoencoder::Autoencoder(const char* name, AutoencodeDataset& dataset, const char* conf, const char* options, MacroPack* pMacros)
: Engine(name, dataset, "(none)", options, pMacros) {
	m_fix_encoder = lookup_option("fix_encoder", false);

	m_build_neuralnet(conf);

	if ((bool)lookup_option("dump_structure")) {
		m_dump_structure();
	}
}

Autoencoder::~Autoencoder() {
	for (vector<Layer*>::iterator it = m_elayers.begin(); it != m_elayers.end(); it++) {
		delete (*it);
	}

	for (vector<Layer*>::iterator it = m_dlayers.begin(); it != m_dlayers.end(); it++) {
		delete (*it);
	}
}

void Autoencoder::m_build_neuralnet(const char* conf) {
	Dict hconfigs = Value::parse_dict(conf);

	Shape shape = m_dataset.input_shape;
	bool seq = m_dataset.input_seq();

	build_hidden_net(hconfigs["encoder"], shape, seq, m_elayers, m_epms);

	Shape code_shape = shape;
	bool code_seq = seq;

	build_hidden_net(hconfigs["decoder"], shape, seq, m_dlayers, m_dpms);

	assert(m_dataset.input_shape == shape);

	if (hconfigs.find("supervised") != hconfigs.end()) {
		build_hidden_net(hconfigs["supervised"], code_shape, code_seq, m_layers, m_pms);

		if ((bool)lookup_option("use_output_layer")) {
			build_output_net(code_shape, code_seq, m_layers, m_pms);
		}
	}
}

void Autoencoder::m_dump_structure() {
	logger.Print("Autoencoder structure");

	int64 pm_count = 0;

	pm_count += Engine::m_dump_structure(m_elayers, "Encoder", 1);
	pm_count += Engine::m_dump_structure(m_dlayers, "Decoder", 1);

	if (m_layers.size() > 0) {
		pm_count += Engine::m_dump_structure(m_layers, "Supervised", 1);
	}

	logger.Print("Total parameter count: %lld pms", pm_count);
}

void Autoencoder::exec_all(const char* args, bool skip_test) {
	exec_autoencode(args);
	exec_suprtvised(args, skip_test);
}

void Autoencoder::exec_autoencode(const char* args) {
	m_train_channel = data_channel::autoencode;
	Engine::exec_all(args, true);
}

void Autoencoder::exec_suprtvised(const char* args, bool skip_test) {
	m_train_channel = data_channel::train;
	Engine::exec_all(args, skip_test);
}

void Autoencoder::m_train_prepare() {
}

Dict Autoencoder::forward_neuralnet(Dict xs, Layers& layers) {
	if (m_train_channel == data_channel::autoencode) {
		Dict code = Engine::forward_neuralnet(xs, m_elayers);
		Dict repl = Engine::forward_neuralnet(code, m_dlayers);
		return repl;
	}
	else if (m_train_channel == data_channel::train) {
		Dict code = Engine::forward_neuralnet(xs, m_elayers);
		Dict out = Engine::forward_neuralnet(code, m_layers);
		return out;
	}
	else {
		throw KaiException(KERR_ASSERT);
		return Dict();
	}
}

Dict Autoencoder::backprop_neuralnet(Dict G_outs, Layers& layers) {
	if (m_train_channel == data_channel::autoencode) {
		Dict G_code = Engine::backprop_neuralnet(G_outs, m_dlayers);
		Dict G_xs = Engine::backprop_neuralnet(G_code, m_elayers);
		return G_xs;
	}
	else if (m_train_channel == data_channel::train) {
		Dict G_code = Engine::backprop_neuralnet(G_outs, m_layers);
		if (m_fix_encoder) {
			m_block_update = true;
			Dict G_xs = Engine::backprop_neuralnet(G_code, m_elayers);
			m_block_update = false;
			return G_xs;
		}
		else {
			Dict G_xs = Engine::backprop_neuralnet(G_code, m_elayers);
			return G_xs;
		}
	}
	else {
		throw KaiException(KERR_ASSERT);
		return Dict();
	}
}

Dict Autoencoder::backprop_postproc(Dict xs, Dict ys, Dict outs) {
	if (m_train_channel == data_channel::autoencode) {
		backprop_extra_cost();
		return m_dataset.backprop_postproc_autoencode_sys(xs, outs);
	}
	else if (m_train_channel == data_channel::train) {
		return Engine::backprop_postproc(xs, ys, outs);
	}
	else {
		throw KaiException(KERR_ASSERT);
		return Dict();
	}
}

Dict Autoencoder::m_evaluate_loss(Dict xs, Dict ys, Dict outs, string mode) {
	assert(mode == "default");
	if (m_train_channel == data_channel::autoencode) {
		Dict loss = m_dataset.forward_postproc_autoencode_sys(xs, outs);
		forward_extra_cost(loss, xs, ys, outs);
		return loss;
	}
	else if (m_train_channel == data_channel::train) {
		return Engine::m_evaluate_loss(xs, ys, outs);
	}
	else {
		throw KaiException(KERR_ASSERT);
		return Dict();
	}
}

Dict Autoencoder::m_eval_accuracy(Dict xs, Dict ys, Dict outs, string mode) {
	assert(mode == "default");
	if (m_train_channel == data_channel::autoencode) {
		if (outs.size() == 0) {
			Dict def_xs = m_strap_default_forward(xs);
			Dict def_repl = forward_neuralnet(def_xs, m_elayers);
			outs = m_unstrap_default_forward(def_repl);
		}
		assert(outs.size() > 0);
		return m_dataset.eval_accuracy_autoencode_sys(xs, outs);
	}
	else if (m_train_channel == data_channel::train) {
		return Engine::m_eval_accuracy(xs, ys, outs);
	}
	else {
		throw KaiException(KERR_ASSERT);
		return Dict();
	}
}

void Autoencoder::forward_extra_cost(Dict& loss, Dict xs, Dict ys, Dict outs) {
	if (m_train_channel == data_channel::autoencode) {
		Engine::forward_extra_cost(loss, m_epms, xs, ys, outs);
		Engine::forward_extra_cost(loss, m_dpms, xs, ys, outs);
	}
	else if (m_train_channel == data_channel::train) {
		Engine::forward_extra_cost(loss, m_epms, xs, ys, outs);
		Engine::forward_extra_cost(loss, m_pms, xs, ys, outs);
	}
	else {
		throw KaiException(KERR_ASSERT);
	}
}

void Autoencoder::m_invoke_validate(int nepoch, int nbatch) {
	Dict xs, ys;

	m_dataset.get_data(data_channel::validate, xs, ys);
	Dict def_xs = m_strap_default_forward(xs);
	Dict hidden = forward_neuralnet(def_xs, m_layers);
	Dict outs = m_unstrap_default_forward(hidden);
	Dict acc = m_eval_accuracy(xs, ys, outs);

	Dict proc_data;

	proc_data["job"] = "valid_report";
	proc_data["acc"] = acc;
	proc_data["nepoch"] = nepoch;
	proc_data["nbatch"] = nbatch;

	m_mu_reporter->lock();
	m_report_Queue.push(proc_data);
	m_mu_reporter->unlock();
}

void Autoencoder::test() {
	if (m_train_channel == data_channel::train) {
		Engine::test();
		return;
	}
	else if (m_train_channel != data_channel::autoencode) {
		throw KaiException(KERR_ASSERT);
	}

	Dict xs, ys, outs;

	m_time2 = time(NULL);

	int64 batch_size = lookup_option("batch_size");

	int64 test_count = lookup_option("test_count");
	int64 test_batch_size = lookup_option("test_batch_size");

	if (test_count == 0) test_count = m_dataset.test_count();
	if (test_batch_size == 0) test_batch_size = batch_size;

	int64 test_batch_count = test_count / test_batch_size;
	m_dataset.open_data_channel_batchs(data_channel::test, test_batch_count, test_batch_size);

	Dict accs;

	for (int64 i = 0; i < test_batch_count; i++) {
		m_dataset.get_data(data_channel::test, xs, ys);
		//CudaConn::GetCudaMem(xs, "x::test");
		Dict acc = m_eval_accuracy(xs, ys);
		Value::dict_accumulate(accs, acc);
	}
	Dict acc_mean = Value::dict_mean_reset(accs);
	m_invoke_test_report(acc_mean);

	/*
	if (test_count == 0 && test_batch_size == 0) {
		m_dataset.open_data_channel_all(data_channel::test);
		m_dataset.get_data(data_channel::test, xs, ys);
		//CudaConn::GetCudaMem(xs, "x::test");
		Dict acc = m_eval_accuracy(xs, ys);
		m_invoke_test_report(acc);
	}
	else {
		if (test_count == 0) test_count = m_dataset.test_count();

		int test_batch_count = test_count / test_batch_size;
		m_dataset.open_data_channel_batchs(data_channel::test, test_batch_count, test_batch_size);

		Dict accs;

		for (int i = 0; i < test_batch_count; i++) {
			m_dataset.get_data(data_channel::test, xs, ys);
			//CudaConn::GetCudaMem(xs, "x::test");
			Dict acc = m_eval_accuracy(xs, ys);
			Value::dict_accumulate(accs, acc);
		}
		Dict acc_mean = Value::dict_mean_reset(accs);
		m_invoke_test_report(acc_mean);
	}
	*/

	m_dataset.close_data_channel(data_channel::test);
}

void Autoencoder::visualize(int64 count) {
	if (m_train_channel == data_channel::train) {
		Engine::visualize(count);
	}
	else if (m_train_channel == data_channel::autoencode) {
		visualize_autoencode(count);
	}
	else {
		throw KaiException(KERR_ASSERT);
	}
}

void Autoencoder::visualize_autoencode(int64 count) {
	logger.Print("Model %s Autoencoding Visualization", m_name.c_str());

	Dict xs, ys, output;

	if (count > 0) m_dataset.open_data_channel_once(data_channel::visualize, count);

	m_dataset.get_data(data_channel::visualize, xs, ys);

	Dict def_xs = m_strap_default_forward(xs);
	Dict code = Engine::forward_neuralnet(def_xs, m_elayers);
	Dict repl = Engine::forward_neuralnet(code, m_dlayers);
	Dict outs = m_unstrap_default_forward(repl);

	assert(!lookup_option("show_maps"));

	ADP()->visualize_autoencode(xs, code, repl, outs, ys);

	if (count > 0) m_dataset.close_data_channel(data_channel::visualize);
}

void Autoencoder::save_param(string name) {
	string dir = KArgs::param_root;
	Util::mkdir(dir.c_str());

	// 복합 레이어와 숨어 있는 배치정규화 레이어 잘 처리되는지 확인 필요
	char filepath[1024];
	snprintf(filepath, 1024, "%s%s%s_%lld.pmk", dir.c_str(), m_name.c_str(), name.c_str(), m_acc_epoch);
	FILE* fid = Util::fopen(filepath, "wb");
	Value::serial_save(fid, m_epms);
	Value::serial_save(fid, m_dpms);
	Value::serial_save(fid, m_pms);
	fclose(fid);
}

void Autoencoder::load_param(string name) {
	throw KaiException(KERR_ASSERT);
}

Layer* Autoencoder::seek_named_layer(string name) {
	for (vector<Layer*>::iterator it = m_elayers.begin(); it != m_elayers.end(); it++) {
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

AutoencoderHash::AutoencoderHash(const char* name, AutoencodeDataset& dataset, const char* conf, const char* options, MacroPack* pMacros)
	: Autoencoder(name, dataset, conf, options, pMacros) {
}

AutoencoderHash::~AutoencoderHash() {
}

void AutoencoderHash::semantic_hasing_index(int64 batch_size) {
	Dict xs, ys;

	if (batch_size < 0) batch_size = m_dataset.train_count();

	int64 batch_count = m_dataset.open_data_channel_epochs(data_channel::train, 1, batch_size);

	assert(batch_count == 1);

	m_dataset.get_data(data_channel::train, xs, ys);

	Dict def_xs = m_strap_default_forward(xs);
	Dict def_ys = m_strap_default_forward(ys);
	Dict def_code = Engine::forward_neuralnet(def_xs, m_elayers);
	
#ifdef DEBUG_HASH
	Array<float> code_data = def_code["data"];
	Array<float> y_data = def_ys["data"];

	Array<int64> hash_cuda = kmath->get_hash_idx(code_data);
	Array<int64> label_cuda = kmath->argmax(y_data, 0);

	m_index_code = CudaConn::ToHostArray(code_data, "code");
	m_label = CudaConn::ToHostArray(label_cuda, "label");

	Array<int64> hash_idx = CudaConn::ToHostArray(hash_cuda, "hash");

	for (int64 m = 0; m < batch_size; m++) {
		int64 hash_key = hash_idx[Idx(m)];
		if (m_hash_tab.find(hash_key) == m_hash_tab.end()) {
			m_hash_tab[hash_key] = vector<int64>();
		}
		m_hash_tab[hash_key].push_back(m);
	}
#else
	Array<float> y_data = def_ys["data"];

	m_index_code = def_code["data"];
	m_label = kmath->argmax(y_data, 0);
	m_xs_dat = def_xs["data"];
	m_xs_dat = m_xs_dat.to_host();
#endif

	m_dataset.close_data_channel(data_channel::train);
}

void AutoencoderHash::semantic_hasing_search(int64 count, int64 max_rank) {
	Dict xs, ys;

	m_dataset.open_data_channel_once(data_channel::validate, count);

	m_dataset.get_data(data_channel::validate, xs, ys);

	Dict def_xs = m_strap_default_forward(xs);
	Dict def_ys = m_strap_default_forward(ys);
	Dict def_code = Engine::forward_neuralnet(def_xs, m_elayers);

	Dict repl = Engine::forward_neuralnet(def_code, m_dlayers);
	Dict outs = m_unstrap_default_forward(repl);

#ifdef DEBUG_HASH
	Array<float> code_data = def_code["data"];
	Array<float> y_data = def_ys["data"];

	Array<int64> hash_cuda = kmath->get_hash_idx(code_data);
	Array<int64> label_cuda = kmath->argmax(y_data, 0);

	Array<float> code = CudaConn::ToHostArray(code_data, "code");
	Array<int64> label = CudaConn::ToHostArray(label_cuda, "label");

	Array<int64> hash_idx = CudaConn::ToHostArray(hash_cuda, "hash");

	for (int64 n = 0; n < count; n++) {
		int64 hash_key = hash_idx[Idx(n)];
		if (m_hash_tab.find(hash_key) == m_hash_tab.end()) {
			logger.Print("search result %d: no matched data", hash_key);
		}
		else {
			Array<float> matched_code, curr_code, dot_sum, sorted;
			Array<int64> sort_idx;

			matched_code = m_code.fetch_rows(m_hash_tab[hash_key]);
			curr_code = code[Axis(n, _all_)];
			dot_sum = matched_code.dotsum(curr_code.reshape(Shape(-1)));
			sorted = dot_sum.sort(sortdir::sd_desc, sort_idx);

			logger.Print("search result %d: %lld matched data", n + 1, dot_sum.axis_size(0));

			for (int64 m = 0; m < dot_sum.axis_size(0); m++) {
				if (m == max_rank) break;
				logger.Print("   matched %d-%d: score = %16.9e, labels = (key:%d vs data:%d)", n + 1, m + 1, sorted[Idx(m)], label[Idx(n)], m_label[Idx(sort_idx[Idx(m)])]);
			}
		}
	}
#else
	Array<float> key_code = def_code["data"];
	Array<float> key_y_data = def_ys["data"];

	//Array<int64> key_hash_cuda = kmath->get_hash_idx(key_code_data);
	Array<int64> key_label_cuda = kmath->argmax(key_y_data, 0);

	Array<float> hash_match_point = kmath->get_hash_match_point(key_code, m_index_code);
	Array<float> vector_dist = kmath->get_vector_dist(key_code, m_index_code);

	Array<float> matched = CudaConn::ToHostArray(hash_match_point, "matched");
	Array<float> distance = CudaConn::ToHostArray(vector_dist, "dist");
	Array<int64> key_labels = CudaConn::ToHostArray(key_label_cuda, "key_labels");
	Array<int64> dat_label = CudaConn::ToHostArray(m_label, "dat_label");

	Array<int64> rank1 = kmath->sort_columns(distance, sortdir::sd_asc, 100);
	Array<int64> rank2 = kmath->sort_columns(matched, sortdir::sd_desc, distance, sortdir::sd_asc, 100);
	
	/*
	Dict def_xs = m_strap_default_forward(xs);
	Dict code = Engine::forward_neuralnet(def_xs, m_elayers);
	Dict repl = Engine::forward_neuralnet(code, m_dlayers);
	Dict outs = m_unstrap_default_forward(repl);
	*/

	assert(!lookup_option("show_maps"));

	Array<float> key_dat = def_xs["data"];
	Array<float> repl_dat = repl["data"];

	ADP()->visualize_hash(rank1, rank2, key_labels, dat_label, distance, key_dat.to_host(), repl_dat.to_host(), m_xs_dat);

	// 두 결과에 따른 상위 랭커들을 숫자 값가 그림을 곁들여 비교해 보여준다,
	int n = 0;
	/*
	Array<int64> hash_idx = CudaConn::ToHostArray(hash_cuda, "hash");

	Array<int64> m_hash;
	Array<float> code = CudaConn::ToHostArray(code_data, "code");
	Array<int64> label = CudaConn::ToHostArray(label_cuda, "label");

	Array<int64> hash_idx = CudaConn::ToHostArray(hash_cuda, "hash");

	for (int64 n = 0; n < count; n++) {
		int64 hash_key = hash_idx[Idx(n)];
		if (m_hash_tab.find(hash_key) == m_hash_tab.end()) {
			logger.Print("search result %d: no matched data", hash_key);
		}
		else {
			Array<float> matched_code, curr_code, dot_sum, sorted;
			Array<int64> sort_idx;

			matched_code = m_code.fetch_rows(m_hash_tab[hash_key]);
			curr_code = code[Axis(n, _all_)];
			dot_sum = matched_code.dotsum(curr_code.reshape(Shape(-1)));
			sorted = dot_sum.sort(sortdir::sd_desc, sort_idx);

			logger.Print("search result %d: %lld matched data", n + 1, dot_sum.axis_size(0));

			for (int64 m = 0; m < dot_sum.axis_size(0); m++) {
				if (m == max_rank) break;
				logger.Print("   matched %d-%d: score = %16.9e, labels = (key:%d vs data:%d)", n + 1, m + 1, sorted[Idx(m)], label[Idx(n)], m_label[Idx(sort_idx[Idx(m)])]);
			}
		}
	}
	*/
#endif

	m_dataset.close_data_channel(data_channel::validate);
}
