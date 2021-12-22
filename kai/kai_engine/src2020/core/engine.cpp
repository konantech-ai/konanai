/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "../core/common.h"
#include "engine.h"
#include "value.h"
#include "random.h"
#include "func_timer.h"
#include "log.h"

#include "../cuda/cuda_conn.cuh"
#include "../cuda/cuda_math.h"

#include "../int_plugin/layer.cuh"
#include "../int_plugin/optimizer.cuh"

// l2decay 지정시 cuda memory 할당 실패로 욜로 사망: 비용 계산 때의 추가 메모리 부담 때문인지 확인하기 위함
#define TEMP_IGNORE_DECAY_COST

const char* Engine::ms_default_option = "{'optimizer':'', 'rand_std':0.030, 'show_maps':False, \
	'l2_decay' : 0, 'l1_decay': 0, 'dump_structure':False, 'macros':{}, 'use_output_layer':True, \
	'need_maps': False, 'learning_rate':0.001, 'clip_grad':0, 'epoch_show_grad_norm':0, \
	'epoch_count':10, 'batch_size': 10, 'learning_rate': 0.001, 'epoch_save':0, \
	'report': 0, 'valid_count': 100, 'show_cnt':3, 'show_params': False, 'batch_report':0, 'debug_bypass_neuralnet': False, \
	'temp_batch_count': 0, 'test_count':0, 'test_batch_size': 0, 'skip_test': False, 'in_batch_valid':0, 'in_batch_save':0, 'in_batch_visualize': 0, \
	'epoch_visualize':0, 'start_epoch':0, 'start_batch' : 0, 'rand_sync':False, 'acc_time': 0, 'pre_train_epoch':3 }";

Engine::Engine(const char* name, Dataset& dataset, const char* conf, const char* options, MacroPack* pMacros)
: m_name(name), m_dataset(dataset) {
	m_pMacros = pMacros;

	Dict sys_options = Value::parse_dict(ms_default_option);
	m_engine_options = Value::merge_dict(sys_options, options);
	m_exec_options = m_engine_options;

	string optimizer = lookup_option("optimizer");
	if (optimizer == "") m_optimizer = Optimizer::check_in_curr_instance(m_exec_options);
	else m_optimizer = Optimizer::check_in_named_instance(optimizer, m_exec_options);

	m_train_channel = data_channel::train;

	m_is_training = false;
	m_block_update = false;

	m_acc_epoch = 0;
	
	m_next_layer_id = 0;
	
	if (string(conf) != "(none)") {
		m_build_neuralnet(conf);

		if ((bool)lookup_option("dump_structure")) {
			m_dump_structure(m_layers, "Neuralnet", 0);
		}
	}

	m_reporter_cont = true;
	m_mu_reporter = new std::mutex();

	m_reporter_seed = Random::dice(100000);

	m_reporter = new std::thread(ms_report, this);
}

Engine::~Engine() {
	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		delete (*it);
	}

	m_reporter_cont = false;
	m_reporter->join();

	m_optimizer->check_out();

	delete m_reporter;
	delete m_mu_reporter;
}

Dict Engine::get_default_options() {
	return Value::parse_dict(ms_default_option);
}

Dict Engine::get_options() {
	return m_exec_options;
}

Value Engine::lookup_option(string key, Value def) {
	return Value::seek_option(m_exec_options, key, def);
}

void Engine::m_set_options(const char* args) {
	m_exec_options = Value::merge_dict(m_engine_options, args);

	m_epoch_count = lookup_option("epoch_count");
	m_valid_count = lookup_option("valid_count");
	m_batch_size = lookup_option("batch_size");
	m_batch_report = lookup_option("batch_report");
	m_temp_batch_count = lookup_option("temp_batch_count");
	m_report = lookup_option("report");
	m_epoch_visualize = lookup_option("epoch_visualize");
	m_epoch_save = lookup_option("epoch_save");
	m_epoch_show_grad_norm = lookup_option("epoch_show_grad_norm");
	m_in_batch_valid = lookup_option("in_batch_valid");
	m_in_batch_visualize = lookup_option("in_batch_visualize");
	m_in_batch_save = lookup_option("in_batch_save");
	m_start_epoch = lookup_option("start_epoch");
	m_start_batch = lookup_option("start_batch");
	m_show_cnt = lookup_option("show_cnt");

	m_rand_sync = lookup_option("rand_sync");
	m_show_params = lookup_option("show_params");

	m_clip_grad = lookup_option("clip_grad");

	m_acc_time = lookup_option("acc_time");

	if (m_valid_count > m_dataset.validate_count()) m_valid_count = m_dataset.validate_count();
}

void Engine::exec_all(const char* args, bool skip_test) {
	FuncTimer::init();
	m_set_options(args);

	train();
	if (!skip_test && !lookup_option("skip_test")) test();
	if (m_show_cnt > 0) visualize(m_show_cnt);
	if (m_show_params) show_param_dist();
	save_param();
	FuncTimer::dump();
}

int64 Engine::m_dump_structure(Layers& layers, const char* name, int64 indent) {
	int64 param_count = 0;

	logger.Print("%*s%s:", indent * 2, "", name);

	for (vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {
		Layer* pLayer = *it;
		param_count += pLayer->dump_structure(indent+1);
	}

	logger.Print("%*s%s parameter count: %lld pms", indent *2, "", name, param_count);

	return param_count;
}

void Engine::m_build_neuralnet(const char* conf) {
	List hconfigs = Value::parse_list(conf);

	Shape shape = m_dataset.input_shape;
	bool seq = m_dataset.input_seq();

	build_hidden_net(hconfigs, shape, seq, m_layers, m_pms);
	
	if ((bool)lookup_option("use_output_layer")) {
		build_output_net(shape, seq, m_layers, m_pms);
	}
}

void Engine::build_hidden_net(List hconfigs, Shape& shape, bool& seq, Layers& layers, List& pms) {
	if (hconfigs.size() > 0 && hconfigs[0].type() != vt::list && hconfigs[0].type() != vt::kint) {
		List temp_conf;
		temp_conf.push_back(hconfigs);
		hconfigs = temp_conf;
	}

	for (List::iterator it = hconfigs.begin(); it != hconfigs.end(); it++) {
		Layer* pPlayer = Layer::CreateLayer(*it, shape, seq, *this);
		layers.push_back(pPlayer);
		pms.push_back(pPlayer->m_param);
	}
}

void Engine::build_output_net(Shape& shape, bool& seq, Layers& layers, List& pms) {
	int64 output_cnt = m_dataset.output_shape.total_size();
	List hconfig = Value::decode_fmt_list("['full',{'width':%d,'actfunc':'none'}]", output_cnt);
	Layer* pPlayer = Layer::CreateLayer(hconfig, shape, seq, *this);
	layers.push_back(pPlayer);
	pms.push_back(pPlayer->m_param);
}

/*
int64 Engine::m_get_batch_count(int64 batch_size) {
	return m_dataset.train_count() / batch_size;
}
*/

/*
void Engine::m_get_train_data(Array<float>& xs, Value& ys) {
	m_dataset.get_data(data_channel::train, xs, ys);
}
*/

void Engine::train() {
	m_time1 = m_time2 = time(NULL);

	string title;
	if (m_train_channel == data_channel::train) title = "train";
	else if (m_train_channel == data_channel::autoencode) title = "autoencode";
	else throw KaiException(KERR_ASSERT);

	if (m_report != 0) logger.Print("Model %s %s started", m_name.c_str(), title.c_str());

	if (m_rand_sync) {
		throw KaiException(KERR_ASSERT); // 랜덤 함수 사용 요소: 데이터 생성 과정, dropout 마스크...
		// 랜덤 함수를 래핑하고 사용 횟수를 카운트하는 방안?
		// curand 사용 문제도 있음...
	}

	m_batch_count = m_dataset.open_data_channel_epochs(m_train_channel, m_epoch_count, m_batch_size);

	if (m_temp_batch_count > 0) m_batch_count = m_temp_batch_count;

	if (m_valid_count > 0) m_dataset.open_data_channel_repeat(data_channel::validate, m_valid_count);
	if (m_show_cnt > 0) m_dataset.open_data_channel_repeat(data_channel::visualize, m_show_cnt);

	m_train_prepare();

#ifdef KAI2021_WINDOWS

#else
	m_batch_count = m_dataset.open_data_channel_epochs(m_train_channel, m_epoch_count, m_batch_size);
#endif
	
	// 현재 m_start_batch 값 주었을 때
	//     m_batch_count = m_dataset.open_data_channel_epochs(m_train_channel, m_epoch_count, m_batch_size);
	// 명령에서 설정한 데이터 갯수와 이용하는 데이터 갯수의 불일치가 발생하는 상황
	// 처리 후 남겨지는 데이터가 문제를 일으킬 수 있으며 에포크 단위의 데이터 분절이 일어나지 않을 수도 있음에 유의

	int64 start_batch = m_start_batch;

	for (int64 epoch = m_start_epoch; epoch < m_epoch_count; epoch++) {
		Dict xs, ys;

		for (int64 nth = start_batch; nth < m_batch_count; nth++) {
			m_dataset.get_data(m_train_channel, xs, ys);

			// 배치 출력을 m_train_step의 m_invoke_forward_postproc() 호출과 별도로 비동기적 호출로 처리해 로직을 명료화시킬 것
			/*
			int64 nepoch = 0;
			int64 nbatch = 0;

			if (nth == m_batch_count - 1 && m_report > 0 and (epoch + 1) % m_report == 0) {
				nepoch = epoch + 1;
			}

			if (m_batch_report > 0 && (nth + 1) % m_batch_report == 0) {
				nbatch = nth + 1;
			}

			m_train_step(xs, ys, nepoch, nbatch); // , acc);

			nepoch = 0;
			*/
			m_train_step(xs, ys); // , acc);

			//throw KaiException(KERR_ASSERT);
			if (m_batch_report > 0 && (nth + 1) % m_batch_report == 0) {
				m_invoke_log_train(epoch, nth);
			}

			if (m_in_batch_valid > 0 && (nth + 1) % m_in_batch_valid == 0) {
				//m_validate(epoch, epoch_count, nth, batch_count, costs, accs, time1, time2);
				m_invoke_validate(epoch + 1, nth+1);
			}

			if (m_in_batch_visualize > 0 && (nth + 1) % m_in_batch_visualize == 0) {
				visualize(0);
			}

			if (m_in_batch_save > 0 && (nth + 1) % m_in_batch_save == 0) {
				char buf[128];
#ifdef KAI2021_WINDOWS
				mysprintf(buf, "-epoch-%04d-batch-%08d", epoch, nth + 1);
#else
				sprintf(buf, "-epoch-%04d-batch-%08d", epoch, nth + 1);
#endif
				save_param((string)buf);
			}
		}

		m_acc_epoch++;

		if (m_epoch_save > 0 && m_acc_epoch % m_epoch_save == 0) {
			save_param();
		}

		if (m_report > 0 && m_acc_epoch % m_report == 0) {
			m_invoke_validate(m_acc_epoch, 0);
		}

		if (m_epoch_visualize > 0 && (m_acc_epoch) % m_epoch_visualize == 0) {
			visualize(0);
		}

		if (m_epoch_show_grad_norm > 0 && m_acc_epoch % m_epoch_show_grad_norm == 0) {
			m_show_n_flush_grad_norm(m_acc_epoch);
		}

		start_batch = 0;
	}

	m_invoke_train_report(title);

	m_dataset.close_data_channel(m_train_channel);

	if (m_valid_count > 0) m_dataset.close_data_channel(data_channel::validate);
	if (m_show_cnt > 0) m_dataset.close_data_channel(data_channel::visualize);
}

void Engine::m_train_step(Dict xs, Dict ys) {
	m_is_training = true;

	Dict outs, G_outs;

	if (lookup_option("debug_bypass_neuralnet")) {
		throw KaiException(KERR_ASSERT);
		/*
		int64 mb_size = xs.axis_size(0);
		assert(!m_dataset.output_seq());
		output = kmath->random_normal(0, 1, m_dataset.output_shape.add_front(mb_size));
		for (Dict::iterator it = m_saved_shape.begin(); it != m_saved_shape.end(); it++) {
			string key = it->first;
			Shape shape = it->second;
			assert(!m_saved_seq[key]);
			Array<float> data = kmath->random_normal(0, 1, shape.add_front(mb_size));
			m_saved_data[key] = data;
		}
		*/
	}
	else {
		//{ CudaConn temp("before forward", NULL); temp.dump_note("before forward"); }

		Dict def_xs = m_strap_default_forward(xs);
		Dict def_out = forward_neuralnet(def_xs, m_layers);
		outs = m_unstrap_default_forward(def_out);
		//{ CudaConn temp("after forward", NULL); temp.dump_note("after forward"); }
		//outs = m_get_estimate(xs, hidden);
	}

	m_invoke_forward_postproc(xs, ys, outs);

	G_outs = backprop_postproc(xs, ys, outs);

	Dict G_def_outs = m_strap_default_backprop(G_outs);
	//Dict G_hidden = m_set_estimate_grad(G_outs);
	//{ CudaConn temp("before backprop", NULL); temp.dump_note("before backprop"); }

	backprop_neuralnet(G_def_outs, m_layers);

	//{ CudaConn temp("after backprop", NULL); temp.dump_note("after backprop"); }

	float grad_norm = m_optimizer->flush_update(m_epoch_show_grad_norm > 0, m_clip_grad);
	if (m_epoch_show_grad_norm > 0) m_grad_norm.push_back(grad_norm);

	m_is_training = false;
}

void Engine::test() {
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
	}
	*/

	m_dataset.close_data_channel(data_channel::test);
}

void Engine::m_invoke_forward_postproc(Dict xs, Dict ys, Dict outs, string mode) {	// 비동기적으로 loss, acc 게산 및 보고
	Dict proc_data;

	proc_data["job"] = "postproc";
	proc_data["xs"] = xs;
	proc_data["ys"] = ys;
	proc_data["outs"] = outs;
	proc_data["mode"] = mode;

	m_mu_reporter->lock();
	m_report_Queue.push(proc_data);
	m_mu_reporter->unlock();
}

void Engine::m_invoke_validate(int64 nepoch, int64 nbatch) {
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

void Engine::m_invoke_log_train(int64 nepoch, int64 nbatch) {
	Dict proc_data;

	proc_data["job"] = "log_train";
	proc_data["nepoch"] = nepoch;
	proc_data["nbatch"] = nbatch;

	m_mu_reporter->lock();
	m_report_Queue.push(proc_data);
	m_mu_reporter->unlock();
}

void Engine::m_invoke_train_report(string title) {
	Dict proc_data;

	proc_data["job"] = "train_report";
	proc_data["title"] = title;

	m_mu_reporter->lock();
	m_report_Queue.push(proc_data);
	m_mu_reporter->unlock();
}

void Engine::m_invoke_test_report(Dict acc) {
	Dict proc_data;

	proc_data["job"] = "test_report";
	proc_data["acc"] = acc;

	m_mu_reporter->lock();
	m_report_Queue.push(proc_data);
	m_mu_reporter->unlock();
}

void Engine::ms_report(void* aux) {
	Engine* pInstance = (Engine*)aux;
	pInstance->m_report_loop();
}

void Engine::m_report_loop() {
	Random::seed(m_reporter_seed);

	CudaConn::SetDevice();

	Dict losses, accs;
	Dict losses_batch, accs_batch;

	while (true) {
		if (!m_reporter_cont) break;

		while (m_report_Queue.size() == 0) {
			m_sleep();
			if (!m_reporter_cont) break;
			continue;
		}
		
		if (!m_reporter_cont) break;

		m_mu_reporter->lock();
		Dict data = m_report_Queue.front();
		m_report_Queue.pop();
		m_mu_reporter->unlock();

		string job = data["job"];
		if (job == "postproc") {
			Dict xs = data["xs"];
			Dict ys = data["ys"];
			Dict outs = data["outs"];
			string mode = data["mode"];
			forward_postproc(xs, ys, outs, mode, losses, accs, losses_batch, accs_batch);
		}
		else if (job == "valid_report") {
			Dict acc = data["acc"];
			int64 nepoch = data["nepoch"];
			int64 nbatch = data["nbatch"];
			valid_report(losses, accs, acc, nepoch, nbatch);
		}
		else if (job == "log_train") {
			int64 nepoch = data["nepoch"];
			int64 nbatch = data["nbatch"];
			log_train(losses_batch, accs_batch, nepoch, nbatch);
		}
		else if (job == "train_report") {
			string title = data["title"];
			train_report(title);
		}
		else if (job == "test_report") {
			Dict acc = data["acc"];
			test_report(acc);
		}
		/*
		else if (job == "postproc_autoencode") {
			Dict xs = data["xs"];
			Dict outs = data["outs"];
			forward_autoencode_postproc(xs, outs, losses, accs);
		}
		*/
		else {
			throw KaiException(KERR_ASSERT);
		}
	}
}

void Engine::m_sleep() {
	Util::nanosleep(100000); // 0.1ms
}

void Engine::forward_postproc(Dict xs, Dict ys, Dict outs, string mode, Dict& losses, Dict& accs, Dict& losses_batch, Dict& accs_batch) {	// 비동기적으로 loss, acc 계산 및 집계
	Dict loss = m_evaluate_loss(xs, ys, outs, mode);
	Dict acc = m_eval_accuracy(xs, ys, outs, mode);

	Value::dict_accumulate(losses, loss);
	Value::dict_accumulate(accs, acc);

	Value::dict_accumulate(losses_batch, loss);
	Value::dict_accumulate(accs_batch, acc);
}

void Engine::valid_report(Dict& losses, Dict& accs, Dict acc, int64 nepoch, int64 nbatch) {
	if (nepoch > 0 || nbatch > 0) {
		time_t now = time(NULL);
		Dict loss_mean = Value::dict_mean_reset(losses);
		Dict acc_mean = Value::dict_mean_reset(accs);

		m_dataset.log_train(nepoch, m_epoch_count, nbatch, m_batch_count, loss_mean, acc_mean, acc, now - m_time2, now - m_time1 + m_acc_time);
		m_time2 = now;
	}
}

void Engine::log_train(Dict& losses, Dict& accs, int64 nepoch, int64 nbatch) {
	// 항상 만족되는 불필요한 조건으로 보여 삭제함, 확인 필요
	//if (nepoch > 0 || nbatch > 0) {
	//}

	time_t now = time(NULL);
	Dict loss_mean = Value::dict_mean_reset(losses);
	Dict acc_mean = Value::dict_mean_reset(accs);

	m_dataset.log_train_batch(nepoch+1, m_epoch_count, nbatch+1, m_batch_count, loss_mean, acc_mean, now - m_time2, now - m_time1 + m_acc_time);
	m_time2 = now;
}

void Engine::train_report(string title) {
	int64 tm_total = time(NULL) - m_time1 + m_acc_time;

	logger.Print("Model %s %s ended in %lld secs", m_name.c_str(), title.c_str(), tm_total);
}

void Engine::test_report(Dict acc) {
	time_t now = time(NULL);

	m_dataset.log_test(m_name, acc, now - m_time1 + m_acc_time, now - m_time2);
}

Dict Engine::m_evaluate_loss(Dict xs, Dict ys, Dict outs, string mode) {
	assert(mode == "default");
	//Dict loss = m_dataset.forward_postproc_sys(xs, estimate, ys);
	Dict loss = m_dataset.forward_postproc_sys(xs, ys, outs);
	forward_extra_cost(loss, m_pms, xs, ys, outs);
	return loss;
}

void Engine::forward_extra_cost(Dict& loss, List pms, Dict xs, Dict ys, Dict outs) {
	if (l2_decay == 0 && l1_decay == 0) return;

#ifdef TEMP_IGNORE_DECAY_COST
	return;
#endif

	float abs_sum = 0, sq_sum = 0;
	m_collect_params_cost(pms, abs_sum, sq_sum);

	if (l2_decay > 0) loss["#L2#"] = l2_decay * sq_sum / 2;
	if (l1_decay > 0) loss["#L1#"] = l1_decay * abs_sum;
}

Dict Engine::m_eval_accuracy(Dict xs, Dict ys, Dict outs, string mode) {
	assert(mode == "default");
	if (outs.size() == 0) {
		Dict def_xs = m_strap_default_forward(xs);
		Dict hidden = forward_neuralnet(def_xs, m_layers);
		outs = m_unstrap_default_forward(hidden);
	}
	return m_dataset.eval_accuracy_sys(xs, ys, outs);
}

/*
Value Engine::m_eval_accuracy(Array<float> xs, Value ys, Array<float> output) {
	if (output.is_empty()) output = forward_neuralnet(xs, m_layers);
	return m_dataset.eval_accuracy_sys(this, xs, ys, output);
}

float Engine::forward_postproc(Dict estimate, Value ys) {
	float loss = m_dataset.forward_postproc_sys(this, output, ys);

	loss += forward_extra_cost(output, ys);

	return loss;
}

Array<float> Engine::m_get_estimate(Array<float> xs, Array<float> output) {
	if (output.is_empty()) output = forward_neuralnet(xs, m_layers);
	return m_dataset.get_estimate_sys(this, output);
}
*/

/*
void Engine::m_validate(int64 epoch, int64 epoch_count, int64 batch, int64 batch_cnt, List costs, List accs, time_t time1, time_t &time2) {
	Dict xs, ys, acc;

	m_dataset.get_data(data_channel::validate, xs, ys);
	CudaConn::GetCudaMem(xs, "x::validate");
	throw KaiException(KERR_ASSERT);
	acc = m_eval_accuracy(xs, ys);
	time_t time3 = time(NULL);
	m_dataset.log_train(epoch, epoch_count, batch, batch_cnt, costs, accs, acc, time3 - time2, time3 - time1);
	time2 = time3;
}
*/

Dict Engine::forward_neuralnet(Dict xs, Layers& layers) {
	Dict hidden = xs;
	for (vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {
		Layer* pLayer = *it;
		hidden = pLayer->forward(hidden);
	}
	return hidden;
}

Dict Engine::backprop_postproc(Dict xs, Dict ys, Dict outs) {
	backprop_extra_cost();
	return m_dataset.backprop_postproc_sys(ys, outs);
}

void Engine::backprop_extra_cost() {
}

Dict Engine::backprop_neuralnet(Dict G_hidden, Layers& layers) {
	for (vector<Layer*>::reverse_iterator it = layers.rbegin(); it != layers.rend(); it++) {
		Layer* pLayer = *it;
		G_hidden = pLayer->backprop(G_hidden);
	}

	return G_hidden;
}

void Engine::visualize(int64 count) {
	logger.Print("Model %s Visualization", m_name.c_str());

	if (count > 0) m_dataset.open_data_channel_once(data_channel::visualize, count);

	assert(!lookup_option("show_maps"));

	Dict xs, ys;
	
	m_dataset.get_data(data_channel::visualize, xs, ys);

	//CudaConn::GetCudaMem(xs, "x::test");

	Dict def_xs = m_strap_default_forward(xs);
	Dict hidden = forward_neuralnet(def_xs, m_layers);
	Dict outs = m_unstrap_default_forward(hidden);

	assert(!lookup_option("show_maps"));

	m_dataset.visualize_main(xs, ys, outs);

	if (count > 0) m_dataset.close_data_channel(data_channel::visualize);
}

void Engine::show_param_dist() {
	if (KArgs::show_image) {
		throw KaiException(KERR_ASSERT);
	}

	float mu, sigma;
	int64 param_cnt, near_zero_cnt;
	
	m_collect_params_dist(mu, sigma, param_cnt, near_zero_cnt, 1.0e-5f);

	logger.Print("Parameter Distribution: mu = %16.9e, sigma = %16.9e", mu, sigma);
	logger.Print("Near zero parameters = %4.1f%%(%lld/%lld)", (float)near_zero_cnt * 100 / (float)param_cnt, near_zero_cnt, param_cnt);
}

void Engine::save_param(string name) {
	string dir = KArgs::param_root;

	Util::mkdir(dir.c_str());
	dir += logger.get_date() + "/";
	Util::mkdir(dir.c_str());

	// 복합 레이어와 숨어 있는 배치정규화 레이어 잘 처리되는지 확인 필요
	char filepath[1024];
	snprintf(filepath, 1024, "%s%s%s_%lld.pmk", dir.c_str(), m_name.c_str(), name.c_str(), m_acc_epoch);

	FILE* fid = Util::fopen(filepath, "wb");

	Value::serial_save(fid, m_pms);
	fclose(fid);
}

void Engine::load_param(int64 epoch, int64 batch) {
	char buf[128];
#ifdef KAI2021_WINDOWS
	mysprintf(buf, "-epoch-%04lld-batch-%08lld", epoch, batch);
#else
	sprintf(buf, "-epoch-%04lld-batch-%08lld", epoch, batch);
#endif
	load_param((string)buf);
}

void Engine::load_param(string name) {
	string filepath = KArgs::param_root + name;

	FILE* fid = Util::fopen(filepath.c_str(), "rb");
	assert(fid != NULL);
	Value::serial_load_params(fid, m_pms);
#ifdef KAI2021_WINDOWS
	int64 read_size = 0;
	throw KaiException(KERR_ASSERT);
#else
	int64 read_size = ftello64(fid);
#endif
	fclose(fid);
}

void Engine::m_collect_params_dist(float& mu, float& sigma, int64& param_cnt, int64& near_zero_cnt, float threshold) {
	param_cnt = near_zero_cnt = 0;

	float param_sum = 0, param_sq_sum = 0;
	m_collect_params_info(m_pms, param_cnt, near_zero_cnt, threshold, param_sum, param_sq_sum);

	mu = param_sum / (float)param_cnt;
	sigma = (float) ::sqrt(param_sq_sum / (float)param_cnt - mu * mu);
}

void Engine::m_collect_params_info(Value value, int64& param_cnt, int64& near_zero_cnt, float threshold, float& param_sum, float& param_sq_sum) {
	if (value.type() == vt::list) {
		List list = value;
		for (List::iterator it = list.begin(); it != list.end(); it++) {
			m_collect_params_info(*it, param_cnt, near_zero_cnt, threshold, param_sum, param_sq_sum);
		}
	}
	else if (value.type() == vt::dict) {
		Dict dict = value;
		for (Dict::iterator it = dict.begin(); it != dict.end(); it++) {
			if (it->first != "w" && it->first != "k") continue;
			m_collect_params_info(it->second, param_cnt, near_zero_cnt, threshold, param_sum, param_sq_sum);
		}
	}
	else if (value.type() == vt::farray) {
		Array<float> params = value;
		param_cnt += params.total_size();
		kmath->acc_nearzero_sum_sqnum(params, param_sum, param_sq_sum, near_zero_cnt, threshold);
	}
}

void Engine::m_collect_params_cost(Value value, float& abs_sum, float& sq_sum) {
	if (value.type() == vt::list) {
		List list = value;
		for (List::iterator it = list.begin(); it != list.end(); it++) {
			m_collect_params_cost(*it, abs_sum, sq_sum);
		}
	}
	else if (value.type() == vt::dict) {
		Dict dict = value;
		for (Dict::iterator it = dict.begin(); it != dict.end(); it++) {
			if (it->first != "w" && it->first != "k") continue;
			m_collect_params_cost(it->second, abs_sum, sq_sum);
		}
	}
	else if (value.type() == vt::farray) {
		Array<float> params = value;
		kmath->acc_abs_n_sq_num(params, abs_sum, sq_sum);
		//abs_sum += params.abs().sum();
		//sq_sum += params.square().sum();
	}
}

void Engine::save_named_wvec_param(string layer_name, string dic_path) {
	string full_path = KArgs::param_root + dic_path;
	Dict param = m_seek_wvec_param(layer_name);
	FILE* fid = Util::fopen(full_path.c_str(), "wb");
	assert(fid != NULL);
	Util::save_dict(fid, param);
	fclose(fid);
}

void Engine::save_named_param_set(string param_path, string dic_path) {
	string full_path = KArgs::param_root + dic_path;
	Dict param_set = m_seek_named_param_set(param_path);
	FILE* fid = Util::fopen(full_path.c_str(), "wb");
	assert(fid != NULL);
	Util::save_dict(fid, param_set);
	fclose(fid);
}

/*
void Engine::load_named_param(string dic_path, Array<float>& arr) {
	string full_path = KArgs::param_root + dic_path;
	FILE* fid = fopen(full_path.c_str(), "rb");
	assert(fid != NULL);
	Util::read_farray(fid, arr);
	fclose(fid);
}
*/

void Engine::load_named_param_set(string dic_path, Dict& pm, string key, bool isAdam, bool nsingle) {
	// 현재 adam 알고리즘의 s, t, n 등 보조 파라미터도 저장하도록 프로그램 구조 수정됨, 단 파일 저장/적재 함수는 미완
	throw KaiException(KERR_ASSERT);
	string full_path = KArgs::param_root + dic_path;
	FILE* fid = Util::fopen(full_path.c_str(), "rb");
	assert(fid != NULL);
	Array<float> parr, sarr, tarr;
	Util::read_farray(fid, parr);
	pm[key] = parr;
	// 학습 과정이 다르면 모멘텀 정보에 일관성이 없으므로 새로 초기화해 쓰는 것이 맞을 듯, 현재 저장 측도 w 배열만 저장 중
	/*
	if (isAdam) {
		Util::read_farray(fid, sarr);
		Util::read_farray(fid, tarr);
		pm["s" + key] = sarr;
		pm["t" + key] = tarr;
		if (nsingle) {
			int64 nstep = Util::read_int(fid);
			pm["n" + key] = nstep;
		}
		else {
			Array<int64> narr;
			Util::read_narray(fid, narr);
			pm["n" + key] = narr;
		}
	}
	*/
	fclose(fid);
}

Dict Engine::m_seek_wvec_param(string layer_name) {
	Dict param;
	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		Layer* pLayer = *it;
		if (pLayer->seek_wvec_param(layer_name, param)) return param;
	}
	throw KaiException(KERR_ASSERT);
	return param;
}

Dict Engine::m_seek_named_param_set(string param_path) {
	Dict param_set;
	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		Layer* pLayer = *it;
		if (pLayer->seek_named_param_set(param_path, param_set)) return param_set;
	}
	throw KaiException(KERR_ASSERT);
	return param_set;
}

Dict Engine::get_saved_data(string name, bool from_ext) {
	if (from_ext) {
		throw KaiException(KERR_ASSERT); // data 값이 배열에서 사전으로 바뀜에 맞게 처리
		//return CudaConn::ToHostArray(m_saved_data[name], "");
#ifdef KAI2021_WINDOWS
		Dict temp;
		return temp;
#else
#endif
	}
	else
		return m_saved_data[name];
}

Dict Engine::get_saved_derv(string name, bool from_ext) {
	if (from_ext) {
		throw KaiException(KERR_ASSERT); // data 값이 배열에서 사전으로 바뀜에 맞게 처리
		//return CudaConn::ToHostArray(m_saved_derv[name], "");
#ifdef KAI2021_WINDOWS
		Dict temp;
		return temp;
#else
#endif
	}
	else {
		return m_saved_derv[name];
	}
}

void Engine::set_saved_data(string name, Dict data, bool from_ext) {
	if (from_ext) {
		throw KaiException(KERR_ASSERT); // data 값이 배열에서 사전으로 바뀜에 맞게 처리
		//m_saved_data[name] = CudaConn::ToCudaArray(data, "");
	}
	else
		m_saved_data[name] = data;
}
void Engine::set_saved_derv(string name, Dict data, bool from_ext) {
	if (from_ext) {
		throw KaiException(KERR_ASSERT); // data 값이 배열에서 사전으로 바뀜에 맞게 처리
		//m_saved_derv[name] = CudaConn::ToCudaArray(data, "");
	}
	else {
		m_saved_derv[name] = data;
	}
}

void Engine::copy_params(Engine& srcEngine, string pairs) {
	Dict dict_pairs = Value::parse_dict((pairs.c_str()));

	for (auto it = dict_pairs.begin(); it != dict_pairs.end(); it++) {
		Layer* pLayerDest = seek_named_layer(it->first);
		Layer* pLayerSrc = srcEngine.seek_named_layer(it->second);

		pLayerDest->copyParam(pLayerSrc);
	}
}

Layer* Engine::seek_named_layer(string name) {
	for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
		Layer* pLayer = (*it)->seek_named_layer(name);
		if (pLayer) return pLayer;
	}

	throw KaiException(KERR_ASSERT);
	return NULL;
}

Dict Engine::m_strap_default_forward(Dict dat) {
	m_saved_data = dat;
	if (dat.find("default") == dat.end()) {
		if (m_dataset.use_custom_data_format()) return dat;
		throw KaiException(KERR_ASSERT);
	}
	return dat["default"];
}

Dict Engine::m_unstrap_default_forward(Dict dat) {
	Dict outs = m_saved_data;
	outs["default"] = dat;
	m_saved_data.clear();
	return outs;
}

Dict Engine::m_strap_default_backprop(Dict dat) {
	m_saved_derv = dat;
	if (dat.find("default") == dat.end()) {
		logger.Print("dat: %s", Value::description(dat).c_str());
		throw KaiException(KERR_ASSERT);
	}
	return dat["default"];
}

Dict Engine::m_unstrap_default_backprop(Dict dat) {
	Dict outs = m_saved_derv;
	outs["default"] = dat;
	m_saved_derv.clear();
	return outs;
}

/*
Dict Engine::m_get_estimate(Dict xs, Dict hidden) { // output 값이 처리안된 값이면 전방처리 발동 + side 처리된 보조 출력 모아주기
	Array<float> net_output;
	if (hidden.size() == 0)
		hidden = forward_neuralnet(xs, m_layers);

	Dict outs = m_saved_data;
	outs["default"] = hidden;

	m_saved_data.clear();

	return outs;
}

Dict Engine::m_set_estimate_grad(Dict G_outs) {
	m_saved_derv = G_outs;
	Dict G_hidden = G_outs["default"];
	return G_hidden;
}
*/

bool Engine::use_custom_data_format() {
	return m_dataset.use_custom_data_format();
}

void Engine::m_show_n_flush_grad_norm(int64 epoch) {
	int64 nsize = m_grad_norm.size();

	float sum = 0, sqsum = 0, max = 0;

	for (int64 n = 0; n < nsize; n++) {
		float norm = (float)m_grad_norm[n];

		sum += norm;
		sqsum += norm * norm;
		if (norm > max) max = norm;
	}

	float mean = sum / nsize;
	float std = sqrt(sqsum / nsize - mean * mean);

	logger.Print("    Epoch %lld grad Norm report: %lld entries, mean: %16.9e, std: %16.9e, max: %16.9e", epoch, nsize, mean, std, max);

	m_grad_norm.clear();
}
