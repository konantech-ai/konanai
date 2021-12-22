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
#include "dataset.h"
#include "engine.h"
#include "log.h"

#include "../cuda/cuda_conn.cuh"

//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif

string Dataset::ms_img_save_folder = "default";
bool Dataset::ms_img_display_mode = false;

Dataset::Dataset(string name, string mode, bool x_seq, bool y_seq, string datetime) : m_data_idx(NULL) {
	m_name = name;
	m_enumMode = ms_str_to_mode(mode, &m_custom_mode);
	m_gen_dat_mutex = new std::mutex();

	if (datetime == "") {
		datetime = logger.get_datetime();
		m_need_to_load_data_index = false;
	}
	else {
		m_need_to_load_data_index = true;
	}

	string dir = KArgs::data_index_root;
	Util::mkdir(dir.c_str());

	char fname[1024];
#ifdef KAI2021_WINDOWS
	snprintf(fname, 1024, "%s%s-%s", dir.c_str(), m_name.c_str(), datetime.c_str());
#else
	snprintf(fname, 1024, "%s%s-%s.dax", dir.c_str(), m_name.c_str(), datetime.c_str());
#endif
	m_data_index_path = fname;
}

Dataset::~Dataset() {
	for (map< enum data_channel, DataChannel*>::iterator it = m_channels.begin(); it != m_channels.end(); it++) {
		DataChannel* pChannel = it->second;
		delete pChannel;
	}

	delete m_gen_dat_mutex;
}

int64 Dataset::train_count() {
	return m_data_count[data_channel::train];
}

int64 Dataset::test_count() {
	return m_data_count[data_channel::test];
}

int64 Dataset::validate_count() {
	return m_data_count[data_channel::validate];
}

enum loss_mode Dataset::ms_str_to_mode(string mode, string* p_custom_mode) {
	if (mode == "regression") return loss_mode::regression;
	else if (mode == "binary") return loss_mode::binary;
	else if (mode == "classify") return loss_mode::classify;
	else if (mode == "class_idx") return loss_mode::classify_idx;
	else if (mode == "class_1st") return loss_mode::classify_1st;
	else if (mode == "autoencode") return loss_mode::autoencode;
	else {
		assert(p_custom_mode != NULL);
		*p_custom_mode = mode;
		return loss_mode::custom;
	}
}

string Dataset::m_get_mode_str() const {
	if (m_enumMode == loss_mode::regression) return "regression";
	else if (m_enumMode == loss_mode::binary) return "binary";
	else if (m_enumMode == loss_mode::classify) return "classify";
	else if (m_enumMode == loss_mode::classify_idx) return "class_idx";
	else if (m_enumMode == loss_mode::classify_1st) return "class_1st";
	else if (m_enumMode == loss_mode::autoencode) return "autoencode";
	else return m_custom_mode;
}

string Dataset::description() {
	char buff[100];
	int64 tr_cnt = m_data_count[data_channel::train];
	int64 te_cnt = m_data_count[data_channel::test];
	int64 va_cnt = m_data_count[data_channel::validate];
	snprintf(buff, sizeof(buff), "%s(%s, %lld+%lld+%lld)", m_name.c_str(), m_get_mode_str().c_str(), tr_cnt, te_cnt, va_cnt);
	std::string buffAsStdStr = buff;
	return buffAsStdStr;
}

Dict Dataset::forward_postproc_sys(Dict xs, Dict ys, Dict outs) {
	Dict loss;

	if (m_enumMode == loss_mode::custom) {
		loss = forward_postproc(xs, ys, outs, m_custom_mode);
	}
	else {
		Dict y_def, o_def = outs["default"];
		if (m_enumMode != loss_mode::classify_1st) y_def = ys["default"];

		if (CudaConn::UsingCuda())
			loss["default"] = m_forward_postproc_cuda(y_def, o_def, m_enumMode);
		else
			loss["default"] = m_forward_postproc_no_cuda(y_def, o_def, m_enumMode);
	}

	return loss;
}

Dict Dataset::forward_postproc_autoencode_sys(Dict xs, Dict outs) {
	Dict loss;

	Dict x_def = xs["default"];
	Dict o_def = outs["default"];
	if (CudaConn::UsingCuda())
		loss["default"] = m_forward_postproc_cuda(x_def, o_def, loss_mode::autoencode);
	else
		loss["default"] = m_forward_postproc_no_cuda(x_def, o_def, loss_mode::autoencode);

	return loss;
}

float Dataset::forward_postproc_base(Dict y, Dict out, string mode) {
	enum loss_mode enumMode = ms_str_to_mode(mode);

	float loss = 0;

	if (CudaConn::UsingCuda()) {
		loss = m_forward_postproc_cuda(y, out, enumMode);
	}
	else {
		loss = m_forward_postproc_no_cuda(y, out, enumMode);
	}

	return loss;
}

float Dataset::m_forward_postproc_no_cuda(Dict y, Dict out, enum loss_mode enumMode) {
	float loss = 0;

	Array<float> est = out["data"], ans;

	if (output_seq()) est = est.merge_time_axis();

	if (enumMode != loss_mode::classify_1st && enumMode != loss_mode::classify_idx) {
		ans = y["data"];
		if (output_seq()) ans = ans.merge_time_axis();
	}

	if (enumMode == loss_mode::regression) {
		Array<float> diff = est - ans;
		Array<float> square = kmath->square(diff);
		loss = kmath->mean(square);
	}
	else if (enumMode == loss_mode::binary) {
		Array<float> entropy = kmath->sigmoid_cross_entropy_with_logits(ans, est);

		loss = kmath->mean(entropy);
	}
	else if (enumMode == loss_mode::classify) {
		Array<float> entropy = kmath->softmax_cross_entropy_with_logits(ans, est);
		loss = kmath->mean(entropy);
	}
	else if (enumMode == loss_mode::classify_idx) {
		Array<int64> ans = y["wids"];
		if (output_seq()) ans = ans.merge_time_axis();
		Array<float> entropy = kmath->softmax_cross_entropy_with_logits_idx(ans, est);
		loss = kmath->mean(entropy);
	}
	else if (enumMode == loss_mode::classify_1st) {
		Array<float> entropy = kmath->softmax_cross_entropy_with_logits_1st(est);
		loss = kmath->mean(entropy);
	}
	else if (enumMode == loss_mode::autoencode) {
		Array<float> diff = est - ans;
		Array<float> square = kmath->square(diff);
		loss = kmath->mean(square);
		//aux["diff"] = diff;
	}

	return loss;
}

Dict Dataset::backprop_postproc_sys(Dict ys, Dict outs) {
	Dict G_outs;

	if (m_enumMode == loss_mode::custom) {
		G_outs = backprop_postproc(ys, outs, m_custom_mode);
	}
	else {
		Dict y_def, o_def = outs["default"];
		if (m_enumMode != loss_mode::classify_1st) y_def = ys["default"];

		Dict G_o_def;
		if (CudaConn::UsingCuda()) 
			G_o_def = m_backprop_postproc_cuda(y_def, o_def, m_enumMode);
		else
			G_o_def = m_backprop_postproc_no_cuda(y_def, o_def, m_enumMode);

		G_outs["default"] = G_o_def;
	}

	return G_outs;
}

Dict Dataset::backprop_postproc_autoencode_sys(Dict xs, Dict outs) {
	Dict G_outs;

	Dict x_def = xs["default"];
	Dict o_def = outs["default"];
	Dict G_o_def;
	if (CudaConn::UsingCuda())
		G_o_def = m_backprop_postproc_cuda(x_def, o_def, loss_mode::autoencode);
	else
		G_o_def = m_backprop_postproc_no_cuda(x_def, o_def, loss_mode::autoencode);

	G_outs["default"] = G_o_def;

	return G_outs;
}

Dict Dataset::backprop_postproc_base(Dict y, Dict out, string mode) {
	enum loss_mode enumMode = ms_str_to_mode(mode);

	Dict G_hidden;

	if (CudaConn::UsingCuda()) {
		G_hidden = m_backprop_postproc_cuda(y, out, enumMode);
		//CudaConn::GetHostMem(G_hidden, "G_hidden::backprop_postproc_base");
	}
	else {
		G_hidden = m_backprop_postproc_no_cuda(y, out, enumMode);
	}

	return G_hidden;
}

Dict Dataset::m_backprop_postproc_no_cuda(Dict y, Dict out, enum loss_mode enumMode) {
	Array<float> est = out["data"], ans;
	int64 mb_size = 0;

	Shape shape = est.shape();

	if (output_seq()) {
		mb_size = est.axis_size(0);
		est = est.merge_time_axis();
	}

	if (enumMode != loss_mode::classify_1st && enumMode != loss_mode::classify_idx) {
		ans = y["data"];
		if (output_seq()) ans = ans.merge_time_axis();
	}

	Array<float> G_out_data;

	if (enumMode == loss_mode::regression) {
		Array<float> diff = est - ans;
		Shape shape = diff.shape();
		Array<float> g_loss_square = kmath->ones(shape) / (float)shape.total_size();
		Array<float> g_square_diff = diff * 2;
		Array<float> G_square = g_loss_square;
		G_out_data = g_square_diff * G_square;
	}
	else if (enumMode == loss_mode::binary) {
		Shape shape = est.shape();
		Array<float> g_loss_entropy = kmath->ones(shape) / (float)shape.total_size();
		Array<float> g_entropy_output = kmath->sigmoid_cross_entropy_with_logits_derv(ans, est);
		Array<float> G_entropy = g_loss_entropy;
		G_out_data = g_entropy_output * G_entropy;
	}
	else if (enumMode == loss_mode::classify) {
		float G_entropy = 1.0f / (float)est.axis_size(0);
		Array<float> g_entropy_output = kmath->softmax_cross_entropy_with_logits_derv(ans, est);
		G_out_data = g_entropy_output * G_entropy;
	}
	else if (enumMode == loss_mode::classify_idx) {
		Array<int64> ans = y["wids"];
		if (output_seq()) ans = ans.merge_time_axis();
		float G_entropy = 1.0f / (float)est.axis_size(0);
		Array<float> g_entropy_output = kmath->softmax_cross_entropy_with_logits_idx_derv(ans, est);
		G_out_data = g_entropy_output * G_entropy;
	}
	else if (enumMode == loss_mode::classify_1st) {
		//Array<float> entropy = aux["entropy"];
		//float G_entropy = G_loss / (float)entropy.shape().total_size();
		float G_entropy = 1.0f / (float)est.axis_size(0);
		Array<float> g_entropy_output = kmath->softmax_cross_entropy_with_logits_1st_derv(est);
		G_out_data = g_entropy_output * G_entropy;
	}
	else if (enumMode == loss_mode::autoencode) {
		Array<float> diff = est - ans;
		Shape shape = diff.shape();
		Array<float> g_loss_square = kmath->ones(shape) / (float)shape.total_size();
		Array<float> g_square_diff = diff * 2;
		Array<float> G_square = g_loss_square;
		G_out_data = g_square_diff * G_square;
	}

	Dict G_out;
	G_out["data"] = G_out_data.reshape(shape);
	return G_out;
}

Dict Dataset::eval_accuracy_sys(Dict xs, Dict ys, Dict outs) {
	Dict accuracy;

	if (m_enumMode == loss_mode::custom) {
		accuracy = eval_accuracy(xs, ys, outs, m_custom_mode);
	}
	else {
		Dict x_def, y_def, o_def = outs["default"];
		if (m_enumMode != loss_mode::classify_1st) {
			x_def = xs["default"];
			y_def = ys["default"];
		}

		if (CudaConn::UsingCuda()) 
			accuracy["default"] = m_eval_accuracy_cuda(x_def, y_def, o_def, m_enumMode);
		else
			accuracy["default"] = m_eval_accuracy_no_cuda(x_def, y_def, o_def, m_enumMode);
	}

	return accuracy;
}

Dict Dataset::eval_accuracy_autoencode_sys(Dict xs, Dict outs) {
	Dict accuracy;

	Dict x_def = xs["default"];
	Dict o_def = outs["default"];

	if (CudaConn::UsingCuda()) 
		accuracy["autoencode"] = m_eval_accuracy_cuda(x_def, x_def, o_def, loss_mode::autoencode);
	else
		accuracy["autoencode"] = m_eval_accuracy_no_cuda(x_def, x_def, o_def, loss_mode::autoencode);

	return accuracy;
}

float Dataset::eval_accuracy_base(Dict x, Dict y, Dict output, string mode) {
	//CudaConn cuda("Dataset::eval_accuracy_base");

	enum loss_mode enumMode = ms_str_to_mode(mode);

	float accuracy;

	if (CudaConn::UsingCuda()) {
		//x = CudaConn::ToCudaArray(x, "x::eval_accuracy_base");
		//output = CudaConn::ToCudaArray(output, "output::eval_accuracy_base");

		accuracy = m_eval_accuracy_cuda(x, y, output, enumMode);
	}
	else {
		accuracy = m_eval_accuracy_no_cuda(x, y, output, enumMode);
	}

	return accuracy;
}

float Dataset::m_eval_accuracy_no_cuda(Dict x, Dict y, Dict out, enum loss_mode enumMode) {
	float accuracy = 0;

	Array<float> ans, est = out["data"];

	if (output_seq()) est = est.merge_time_axis();

	if (enumMode != loss_mode::classify_1st && enumMode != loss_mode::classify_idx) {
		ans = y["data"];
		if (output_seq()) ans = ans.merge_time_axis();
	}

	if (enumMode == loss_mode::regression) {
		float mse = kmath->mean(kmath->square(est - ans));
		accuracy = (float) 1.0 - (float) sqrt(mse) / kmath->mean(ans);
	}
	else if (enumMode == loss_mode::binary) {
		Array<bool> estimate = est > 0;
		Array<bool> answer = ans > 0.5;
		Array<bool> correct = estimate == answer;
		accuracy = kmath->mean(correct);
	}
	else if (enumMode == loss_mode::classify) {
		Array<int64> estimate = kmath->argmax(est, 0);
		Array<int64> answer = kmath->argmax(ans, 0);
		Array<bool> correct = estimate == answer;
		accuracy = kmath->mean(correct);
	}
	else if (enumMode == loss_mode::classify_idx) {
		Array<int64> estimate = kmath->argmax(est, 0);
		Array<int64> ans = y["wids"];
		if (output_seq()) ans = ans.merge_time_axis();
		Array<bool> correct = estimate == ans;
		accuracy = kmath->mean(correct);
	}
	else if (enumMode == loss_mode::classify_1st) {
		Array<int64> estimate = kmath->argmax(est, 0);
		Array<bool> correct = estimate == 0;
		accuracy = kmath->mean(correct);
	}
	else if (enumMode == loss_mode::autoencode) {
		Array<float> repl = est;
		Array<float> org = x["data"];
		float mse = kmath->mean(kmath->square(repl - org));
		accuracy = (float)1.0 - (float)sqrt(mse) / kmath->mean(org);
	}

	return accuracy;
}

Dict Dataset::forward_postproc(Dict xs, Dict ys, Dict estimate, string mode) {
	throw KaiException(KERR_ASSERT);
	return Dict();
}

Dict Dataset::backprop_postproc(Dict ys, Dict estimate, string mode) {
	throw KaiException(KERR_ASSERT);
	return Dict();
}

Dict Dataset::eval_accuracy(Dict x, Dict y, Dict out, string mode) {
	throw KaiException(KERR_ASSERT);
	return Dict();
}

void Dataset::gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys) {
	m_gen_dat_mutex->lock();

	Dict xs_default, ys_default;

	prepare_minibatch_data(data_idxs, size);

	Value x_to_y;
	List x_to_ys;

	Shape xshape = input_shape;
	
	if (input_seq()) xshape = xshape.add_front(input_timesteps());

	Array<float> xarr(xshape.add_front(size));
	float* px = xarr.data_ptr();

	int64 xsize = xshape.total_size();

	for (int64 n = 0; n < size; n++) {
		if (input_seq())
			gen_seq_xdata(n, data_idxs[n], xsize, px, x_to_y);
		else
			gen_plain_xdata(n, data_idxs[n], xsize, px, x_to_y);
		x_to_ys.push_back(x_to_y);
		px += xsize;
	}


	xs_default["data"] = CudaConn::ToCudaArray(xarr, "xs");

	if (channel == data_channel::autoencode) {
		ys_default = xs_default;
	}
	else {
		Shape yshape = output_shape;
		if (output_seq()) yshape = yshape.add_front(output_timesteps());

		Array<float> yarr(yshape.add_front(size));

		float* py = yarr.data_ptr();

		int64 ysize = yshape.total_size();

		for (int64 n = 0; n < size; n++) {
			x_to_y = x_to_ys[n];
			if (output_seq())
				gen_seq_ydata(n, data_idxs[n], ysize, py, x_to_y);
			else
				gen_plain_ydata(n, data_idxs[n], ysize, py, x_to_y);
			py += ysize;
		}

		ys_default["data"] = CudaConn::ToCudaArray(yarr, "ys");
	}

	xs["default"] = xs_default;
	ys["default"] = ys_default;

	m_gen_dat_mutex->unlock();
}

void Dataset::prepare_minibatch_data(int64* data_idx, int64 size) {
}

void Dataset::gen_plain_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y) {
	float* pmx = m_default_xs.data_ptr() + data_idx * xsize;
	memcpy(px, pmx, xsize * sizeof(float));
}

void Dataset::gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
	float* pmy = m_default_ys.data_ptr() + data_idx * ysize;
	memcpy(py, pmy, ysize * sizeof(float));
}

void Dataset::gen_seq_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y) {
	float* pmx = m_default_xs.data_ptr() + data_idx * xsize;
	memcpy(px, pmx, xsize * sizeof(float));
}

void Dataset::gen_seq_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
	float* pmy = m_default_ys.data_ptr() + data_idx * ysize;
	memcpy(py, pmy, ysize * sizeof(float));
}

string Dataset::get_acc_str(string& acc_keys, Dict acc) {
	bool single = acc.size() == 1;
	bool def_only = single && (acc.begin()->first == "default");
	
	bool gan_both = (acc.find("discriminor") != acc.end()) && (acc.find("generator") != acc.end());

	bool first = true;

	string acc_str;

	for (Dict::iterator it = acc.begin(); it != acc.end(); it++) {
		string key = it->first;
		if (!def_only) {
			if (first) acc_keys = "[" + key;
			else acc_keys += "," + key;
		}

		float acc_value = (float)acc[key];
		if (gan_both) acc_value *= 2;

		string buf(16, '\0');
		int64 written = std::snprintf(&buf[0], buf.size(), "%7.5f", acc_value);
		buf.resize(written);

		if (single) acc_str = buf;
		else if (first) acc_str = "[" + buf;
		else acc_str += "," + buf;

		first = false;
	}

	if (!def_only) acc_keys += "]";
	if (!single) acc_str += "]";

	return acc_str;
}

string Dataset::get_loss_str(Dict loss) {
	float loss_def = loss["default"];

	string buf(16, '\0');
	int64 written = std::snprintf(&buf[0], buf.size(), "%7.5f", loss_def);
	buf.resize(written);

	return buf;
}

void Dataset::log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2) {
	int64 loss_size = loss_mean.size();
	string loss_decay;

	if (loss_mean.find("#L2#") != loss_mean.end() || loss_mean.find("#L1#") != loss_mean.end()) {
		float loss_L1 = 0;
		float loss_L2 = 0;
		
		if (loss_mean.find("#L2#") != loss_mean.end()) {
			loss_L2 = loss_mean["#L2#"];
			loss_size--;
		}

		if (loss_mean.find("#L1#") != loss_mean.end()) {
			loss_L1 = loss_mean["#L1#"];
			loss_size--;
		}

		if (loss_L1 && loss_L2) loss_decay = "(L2:" + to_string(loss_L2) + ",L1:" + to_string(loss_L1) + ")";
		else if (loss_L2) loss_decay = "(L2:" + to_string(loss_L2) + ")";
		else if (loss_L1) loss_decay = "(L1:" + to_string(loss_L1) + ")";
	}

	string loss_str = get_loss_str(loss_mean);

	string acc_keys;
	string acc_means_str = get_acc_str(acc_keys, acc_mean);
	string acc_last_str = get_acc_str(acc_keys, acc);

	if (batch_count == 0) 
		logger.PrintWait("    Epoch %d/%d: ", epoch, epoch_count);
	else
		logger.PrintWait("    Batch %lld/%lld(in Epoch %lld): ", batch, batch_count, epoch);

	logger.Print("loss=%s%s, accuracy%s=%s/%s (%lld/%lld secs)", loss_str.c_str(), loss_decay.c_str(), acc_keys.c_str(), acc_means_str.c_str(), acc_last_str.c_str(), tm1, tm2);
}

void Dataset::log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2) {
	string loss_str = get_loss_str(loss_mean);

	string acc_keys;
	string acc_means_str = get_acc_str(acc_keys, acc_mean);

	if (batch_count <= 0) 
		logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
	else
		logger.PrintWait("    Batch %lld/%lld(in Epoch %d): ", batch, batch_count, epoch);

	logger.Print("loss=%s, accuracy%s=%ss (%lld/%lld secs)", loss_str.c_str(), acc_keys.c_str(), acc_means_str.c_str(), tm1, tm2);
}

void Dataset::log_test(string name, Dict acc, int64 tm1, int64 tm2) {
	string acc_keys;
	string acc_str = get_acc_str(acc_keys, acc);

	logger.Print("Model %s test report: accuracy%s = %s, (%lld/%lld secs)", name.c_str(), acc_keys.c_str(), acc_str.c_str(), tm2, tm1);
	logger.Print("");
}

void Dataset::m_shuffle_index(int64 size, float tr_ratio, float va_ratio, float vi_ratio) {
	m_data_cnt = size;
	
	int64 tr_cnt = (int64)((float)m_data_cnt * tr_ratio);
	int64 va_cnt = (int64)((float)m_data_cnt * va_ratio);
	int64 vi_cnt = (int64)((float)m_data_cnt * vi_ratio);

	m_data_count[data_channel::train] = tr_cnt;
	m_data_count[data_channel::validate] = va_cnt;
	m_data_count[data_channel::visualize] = vi_cnt;
	m_data_count[data_channel::test] = m_data_cnt - tr_cnt - va_cnt - vi_cnt;

	m_data_begin[data_channel::train] = 0;
	m_data_begin[data_channel::validate] = tr_cnt;
	m_data_begin[data_channel::visualize] = tr_cnt + va_cnt;
	m_data_begin[data_channel::test] = tr_cnt + va_cnt + vi_cnt;

	m_data_idx_arr = kmath->arange(m_data_cnt);
	m_data_idx = m_data_idx_arr.data_ptr();

	if (m_need_to_load_data_index) {
		load_data_index();
		m_need_to_load_data_index = false;
	}
	else {
		kmath->shuffle(m_data_cnt, m_data_idx);
		save_data_index();
	}
}

#ifdef KAI2021_WINDOWS
void Dataset::save_data_index(string suffix) {
	static mutex mu_save_data_indedx;

	FILE* fid = NULL;
	mu_save_data_indedx.lock();
	try {
		string datetime = logger.get_datetime();
		string index_path = m_data_index_path + suffix + ".dax";
		fid = Util::fopen(index_path.c_str(), "wb");
		if (fwrite(&m_data_cnt, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
		if (fwrite(m_data_idx, sizeof(int64), m_data_cnt, fid) != m_data_cnt) throw KaiException(KERR_ASSERT);
		fclose(fid);
		fid = NULL;
	}
	catch (...) {
		if (fid) fclose(fid);
	}
	mu_save_data_indedx.unlock();
}

void Dataset::load_data_index(string suffix) {
	string datetime = logger.get_datetime();

	int64 data_cnt;

	string index_path = m_data_index_path + suffix + ".dax";
	FILE* fid = Util::fopen(index_path.c_str(), "rb");
	if (fread(&data_cnt, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	if (data_cnt != m_data_cnt) throw KaiException(KERR_ASSERT);
	if (fread(m_data_idx, sizeof(int64), m_data_cnt, fid) != m_data_cnt) throw KaiException(KERR_ASSERT);
	fclose(fid);
}
#else
/*
void Dataset::save_data_index() {
	string datetime = logger.get_datetime();

	FILE* fid = Util::fopen(m_data_index_path.c_str(), "wb");
	if (fwrite(&m_data_cnt, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	if (fwrite(m_data_idx, sizeof(int64), m_data_cnt, fid) != m_data_cnt) throw KaiException(KERR_ASSERT);
	fclose(fid);
}

void Dataset::load_data_index() {
	string datetime = logger.get_datetime();

	int64 data_cnt;

	FILE* fid = Util::fopen(m_data_index_path.c_str(), "rb");
	if (fread(&data_cnt, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	if (data_cnt != m_data_cnt) throw KaiException(KERR_ASSERT);
	if (fread(m_data_idx, sizeof(int64), m_data_cnt, fid) != m_data_cnt) throw KaiException(KERR_ASSERT);
	fclose(fid);
}
*/

//hscho
void Dataset::save_data_index(string suffix) {
	static mutex mu_save_data_indedx;

	FILE* fid = NULL;
	mu_save_data_indedx.lock();
	try {
		string datetime = logger.get_datetime();
		string index_path = m_data_index_path + suffix + ".dax";
		fid = Util::fopen(index_path.c_str(), "wb");
		if (fwrite(&m_data_cnt, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
		if (fwrite(m_data_idx, sizeof(int64), m_data_cnt, fid) != m_data_cnt) throw KaiException(KERR_ASSERT);
		fclose(fid);
		fid = NULL;
	}
	catch (...) {
		if (fid) fclose(fid);
	}
	mu_save_data_indedx.unlock();
}

void Dataset::load_data_index(string suffix) {
	string datetime = logger.get_datetime();

	int64 data_cnt;

	string index_path = m_data_index_path + suffix + ".dax";
	FILE* fid = Util::fopen(index_path.c_str(), "rb");
	if (fread(&data_cnt, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	if (data_cnt != m_data_cnt) throw KaiException(KERR_ASSERT);
	if (fread(m_data_idx, sizeof(int64), m_data_cnt, fid) != m_data_cnt) throw KaiException(KERR_ASSERT);
	fclose(fid);
}
#endif

void Dataset::m_show_select_results(Value cest, Value cans, vector<string> names) {
	Array<float> est = cest, ans = cans;

	int64 mb_size = est.shape()[0], vec_size = est.shape()[1];

	Array<int64> est_idx = kmath->argmax(est, 0).to_host();
	Array<int64> ans_idx = kmath->argmax(ans, 0).to_host();

	Array<float> probs = kmath->softmax(est).to_host();

	for (int64 n = 0; n < mb_size; n++) {
		string buffer;
		string delimeter = "[";
		for (int64 i = 0; i < vec_size; i++) {
			float value = probs[Idx(n, i)] + 0.005f;
			buffer += delimeter + to_string((int64)(100 * value));
			delimeter = ", ";
		}
		buffer += "]";

		string estr = names[est_idx[Idx(n)]];
		string astr = names[ans_idx[Idx(n)]];
		string rstr = (estr == astr) ? "O" : "X";

		logger.Print("%s => estimate %s : answer %s => %s", buffer.c_str(), estr.c_str(), astr.c_str(), rstr.c_str());
	}
}

void Dataset::m_show_seq_binary_results(Value cest, Value cans) {
	Array<float> est = cest, ans = cans;
	Array<float> probs = kmath->sigmoid(est).to_host();;

	ans = ans.to_host();

	int64 mb_size = est.shape()[0], vec_size = est.shape()[1];

	for (int64 n = 0; n < mb_size; n++) {
		logger.PrintWait("Est: [");
		for (int64 i = 0; i < vec_size; i++) {
			if (i > 0) logger.PrintWait(",");
			logger.PrintWait("%4.2f", probs[Idx(n, i, 0)]);
		}
		logger.Print("]");

		logger.PrintWait("Ans: [");
		for (int64 i = 0; i < vec_size; i++) {
			if (i > 0) logger.PrintWait(",");
			logger.PrintWait("%4.2f", ans[Idx(n, i, 0)]);
		}
		logger.Print("]\n");
	}
}

void Dataset::m_load_mnist_data(string path, Array<unsigned char>& images, Array<unsigned char>& labels, vector<string>& target_names) {
	string image_path = path + "train-images-idx3-ubyte";
	string label_path = path + "train-labels-idx1-ubyte";

	kmath->load_from_file(image_path, images);
	kmath->load_from_file(label_path, labels);

	for (int64 n = 0; n < 10; n++) target_names.push_back(to_string(n));
}

void Dataset::m_load_cifar10_data(string path, Array<unsigned char>& images, Array<unsigned char>& labels, vector<string>& target_names) {
	images = Array<unsigned char>(Shape(50000, 3072));
	labels = Array<unsigned char>(Shape(50000));

	string meta_path = path + "batches.meta.txt";

	FILE* fid = Util::fopen(meta_path.c_str(), "rt");
	char buffer[128];

	for (int64 i = 0; i < 10; i++) {
		fgets(buffer, 128, fid);
		buffer[strlen(buffer) - 1] = 0;
		target_names.push_back(buffer);
	}

	fclose(fid);

	unsigned char* p_image = images.data_ptr();
	unsigned char* p_label = labels.data_ptr();

	for (int64 n = 0; n < 5; n++) {
		//hs.cho
		//sprintf_s<128>(buffer,"data_batch_%lld.bin", n+1);
		sprintf_s(buffer, 128,"data_batch_%lld.bin", n+1);
		string file_path = path + buffer;

		FILE* fid = Util::fopen(file_path.c_str(), "rb");

		for (int64 i = 0; i < 10000; i++) {
			fread(p_label, sizeof(unsigned char), 1, fid);
			fread(p_image, sizeof(unsigned char), 3072, fid);

			p_label++;
			p_image += 3072;
		}

		fclose(fid);
	}

	images = images.reshape(Shape(-1, 3, 32, 32)).transpose(Idx(0, 2, 3, 1));
}

void Dataset::m_dump_mnist_image_data(Array<float> xs) {
	m_draw_images_horz(xs, Shape(28,28), 5);
}

void Dataset::m_draw_images_horz(Value csx, Shape image_shape, int ratio) {
	if (ms_img_display_mode || ms_img_save_folder != "") {
		//Array<float> images = csx;
		//string savepath = args::
		//hs.cho
#ifdef KAI2021_WINDOWS
		Util::draw_images_horz(ms_img_display_mode, ms_img_save_folder, csx, image_shape, ratio);
#endif	
	
	}
	else {
		logger.Print("Sorry... drawing imgae is not permitted");
	}
}

void Dataset::open_data_channel_repeat(enum data_channel channel, int64 batch_size, int64 max_stack) {  // (데이터 뒤섞기 + data_count 크기의 미니배치 제공) 기능을 반복해 수행, max_stack 이내의 재고 유자
	int64* data_idx = m_data_idx + m_data_begin[channel];
	int64 total_count = m_data_count[channel];

	DataChannel* pChannel = new DataChannel(this, channel, data_idx, total_count, 0, 0, batch_size, max_stack);

	m_channels[channel] = pChannel;
}

void Dataset::open_data_channel_all(enum data_channel channel) {  // 전체 데이터를 하나의 미니배치로 제공, 데이터 뒤섞기 없음
	int64* data_idx = m_data_idx + m_data_begin[channel];
	int64 total_count = m_data_count[channel];

	DataChannel* pChannel = new DataChannel(this, channel, data_idx, total_count, 1, 1, total_count, 1);

	m_channels[channel] = pChannel;
}

void Dataset::open_data_channel_once(enum data_channel channel, int64 data_count) {  // 데이터 뒤섞기 후에 data_count 크기의 미니배치 하나만 제공
	int64* data_idx = m_data_idx + m_data_begin[channel];
	int64 total_count = m_data_count[channel];

	assert(data_count <= total_count);

	DataChannel* pChannel = new DataChannel(this, channel, data_idx, total_count, 1, 1, data_count, 1);

	m_channels[channel] = pChannel;
}

void Dataset::open_data_channel_batchs(enum data_channel channel, int64 batch_count, int64 batch_size) { // 데이터 뒤섞기 후 batch_size 크기의 미니배치를 batch_count 회만큼 반복 제공
	int64* data_idx = m_data_idx + m_data_begin[channel];
	int64 total_count = m_data_count[channel];

	assert(batch_count * batch_size <= total_count);

	DataChannel* pChannel = new DataChannel(this, channel, data_idx, total_count, 1, batch_count, batch_size, batch_count);

	m_channels[channel] = pChannel;
}

int64 Dataset::open_data_channel_epochs(enum data_channel channel, int64 epoch_count, int64 batch_size) {  // 전체 데이터를 batch_size 크기의 미니배치로 분할해 epoch_count 회만큼 반복 제공, 매 에포크 시작시 데이터 뒤섞기 수행, 에포크당 미니배치 개수를 반환
	if (channel != data_channel::train && channel != data_channel::autoencode) {
		throw KaiException(KERR_ASSERT);
	}

	int64* data_idx = m_data_idx + m_data_begin[channel];
	int64 total_count = m_data_count[channel];

	assert(batch_size > 0);

	int64 batch_count = total_count / batch_size;
	
	DataChannel* pChannel = new DataChannel(this, channel, data_idx, total_count, epoch_count, batch_count, batch_size);

	m_channels[channel] = pChannel;

	return batch_count;
}

void Dataset::get_data(enum data_channel channel, Dict& xs, Dict& ys) {  // 채널 오픈시 예약된 내용에 따라 멀티스레드 방식으로 데이터를 준비해 호출 당 하나의 미니배치 제공
	m_channels[channel]->get_data(xs, ys);
}

void Dataset::visualize_main(Dict xs, Dict ys, Dict outs) {
	Dict dxs = xs["default"], dys = ys["default"], douts = outs["default"];
	Array<float> cxs = dxs["data"], cys = dys["data"], couts = douts["data"];

	Array<float> hxs = CudaConn::ToHostArray(cxs, "visualize(x)");
	Array<float> hys = CudaConn::ToHostArray(cys, "visualize(y)");
	Array<float> hout = CudaConn::ToHostArray(couts, "visualize(out)");

	visualize(hxs, hout, hys);
}

void Dataset::close_data_channel(enum data_channel channel) {
	delete m_channels[channel];
	m_channels.erase(channel);
}

AutoencodeDataset::AutoencodeDataset(string name, string mode, float ratio, bool x_seq, bool y_seq)
	: Dataset(name, mode, x_seq, y_seq) {
	m_ratio = ratio;
}

AutoencodeDataset::~AutoencodeDataset() {
}

int64 AutoencodeDataset::train_count() {
	return m_data_count[data_channel::train];
}

int64 AutoencodeDataset::autoencode_count() {
	return m_data_count[data_channel::autoencode];
}

void AutoencodeDataset::visualize_autoencode(Dict xs, Dict code, Dict repl, Dict outs, Dict ys) {
	throw KaiException(KERR_ASSERT);
}

void AutoencodeDataset::visualize_hash(Array<int64> rank1, Array<int64> rank2, Array<int64> key_labels, Array<int64> dat_label, Array<float> distance, Array<float> keys, Array<float> repl, Array<float> xs) {
	throw KaiException(KERR_ASSERT);
}

void GanDataset::gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
	assert(ysize == 1);
	*py = 1.0;
}

string GanDataset::get_loss_str(Dict loss) {
	float loss1 = 0;
	float loss2 = 0;

	if (loss.find("discriminor") != loss.end()) loss1 = loss["discriminor"];
	if (loss.find("generator") != loss.end()) loss2 = loss["generator"];

	string buf(64, '\0');
	int64 written = std::snprintf(&buf[0], buf.size(), "(dis:%7.5f, gen:%7.5f)", loss1, loss2);
	buf.resize(written);

	return buf;
}

void GanDataset::visualize(Value xs, Value estimates, Value answers) {
	throw KaiException(KERR_ASSERT);
}

void GanDataset::log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2) {
	string loss_str = get_loss_str(loss_mean);

	string acc_keys;
	string acc_means_str = get_acc_str(acc_keys, acc_mean);

	if (batch_count <= 0)
		logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
	else
		logger.PrintWait("    Batch %lld/%lld(in Epoch %lld): ", batch, batch_count, epoch);

	logger.Print("loss=%s, accuracy%s=%s (%lld/%lld secs)", loss_str.c_str(), acc_keys.c_str(), acc_means_str.c_str(), tm1, tm2);
}

void GanDataset::log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2) {
	string loss_str = get_loss_str(loss_mean);

	string acc_keys;
	string acc_means_str = get_acc_str(acc_keys, acc_mean);

	if (batch_count <= 0)
		logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
	else
		logger.PrintWait("    Batch %lld/%lld(in Epoch %lld): ", batch, batch_count, epoch);

	logger.Print("loss=%s, accuracy%s=%s (%lld/%lld secs)", loss_str.c_str(), acc_keys.c_str(), acc_means_str.c_str(), tm1, tm2);
}

void GanDataset::log_test(string name, Dict acc, int64 tm1, int64 tm2) {
	string acc_keys;
	string acc_means_str = get_acc_str(acc_keys, acc);

	logger.Print("Test result: accuracy%s=%s (%lld/%lld secs)", acc_keys.c_str(), acc_means_str.c_str(), tm1, tm2);
}
