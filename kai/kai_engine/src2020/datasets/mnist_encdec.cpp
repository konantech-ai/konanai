/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "mnist_encdec.h"
#include "../core/log.h"
#include "../core/random.h"

MnistEncDecDataset::MnistEncDecDataset(string name) : Dataset(name, "dual_acc") {
	string path = KArgs::data_root + "mnist/";

	m_load_mnist_data(path, m_images, m_labels, m_target_names);

	m_images = m_images.reshape(Shape(-1, 28 * 28));

	int64 data_cnt = m_images.axis_size(0);

	m_shuffle_index(data_cnt);
}

MnistEncDecDataset::~MnistEncDecDataset() {
}

Dict MnistEncDecDataset::forward_postproc(Dict xs, Dict ys, Dict outs, string mode) {
	Dict y_def = ys["default"], o_def = outs["default"];
	float loss = Dataset::forward_postproc_base(y_def, o_def, "classify");
	return Value::wrap_dict("default", loss);
}

Dict MnistEncDecDataset::backprop_postproc(Dict ys, Dict outs, string mode) {
	Dict y_def = ys["default"], o_def = outs["default"];
	Dict G_output = Dataset::backprop_postproc_base(y_def, o_def, "classify");
	return Value::wrap_dict("default", G_output);
}

Dict MnistEncDecDataset::eval_accuracy(Dict xs, Dict ys, Dict outs, string mode) {
	Dict y_def = ys["default"], o_def = outs["default"];

	Dict acc;

	float acc_char = Dataset::eval_accuracy_base(xs, y_def, o_def, "classify");

	Array<float> yarr = y_def["data"], oarr = o_def["data"];

	yarr = yarr.to_host();
	oarr = oarr.to_host();

	int64 mb_size = yarr.axis_size(0);

	Array<int64> estimate = hmath.argmax(oarr.merge_time_axis(), 0).split_time_axis(mb_size);
	Array<int64> answer = hmath.argmax(yarr.merge_time_axis(), 0).split_time_axis(mb_size);
	Array<bool> correct = hmath.compare_rows(estimate, answer); // ÁÙ ´ÜÀ§ ºñ±³
	float acc_word = kmath->mean(correct);

	acc["char"] = acc_char;
	acc["word"] = acc_word;

	return acc;
}

MnistEngDataset::MnistEngDataset() : MnistEncDecDataset("mnist_encdec_eng") {
	m_word_len = 6;

	m_set_captions();

	input_shape = m_images.shape().remove_front();
	output_shape = m_captions.axis_size(-1); // length("abc...xyz" + "$")=27
}

MnistEngDataset::~MnistEngDataset() {
}

void MnistEngDataset::m_set_captions() {
	const char* words[10] = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };

	m_captions = hmath.zeros(Shape(10, m_word_len, 27));

	for (int64 n = 0; n < 10; n++) {
		for (int64 m = 0; m < m_word_len; m++) {
			int64 cidx = (m < (int64) strlen(words[n])) ? (words[n][m] - 'a' + 1) : 0;
			assert(cidx >= 0 && cidx < 27);
			m_captions[Idx(n, m, cidx)] = 1.0f;
		}
	}
}

void MnistEngDataset::gen_plain_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y) {
	assert(xsize == input_shape.total_size());

	Array<unsigned char> buf;
	buf.get(m_images, Axis(data_idx, _all_));
	Array<float> fimage = buf.to_float();
	fimage = (fimage - 128.0f) / 128.0f;

	float* pimage = fimage.data_ptr();

	memcpy(px, pimage, xsize*sizeof(float));
}

void MnistEngDataset::gen_seq_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
	int64 digit = m_labels[Idx(data_idx)];
	float* pdigit = m_captions.data_ptr() + digit * ysize;
	memcpy(py, pdigit, ysize * sizeof(float));
}

string MnistEngDataset::m_eng_prob_to_caption(Array<int64> arr, int64 nth) {
	arr = arr.to_host();
	string numstr = "";
	for (int64 n = 0; n < m_word_len; n++) {
		int64 cidx = arr[Idx(nth, n)];
		//if (cidx == 26) break;
		//numstr += (char)(cidx + 'a');
		if (cidx == 0) numstr += '$';
		else numstr += (char)(cidx - 1 + 'a');
	}
	return numstr;
}

void MnistEngDataset::visualize(Value xs, Value estimates, Value answers) {
	m_dump_mnist_image_data(xs);

	Array<float> est = estimates, ans = answers;
	int64 mb_size = est.axis_size(0);
	est = est.merge_time_axis(), ans = ans.merge_time_axis();
	Array<int64> est_idx = kmath->argmax(est, 0), ans_idx = kmath->argmax(ans, 0);
	est_idx = est_idx.split_time_axis(mb_size), ans_idx = ans_idx.split_time_axis(mb_size);

	for (int64 n = 0; n < mb_size; n++) {
		string estr = m_eng_prob_to_caption(est_idx, n);
		string astr = m_eng_prob_to_caption(ans_idx, n);
		logger.Print("estimate: '%s' vs. answer: '%s'", estr.c_str(), astr.c_str());
	}
}

string MnistKorDataset::ms_alphabet = "¿µÀÏÀÌ»ï»ç¿ÀÀ°Ä¥ÆÈ±¸½Ê¹éÃµ³¡";

MnistKorDataset::MnistKorDataset(int64 length) : MnistEncDecDataset("mnist_encdec_kor") {
	m_length = length;

	input_shape = m_images.shape().remove_front();
	output_shape = Shape(14); // length of alphbet
}

MnistKorDataset::~MnistKorDataset() {
}

void MnistKorDataset::gen_seq_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value &to_y) {
	Array<unsigned char> buf;
	int64 num = 0;
	int64 dat_size = input_shape.total_size();

	assert(xsize == m_length * dat_size);

	for (int64 m = 0; m < m_length; m++) {
		int64 idx = (m == 0) ? data_idx : Random::dice(m_data_cnt);
		buf.get(m_images, Axis(idx, _all_));

		Array<float> fimage = buf.to_float();
		fimage = (fimage - 128.0f) / 128.0f;

		float* pimage = fimage.data_ptr();
		memcpy(px, pimage, dat_size * sizeof(float));

		px += dat_size;

		num = num * 10 + m_labels[Idx(idx)];
	}

	to_y = num;
}

void MnistKorDataset::gen_seq_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
	int64 num = from_x;
	int64 m = 0;

	memset(py, 0, sizeof(float) * ysize);

	if (num >= 1000) m_add_digit_pair(py, m, num, 1000);
	if (num >= 100)  m_add_digit_pair(py, m, num, 100);
	if (num >= 10)   m_add_digit_pair(py, m, num, 10);
	if (num >= 1)    m_add_digit_pair(py, m, num, 1);

	while ( m * 14 < ysize) m_set_digit(py, m, 13);
}

void MnistKorDataset::m_add_digit_pair(float* py, int64& m, int64& num, int64 unit) {
	int64 digit = num / unit;
	num = num % unit;

	if (digit >= 2 || unit == 1) m_set_digit(py, m, digit);

	if (unit == 10) m_set_digit(py, m, 10); // 10-th: 10
	else if (unit == 100) m_set_digit(py, m, 11); // 11-th: 100
	else if (unit == 1000) m_set_digit(py, m, 12); // 12-th: 1000
}

void MnistKorDataset::m_set_digit(float* py, int64& m, int64 digit) {
	py[m++ * 14 + digit] = 1.0;
}

string MnistKorDataset::m_kor_prob_to_caption(Array<int64> arr, int64 nth) {
	string numstr = "";
	int64 syl_len =  ms_alphabet.length() / 14;

	for (int64 n = 0; n < 2 * m_length; n++) {
		int64 cidx = arr[Idx(nth, n)];
		if (cidx == 13) numstr += '$';
		else numstr += ms_alphabet.substr(syl_len * cidx, syl_len);
	}
	return numstr;
}

void MnistKorDataset::visualize(Value xs, Value estimates, Value answers) {
	m_dump_mnist_image_data(xs);

	Array<float> est = estimates, ans = answers;
	int64 mb_size = est.axis_size(0);
	est = est.merge_time_axis(), ans = ans.merge_time_axis();
	Array<int64> est_idx = kmath->argmax(est, 0), ans_idx = kmath->argmax(ans, 0);
	est_idx = est_idx.split_time_axis(mb_size), ans_idx = ans_idx.split_time_axis(mb_size);

	est_idx = est_idx.to_host();
	ans_idx = ans_idx.to_host();

	for (int64 n = 0; n < mb_size; n++) {
		string estr = m_kor_prob_to_caption(est_idx, n);
		string astr = m_kor_prob_to_caption(ans_idx, n);
		logger.Print("estimate: '%s' vs. answer: '%s'", estr.c_str(), astr.c_str());
	}
}
