/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "pulsar.h"
#include "../core/log.h"

PulsarDataset::PulsarDataset() : Dataset("pulsar", "binary") {
	vector<vector<string>> rows = Util::load_csv(KArgs::data_root + "chap02/pulsar_stars.csv");

	int64 data_cnt =  rows.size();
	int64 field_cnt = rows[0].size() - 1;

	input_shape = Shape(field_cnt);
	output_shape = Shape(1);

	m_default_xs = Array<float>::zeros(Shape(data_cnt, field_cnt));
	m_default_ys = Array<float>::zeros(Shape(data_cnt, 1));

	for (int n = 0; n < data_cnt; n++) {
		vector<string> row = rows[n];
		for (int k = 0; k < field_cnt; k++) {
			m_default_xs[Idx(n, k)] = std::stof(row[k]);
		}

		m_default_ys[Idx(n, 0)] = std::stof(row[field_cnt]);
	}

	m_shuffle_index(data_cnt);
}

PulsarDataset::~PulsarDataset() {
}

void PulsarDataset::visualize(Value cxs, Value cest, Value cans) {
	Array<float> xs = cxs, est = cest, ans = cans;
	Array<float> probs = kmath->sigmoid(est).to_host();

	int64 mb_size = xs.shape()[0], vec_size = xs.shape()[1];

	for (int n = 0; n < mb_size; n++) {
		string buffer;
		string delimeter = "[";
		for (int i = 0; i < vec_size; i++) {
			float value = xs[Idx(n, i)];
			buffer += delimeter + to_string(value);
			delimeter = ", ";
		}
		buffer += "]";

		float eprob = probs[Idx(n, 0)];
		float aprob = ans[Idx(n, 0)];

		string estr = (eprob > 0.5) ? "pulsar" : "star";
		string astr = (aprob > 0.5) ? "pulsar" : "star";
		string rstr = (estr == astr) ? "O" : "X";

		logger.Print("%s => estimate %s(prob: %3.1f) : answer %s => %s", buffer.c_str(), estr.c_str(), eprob*100, astr.c_str(), rstr.c_str());
	}
}

PulsarSelectDataset::PulsarSelectDataset() : Dataset("pulsar_select", "classify") {
	vector<vector<string>> rows = Util::load_csv(KArgs::data_root + "chap02/pulsar_stars.csv");

	int64 data_cnt = rows.size();
	int64 field_cnt = rows[0].size() - 1;

	input_shape = Shape(field_cnt);
	output_shape = Shape(2);

	m_default_xs = Array<float>::zeros(Shape(data_cnt, field_cnt));
	m_default_ys = Array<float>::zeros(Shape(data_cnt, 2));

	for (int n = 0; n < data_cnt; n++) {
		vector<string> row = rows[n];
		for (int k = 0; k < field_cnt; k++) {
			m_default_xs[Idx(n, k)] = std::stof(row[k]);
		}

		int label = (int)std::stof(row[field_cnt]);
		m_default_ys[Idx(n, label)] = 1.0;
	}

	m_shuffle_index(data_cnt);

	m_targe_name.push_back("star");
	m_targe_name.push_back("pulsar");
}

PulsarSelectDataset::~PulsarSelectDataset() {
}

void PulsarSelectDataset::visualize(Value cxs, Value cest, Value cans) {
	m_show_select_results(cest, cans, m_targe_name);
}
