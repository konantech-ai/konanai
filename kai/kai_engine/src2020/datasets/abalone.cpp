/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "abalone.h"
#include "../core/log.h"

AbaloneDataset::AbaloneDataset() : Dataset("abalone", "regression") {
	vector<string> head;
	vector<vector<string>> rows = Util::load_csv(KArgs::data_root + "chap01/abalone.csv", &head);

	input_shape = Shape(10);
	output_shape = Shape(1);

	m_default_xs = Array<float>::zeros(Shape(rows.size(), 10));
	m_default_ys = Array<float>::zeros(Shape(rows.size(), 1));

	for (unsigned int n = 0; n < rows.size(); n++) {
		vector<string> row = rows[n];
		string sex = row[0];

		if (sex == "I") m_default_xs[Idx(n, 0)] = 1.0;
		else if (sex == "M") m_default_xs[Idx(n, 1)] = 1.0;
		else if (sex == "F") m_default_xs[Idx(n, 2)] = 1.0;

		for (int k = 3; k < 10; k++) m_default_xs[Idx(n, k)] = std::stof(row[k - 2]);

		m_default_ys[Idx(n, 0)] = std::stof(row[8]);
	}

	m_shuffle_index(rows.size());
}

AbaloneDataset::~AbaloneDataset() {
}

/*
void AbaloneDataset::gen_data(int data_idx, int64 xsize, float* px, int64 ysize, float* py) {
	float* pmx = m_xs.data_ptr() + data_idx * xsize;
	float* pmy = m_ys.data_ptr() + data_idx * xsize;

	memcpy(pmx, px, xsize * sizeof(float));
	memcpy(pmy, py, ysize * sizeof(float));
}
*/

/*
void AbaloneDataset::generate_data(int* data_idxs, int size, Value& xs, Value& ys) {
	Array<float> xarr(input_shape.add_front(size));
	Array<float> yarr(output_shape.add_front(size));

	for (int n = 0; n < size; n++) {
		logger.Print("m_xs is in shape %s, data_idxs[n] = %d", m_xs.shape().desc().c_str(), data_idxs[n]);

		xarr[Axis(n, _all_)] = m_xs[Axis(data_idxs[n], _all_)];
		yarr[Axis(n, _all_)] = m_ys[Axis(data_idxs[n], _all_)];
	}

	xs = xarr;
	ys = yarr;
}
*/

void AbaloneDataset::visualize(Value cxs, Value cest, Value cans) {
	Array<float> xs = cxs, est = cest, ans = cans;

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

		logger.Print("%s => estimate %4.1f : answer %4.1f", buffer.c_str(), est[Idx(n, 0)], ans[Idx(n, 0)]);
	}
}
