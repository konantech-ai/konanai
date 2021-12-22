/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "test_dataset.h"
#include "../core/log.h"

TestDataset::TestDataset() : Dataset("test", "binary") {
	input_shape = Shape(4, 4, 1);
	output_shape = Shape(1);

	m_shuffle_index(20);
}

TestDataset::~TestDataset() {
}

void TestDataset::generate_data(int* data_idxs, int size, Array<float>& xs, Value& ys) {
	Array<float> xarr = kmath->random_normal(0, 1.0f, input_shape.add_front(size));
	Array<float> yarr = kmath->random_bernoulli(output_shape.add_front(size), 0.5f);

	xs = xarr;
	ys = yarr;
}

void TestDataset::visualize(Array<float> cxs, Value cest, Value cans) {
	logger.Print("TestDataset::visualize() is called");
	/*
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

		float eprob = est[Idx(n, 0)];
		float aprob = ans[Idx(n, 0)];

		string estr = (eprob > 0.5) ? "pulsar" : "star";
		string astr = (aprob > 0.5) ? "pulsar" : "star";
		string rstr = (estr == astr) ? "O" : "X";

		logger.Print("%s => estimate %s(prob: %f) : answer %s => %s", buffer.c_str(), estr.c_str(), est[Idx(n, 0)], astr.c_str(), rstr.c_str());
	}
	*/
}

