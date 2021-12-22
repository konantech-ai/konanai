/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "cifar10_auto.h"

Cifar10AutoDataset::Cifar10AutoDataset(float ratio)
	: AutoencodeDataset("cifar10_auto", "classify", ratio) {
	logger.Print("dataset loading...");

	string path = KArgs::data_root + "cifar-10-binary/cifar-10-batches-bin/";

	m_load_cifar10_data(path, m_images, m_labels, m_target_names);

	int64 data_cnt = m_images.axis_size(0);
	int64 target_cnt = m_target_names.size();

	input_shape = m_images.shape().remove_front();
	output_shape = Shape(target_cnt);

	m_shuffle_index(data_cnt);

	m_data_count[data_channel::autoencode] = m_data_count[data_channel::train];
	m_data_count[data_channel::train] = int((float)m_data_count[data_channel::train] * m_ratio);
}

Cifar10AutoDataset::~Cifar10AutoDataset() {
}

void Cifar10AutoDataset::gen_plain_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y) {
	assert(xsize == 32 * 32 * 3);
	unsigned char* pimages = m_images.data_ptr();
	pimages += data_idx * xsize;
	for (int64 n = 0; n < xsize; n++) {
		px[n] = ((float)pimages[n] - 128.0f) / 128.0f;
	}
}

void Cifar10AutoDataset::gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
	assert(ysize == 10);
	unsigned char* plabels = m_labels.data_ptr();
	memset(py, 0, ysize * sizeof(float));
	py[plabels[data_idx]] = 1.0;
}

void Cifar10AutoDataset::visualize(Value xs, Value cest, Value cans) {
	m_show_select_results(cest, cans, m_target_names);
}

void Cifar10AutoDataset::visualize_autoencode(Dict xs, Dict code, Dict repl, Dict outs, Dict ys) {
	Dict dxs = xs["default"], douts = outs["default"], dys = ys["default"];
	Array<float> cxs = dxs["data"], couts = douts["data"], cys = dys["data"];

	Array<float> hxs = CudaConn::ToHostArray(cxs, "visualize(x)");
	Array<float> houts = CudaConn::ToHostArray(couts, "visualize(out)");
	Array<float> hys = CudaConn::ToHostArray(cys, "visualize(x)");

	Array<float> hstack = hmath.vstack(hxs, houts);

	m_draw_images_horz(hstack, Shape(32, 32, 3), 5);
}

void Cifar10AutoDataset::visualize_hash(Array<int64> rank1, Array<int64> rank2, Array<int64> key_labels, Array<int64> dat_label, Array<float> distance, Array<float> keys, Array<float> repl, Array<float> xs) {

	int64 nkey = rank1.axis_size(0);

	Array<float> images(Shape(2 * nkey, 12, 3072));

	float* iip = images.data_ptr();
	float* kip = keys.data_ptr();
	float* rip = repl.data_ptr();
	float* xip = xs.data_ptr();

	logger.Print("  Semantic Hashing Search Result");
	for (int64 n = 0; n < nkey; n++) {
		memcpy(iip + n * 2 * 12 * 3072, kip + n * 3072, sizeof(float) * 3072);
		memcpy(iip + (n * 2 + 1) * 12 * 3072, rip + n * 3072, sizeof(float) * 3072);

		logger.Print("Search Result for %lld-th key: class = %d", n, key_labels[Idx(n)]);
		for (int64 m = 0; m < 10; m++) {
			int64 idx1 = rank1[Idx(n, m)];
			int64 idx2 = rank2[Idx(n, m)];

			memcpy(iip + (n * 2 * 12 + 2 + m) * 3072, xip + idx1 * 3072, sizeof(float) * 3072);
			memcpy(iip + ((n * 2 + 1) * 12 + 2 + m) * 3072, xip + idx2 * 3072, sizeof(float) * 3072);

			logger.Print("    [%lld] by dist: %f(%lld),  by hash: %f(%lld)", m, distance[Idx(n, idx1)], dat_label[Idx(idx1)], distance[Idx(n, idx2)], dat_label[Idx(idx2)]);
		}
	}

	m_draw_images_horz(images, Shape(32, 32, 3), 5);
}
