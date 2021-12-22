/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "gan_cifar10.h"
#include "../core/array.h"
#include "../core/log.h"

GanCifar10Dataset::GanCifar10Dataset(string name) : GanDataset(name, "binary") {
	logger.Print("dataset loading...");

	string path = KArgs::data_root + "cifar-10-binary/cifar-10-batches-bin/";

	Array<unsigned char> images;
	Array<unsigned char> labels;
	vector<string> target_names;

	m_load_cifar10_data(path, images, labels, target_names);

	int64 data_cnt = images.axis_size(0);

	input_shape = images.shape().remove_front();
	output_shape = Shape(1); // length of alphbet

	m_default_xs = (images.to_float() - 127.5) / 127.5;

	m_shuffle_index(data_cnt);
	//m_gan_shuffle_index(data_cnt);

	logger.Print("dataset prepared...");
}

GanCifar10Dataset::~GanCifar10Dataset() {
}

void GanCifar10Dataset::visualize(Gan* model, Dict mixed_xs, Dict mixed_ys) {
	Array<float> x_arr = mixed_xs["data"];
	m_draw_images_horz(x_arr, Shape(32, 32, 3), 3);
}
