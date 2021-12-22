/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "mnist_gan.h"
#include "../core/array.h"

GanDatasetPicture::GanDatasetPicture(string name, string filename, string cache_name)
	: GanDataset(name, "binary") {
	Array<float> ys;
	vector<string> target_names;

	string cache_path = KArgs::data_root + cache_name;
	Util::load_kodell_dump_file(cache_path, m_default_xs, ys, &target_names, 0);

	int64 data_cnt = m_default_xs.axis_size(0);

	input_shape = m_default_xs.shape().remove_front();
	output_shape = Shape(1);

	m_shuffle_index(data_cnt);
	//m_gan_shuffle_index(data_cnt);
}

GanDatasetPicture::~GanDatasetPicture() {
}

void GanDatasetPicture::visualize(Gan* model, Dict real_xs, Dict fake_xs) {
	m_draw_images_horz(real_xs, Shape(32, 32), 3);
	m_draw_images_horz(fake_xs, Shape(32, 32), 3);
}

GanDatasetMnist::GanDatasetMnist(string name, string nums)
	: GanDataset(name, "binary") {
	string path = KArgs::data_root + "mnist/";

	Array<unsigned char> images;
	m_load_mnist_data(path, images, m_labels, m_target_names);

	images = images.reshape(Shape(-1, 28 * 28));

	if (nums != "") {
		Shape shape(m_labels.axis_size(0));
		Array<bool> valid(shape);
		valid.reset();
		for (int n = 0; n < (int) nums.size(); n++) {
			unsigned char num = (unsigned char)(nums[n] - '0');
			Array<bool> temp = (m_labels == num);
			valid = valid.logical_or(temp);
		}
		images = images.extract_selected(valid);
	}

	int64 data_cnt = images.axis_size(0);

	input_shape = images.shape().remove_front();
	output_shape = Shape(1); // length of alphbet

	m_default_xs = (images.to_float() - 127.5) / 127.5;

	m_shuffle_index(data_cnt);
	//m_gan_shuffle_index(data_cnt);
}

GanDatasetMnist::~GanDatasetMnist() {
}

void GanDatasetMnist::visualize(Gan* model, Dict mixed_xs, Dict mixed_ys) {
	Array<float> x_arr = mixed_xs["data"];
	Array<float> y_arr = mixed_ys["data"];

	m_dump_mnist_image_data(x_arr);
}

