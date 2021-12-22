/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "flower.h"
#include "../core/util.h"
#include "../core/log.h"

FlowerDataset::FlowerDataset(string data_name, string cache_name, Shape resolution, Shape data_shape) : Dataset("flower", "classify") {
	logger.Print("dataset loading...");
	
	string data_path = KArgs::data_root + data_name;
	string cache_path = KArgs::cache_root + "flowers";
	
	Util::mkdir(KArgs::cache_root);
	Util::mkdir(cache_path);

	cache_path += "/"+ cache_name;

#ifdef KAI2021_WINDOWS
	try {
		load_cache(cache_path);
	}
	catch (exception err) {
		create_cache(data_path, resolution, data_shape);
		save_cache(cache_path);
	}
#else
	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 1);
#endif

	m_default_xs = (m_default_xs - 127.5f) / 127.5f;

	int64 target_cnt = m_target_names.size();

	m_resolution = resolution;

	input_shape = data_shape;
	output_shape = Shape(target_cnt);

	int64 data_cnt = m_default_xs.shape()[0];

	m_shuffle_index(data_cnt);

	logger.Print("dataset prepared...");
}

FlowerDataset::~FlowerDataset() {
}

#ifdef KAI2021_WINDOWS
void FlowerDataset::load_cache(string cache_path) {
	//throw exception("temp");
	logger.Print("load_cache start time: %lld", time(NULL));
	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 1);
	logger.Print("load_cache end time: %lld", time(NULL));
}

void FlowerDataset::save_cache(string cache_path) {
	logger.Print("save_cache start time: %lld", time(NULL));
	Util::save_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 1);
	logger.Print("save_cache end time: %lld", time(NULL));
}

void FlowerDataset::create_cache(string data_path, Shape resolution, Shape data_shape) {
	logger.Print("create_cache start time: %lld", time(NULL));

	m_target_names = Util::list_dir(data_path);

	vector<int> cat_idxs;
	vector<string> file_names;

	for (int idx = 0; idx < m_target_names.size(); idx++) {
		string subpath = data_path + '/' + m_target_names[idx];
		vector<string> filenames = Util::list_dir(subpath);
		for (vector<string>::iterator it = filenames.begin(); it != filenames.end(); it++) {
			if (it->length() < 4 || it->substr(it->length() - 4) != ".jpg") continue;
			file_names.push_back(*it);
			cat_idxs.push_back(idx);
		}
	}

	size_t total_count = cat_idxs.size();

	m_default_xs = Array<float>::zeros(data_shape.add_front(total_count));
	m_default_ys = Array<float>::zeros(Shape(total_count, m_target_names.size()));

	float* pXs = m_default_xs.data_ptr();
	int64 dat_size = data_shape.total_size();

	for (int n = 0; n < cat_idxs.size(); n++) {
		m_default_ys[Idx(n, cat_idxs[n])] = 1.0f;
		string filepath = data_path + '/' + m_target_names[cat_idxs[n]] + '/' + file_names[n];
		Util::load_jpeg_image_pixels(pXs, filepath, data_shape);
		pXs += dat_size;
	}

	logger.Print("create_cache end time: %lld", time(NULL));
}
#else
#endif

/*
void FlowerDataset::generate_data(int* data_idxs, int size, Value& xs, Value& ys) {
	Array<float> xarr(input_shape.add_front(size));
	Array<float> yarr(output_shape.add_front(size));

	if (m_xs.dim() == 2) {
		for (int n = 0; n < size; n++) {
			xarr[Axis(n, _all_)] = m_xs[Axis(data_idxs[n], _all_)];
		}
	}
	else if (m_xs.dim() == 4) {
		for (int n = 0; n < size; n++) {
			xarr[Axis(n, _all_, _all_, _all_)] = m_xs[Axis(data_idxs[n], _all_, _all_, _all_)];
		}
	}
	else {
		throw KaiException(KERR_ASSERT);
	}

	if (m_ys.dim() == 2) {
		for (int n = 0; n < size; n++) {
			yarr[Axis(n, _all_)] = m_ys[Axis(data_idxs[n], _all_)];
		}
	}
	else if (m_ys.dim() == 4) {
		for (int n = 0; n < size; n++) {
			yarr[Axis(n, _all_, _all_, _all_)] = m_ys[Axis(data_idxs[n], _all_, _all_, _all_)];
		}
	}
	else {
		throw KaiException(KERR_ASSERT);
	}

	xarr = xarr / 128.0f - 1.0f;

	xs = xarr;
	ys = yarr;
}
*/

void FlowerDataset::visualize(Value cxs, Value cest, Value cans) {
	m_draw_images_horz(cxs, m_resolution);
	m_show_select_results(cest, cans, m_target_names);
}
