/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "office31.h"
#include "../core/util.h"
#include "../core/array.h"
#include "../core/log.h"
#include "../core/host_math.h"
#include "../cuda/cuda_math.h"

Office31Dataset::Office31Dataset(string data_name, string cache_name, Shape resolution, Shape data_shape) : Dataset("office31", "dual_select") {
	logger.Print("dataset loading...");

	string data_path = KArgs::data_root + data_name;
	string cache_path = KArgs::cache_root + "office31";

	Util::mkdir(KArgs::cache_root);
	Util::mkdir(cache_path);

	cache_path += "/" + cache_name;
//hs.cho
//#ifdef KAI2021_WINDOWS
	try {
		load_cache(cache_path);
	}
	catch (exception err) {
		create_cache(data_path, resolution, data_shape);
		save_cache(cache_path);
	}
//#else
//	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 2);
//#endif

	m_default_xs = m_default_xs / 128.0f - 1.0f;

	m_domain_cnt = m_target_names[0].size();
	m_product_cnt = m_target_names[1].size();

	m_resolution = resolution;

	input_shape = data_shape;
	output_shape = Shape(m_domain_cnt + m_product_cnt);

	int64 data_cnt = m_default_xs.shape()[0];

	m_shuffle_index(data_cnt);

	logger.Print("dataset prepared...");
}

Office31Dataset::~Office31Dataset() {
}

//hs.cho
//#ifdef KAI2021_WINDOWS
void Office31Dataset::load_cache(string cache_path) {
	logger.Print("load_cache start time: %lld", time(NULL));
	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, m_target_names, 2);
	logger.Print("load_cache end time: %lld", time(NULL));
}

void Office31Dataset::save_cache(string cache_path) {
	logger.Print("save_cache start time: %lld", time(NULL));
	Util::save_kodell_dump_file(cache_path, m_default_xs, m_default_ys, m_target_names, 2);
	logger.Print("save_cache end time: %lld", time(NULL));
}

void Office31Dataset::create_cache(string data_path, Shape resolution, Shape data_shape) {
	logger.Print("create_cache start time: %lld", time(NULL));

	m_target_names[0] = Util::list_dir(data_path);
	m_target_names[1] = Util::list_dir(data_path+"/"+ m_target_names[0][0]+"/images");

	size_t dom_count = m_target_names[0].size();
	size_t obj_count = m_target_names[1].size();
	size_t cat_count = dom_count + obj_count;

	vector<int> cat_idxs[2];
	vector<string> file_names;

	for (size_t idx1 = 0; idx1 < dom_count; idx1++) {
		for (size_t idx2 = 0; idx2 < obj_count; idx2++) {
			string subpath = data_path + '/' + m_target_names[0][idx1] + "/images/" + m_target_names[1][idx2];
			vector<string> filenames = Util::list_dir(subpath);
			for (vector<string>::iterator it = filenames.begin(); it != filenames.end(); it++) {
				if (it->length() < 4 || it->substr(it->length() - 4) != ".jpg") continue;
				file_names.push_back(*it);
				cat_idxs[0].push_back((int)idx1);
				cat_idxs[1].push_back((int)idx2);
			}
		}
	}

	size_t total_count = cat_idxs[0].size();

	m_default_xs = Array<float>::zeros(data_shape.add_front(total_count));
	m_default_ys = Array<float>::zeros(Shape(total_count, cat_count));

	float* pXs = m_default_xs.data_ptr();
	int64 dat_size = data_shape.total_size();

	for (size_t n = 0; n < cat_idxs[0].size(); n++) {
		m_default_ys[Idx(n, cat_idxs[0][n])] = 1.0f;
		m_default_ys[Idx(n, dom_count + cat_idxs[1][n])] = 1.0f;
		string filepath = data_path + '/' + m_target_names[0][cat_idxs[0][n]] + "/images/" + m_target_names[1][cat_idxs[1][n]] + "/" + file_names[n];
	//hs.cho
#ifdef KAI2021_WINDOWS
		Util::load_jpeg_image_pixels(pXs, filepath, data_shape);
#endif
		pXs += dat_size;
	}

	logger.Print("create_cache end time: %lld", time(NULL));
}
//#else
//#endif

Dict Office31Dataset::forward_postproc(Dict xs, Dict ys, Dict outs, string mode) {
	Dict ydef = ys["default"], odef = outs["default"];
	Array<float> y_pair = ydef["data"], o_pair = odef["data"];

	Array<float> y_domain, y_product;
	Array<float> o_domain, o_product;

	kmath->hsplit(o_pair, m_domain_cnt, o_domain, o_product);
	kmath->hsplit(y_pair, m_domain_cnt, y_domain, y_product);

	Dict y_dom = Value::wrap_dict("data", y_domain), y_prod = Value::wrap_dict("data", y_product);
	Dict o_dom = Value::wrap_dict("data", o_domain), o_prod = Value::wrap_dict("data", o_product);

	float loss_domain = Dataset::forward_postproc_base(y_dom, o_dom, "classify");
	float loss_product = Dataset::forward_postproc_base(y_prod, o_prod, "classify");

	Dict loss;

	loss["domain"] = loss_domain;
	loss["product"] = loss_product;

	return loss;
}

Dict Office31Dataset::backprop_postproc(Dict ys, Dict outs, string mode) {
	Dict ydef = ys["default"], odef = outs["default"];
	Array<float> y_pair = ydef["data"], o_pair = odef["data"];

	Array<float> y_domain, y_product;
	Array<float> o_domain, o_product;

	kmath->hsplit(o_pair, m_domain_cnt, o_domain, o_product);
	kmath->hsplit(y_pair, m_domain_cnt, y_domain, y_product);

	Dict y_dom = Value::wrap_dict("data", y_domain), y_prod = Value::wrap_dict("data", y_product);
	Dict o_dom = Value::wrap_dict("data", o_domain), o_prod = Value::wrap_dict("data", o_product);

	Dict G_dom_dic = Dataset::backprop_postproc_base(y_dom, o_dom, "classify");
	Dict G_prod_dic = Dataset::backprop_postproc_base(y_prod, o_prod, "classify");

	Array<float> G_dom = G_dom_dic["data"], G_prod = G_prod_dic["data"];
	Array<float> G_output = kmath->hstack(G_dom, G_prod);

	return Value::wrap_dict("default", Value::wrap_dict("data", G_output));
}

Dict Office31Dataset::eval_accuracy(Dict xs, Dict ys, Dict outs, string mode) {
	Dict ydef = ys["default"], odef = outs["default"];
	Array<float> y_pair = ydef["data"], o_pair = odef["data"];

	Array<float> y_domain, y_product;
	Array<float> o_domain, o_product;

	kmath->hsplit(o_pair, m_domain_cnt, o_domain, o_product);
	kmath->hsplit(y_pair, m_domain_cnt, y_domain, y_product);

	Dict y_dom = Value::wrap_dict("data", y_domain), y_prod = Value::wrap_dict("data", y_product);
	Dict o_dom = Value::wrap_dict("data", o_domain), o_prod = Value::wrap_dict("data", o_product);

	Dict acc;

	float acc_domain = Dataset::eval_accuracy_base(xs, y_dom, o_dom, "classify");
	float acc_product = Dataset::eval_accuracy_base(xs, y_prod, o_prod, "classify");

	//logger.Print("accs: %f, %f", (float) acc_domain, (float) acc_product);

	acc["domain"] = acc_domain;
	acc["product"] = acc_product;

	return acc;
}


void Office31Dataset::log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2) {
	float loss_mean_domain = loss_mean["domain"];
	float acc_mean_domain = acc_mean["domain"];
	float acc_domain = acc["domain"];

	float loss_mean_product = loss_mean["product"];
	float acc_mean_product = acc_mean["product"];
	float acc_product = acc["product"];

	float loss_mean_sum = loss_mean_domain + loss_mean_product;

	if (batch_count == 0)
		logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
	else
		logger.PrintWait("    Batch %lld/%lld(in Epoch %lld): ", batch, batch_count, epoch);

	logger.Print("loss=%16.9e(%16.9e+%16.9e), accuracy=(%16.9e,%16.9e)/(%16.9e,%16.9e) (%lld/%lld secs)", loss_mean_sum, loss_mean_domain, loss_mean_product,
		acc_mean_domain, acc_mean_product, acc_domain, acc_product, tm1, tm2);
}

void Office31Dataset::log_test(string name, Dict acc, int64 tm1, int64 tm2) {
	float acc_domain = acc["domain"];
	float acc_product = acc["product"];

	logger.Print("Model %s test report: accuracy = %16.9e/%16.9e, (%lld/%lld secs)", name.c_str(), acc_domain, acc_product, tm2, tm1);
	logger.Print("");
}

/*
Value Office31Dataset::get_estimate(Array<float> output, string mode) {
	Array<float> output_pair, output_domain, output_product;
	Array<float> y_pair, y_domain, y_product;

	output_pair = output;
	output_domain = output_pair[Axis(_all_, Ax(0, m_domain_cnt))];
	output_product = output_pair[Axis(_all_, Ax(m_domain_cnt, m_domain_cnt + m_product_cnt))];

	Array<float> estimate_domain = Dataset::get_estimate_base(output_domain, "classify");
	Array<float> estimate_product = Dataset::get_estimate_base(output_product, "classify");

	return kmath->hstack(estimate_domain, estimate_product);
}
*/

void Office31Dataset::visualize(Value cxs, Value cest, Value cans) {
	Array<float> est_pair, est_domain, est_product;
	Array<float> ans_pair, ans_domain, ans_product;

	est_pair = cest;
	est_domain = est_pair[Axis(_all_, Ax(0, m_domain_cnt))];
	est_product = est_pair[Axis(_all_, Ax(m_domain_cnt, m_domain_cnt + m_product_cnt))];

	ans_pair = cans;
	ans_domain = ans_pair[Axis(_all_, Ax(0, m_domain_cnt))];
	ans_product = ans_pair[Axis(_all_, Ax(m_domain_cnt, m_domain_cnt + m_product_cnt))];
//hs.cho
#ifdef KAI2021_WINDOWS
	m_draw_images_horz(cxs, m_resolution, 3);
	m_show_select_results(est_domain, ans_domain, m_target_names[0]);
	m_show_select_results(est_product, ans_product, m_target_names[1]);
#endif
}
