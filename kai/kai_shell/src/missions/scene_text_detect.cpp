/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "scene_text_detect.h"

/*
SceneTextDetectMission::SceneTextDetectMission(KHSession hSession, KString sub_model, enum Ken_test_level testLevel) : Mission(hSession, testLevel), m_feeder(sub_model), m_reporter(sub_model) {
	m_sub_model = sub_model;
}

SceneTextDetectMission::~SceneTextDetectMission() {
}

void SceneTextDetectMission::Execute() {
	//srand(1234);
	m_createComponents();
	m_execute();
}

void SceneTextDetectMission::m_createComponents() {
	KERR_CHK(KAI_Dataset_create(m_hSession, &m_hDataset, "feeding", { {"name", "scene_text_detect dataset"} }));

	m_feeder.ConnectToKai(m_hSession, m_hDataset);

	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", { {"trace_grad_norm", false}, {"clip_grad", 100000.0f},  {"learning_rate", 0.001f} }));

	KaiDict lossTerms;
	KaiDict accTerms;
	KaiDict subTerms;

	lossTerms["xy"] = "div(sum(mult(square(sub(%pred_xy_seed, %true_xy_seed)), %coef_box)), #data_count)";
	lossTerms["wh"] = "div(sum(mult(square(sub(%pred_wh_log, %true_wh_log)), %coef_box)), #data_count)";
	lossTerms["conf"] = "div(sum(mult(add(%conf_loss_pos, %conf_loss_neg), %coef_conf)), #data_count)";
	lossTerms["class"] = "div(sum(mult(sigmoid_cross_entropy_with_logits(%pred_class_logits, %true_class), %coef_class)), #data_count)";

	accTerms["xy"] = "mean(square(sub(%pred_xy_seed, %true_xy_seed)))";
	accTerms["wh"] = "mean(square(sub(%pred_wh_log, %true_wh_log)))";
	accTerms["conf"] = "mean(square(%conf_loss_pos))";
	accTerms["class"] = "mean(square(equal(argmax(%pred_wh_log), argmax(%true_wh_log))))";

	subTerms["conf_loss_pos"] = "mult(sigmoid_cross_entropy_with_logits(%pred_conf_logit, %object_mask), %conf_pos_mask)";
	subTerms["conf_loss_neg"] = "mult(sigmoid_cross_entropy_with_logits(%pred_conf_logit, %object_mask), %conf_neg_mask)";

	subTerms["coef_box"] = "mult(%coef_conf, %box_loss_scale)";
	subTerms["coef_class"] = "mult(%object_mask, %mix_w)";
	subTerms["coef_conf"] = "%mix_w";

	subTerms["object_mask"] = "%true_conf";
	subTerms["conf_pos_mask"] = "%object_mask";
	subTerms["conf_neg_mask"] = "mult(sub(1.0, %object_mask), %ignore_mask)";

	subTerms["ignore_mask"] = "le(max_col(iou(%pred_box,%true_box)), 0.5)";
	subTerms["pred_box"] = "vstack(%left,%right,%top,%bottom)";

	subTerms["left"] = "add(%center_x, div(%width, 2.0))";
	subTerms["right"] = "sub(%center_x, div(%width, 2.0))";
	subTerms["top"] = "add(%center_y, div(%height, 2.0))";
	subTerms["bottom"] = "sub(%center_y, div(%height, 2.0))";

	subTerms["center_x"] = "mult(add(subvector(%pred_xy_seed,0,1), %grid_x), %grid_size)";
	subTerms["center_y"] = "mult(add(subvector(%pred_xy_seed,1,1), %grid_y), %grid_size)";
	subTerms["width"] = "mult(exp(subvector(%pred_wh_log,0,1)), %anchor_width)";
	subTerms["height"] = "mult(exp(subvector(%pred_wh_log,1,1)), %anchor_height)";

	subTerms["pred_xy_seed"] = "sigmoid(subvector(%pred_map,0,2))";
	subTerms["pred_wh_log"] = "subvector(%pred_map,2,2)";
	subTerms["pred_conf_logit"] = "subvector(%pred_map,4,1)";
	subTerms["pred_class_logits"] = "subvector(%pred_map,5,80)";

	subTerms["true_xy_seed"] = "subvector(%true_map,0,2)";
	subTerms["true_wh_log"] = "subvector(%true_map,2,2)";
	subTerms["true_conf"] = "subvector(%true_map,4,1)";
	subTerms["true_class"] = "subvector(%true_map,5,80)";

	subTerms["mix_w"] = "subvector(%true_map,85,1)";
	subTerms["box_loss_scale"] = "subvector(%true_map,86,1)";
	subTerms["grid_x"] = "subvector(%true_map,87,1)";
	subTerms["grid_y"] = "subvector(%true_map,88,1)";
	subTerms["grid_size"] = "subvector(%true_map,89,1)";
	subTerms["anchor_width"] = "subvector(%true_map,90,1)";
	subTerms["anchor_height"] = "subvector(%true_map,91,1)";

	subTerms["pred_map"] = "@est";
	subTerms["true_map"] = "@ans:true_map";
	subTerms["true_box"] = "@ans:true_box";

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", { {"terms", lossTerms}, {"subterms", subTerms} }));

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", { {"terms", accTerms}, {"subterms", subTerms} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hVisExp, "hungarian", { {"exp", "softmax(@est)"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hPredExp, "hungarian", { {"exp", "softmax(@est)"} }));

	m_feeder.loadData(ms_data_root + "coco-2014", ms_cache_root + "coco-2014");
}

KHNetwork SceneTextDetectMission::m_buildNetwork() {
	KHNetwork hNetwork = 0;
	KHNetwork hConvUnit = 0;
	KHNetwork hResUnit = 0;
	KHNetwork hResChain = 0;
	KHNetwork hYoloHead = 0;
	KHNetwork hYoloMerge = 0;
	KHNetwork hMergeBranch = 0;
	KHNetwork hYoloV3 = 0;

	KERR_CHK(KAI_Network_create(m_hSession, &hConvUnit, "serial", { }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hConvUnit, "conv", { {"ksize", #ksize"}, {"chn", "#chn"}, {"stride", "#stride"1}, {"actfunc", "none"}}));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hConvUnit, "batchnormal", { {"decay", 0.999}, {"epsilon", 1e-5}, {"scale", true} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hConvUnit, "activate", { {"actfunc", "relu"} }));
	KERR_CHK(KAI_Network_regist_macro(m_hSession, hConvUnit, "conv_unit"));

	KERR_CHK(KAI_Network_create(m_hSession, &hVggUnit, "add", { "subnet", "serial" }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hResUnit, "DBL", { {"ksize", 1}, {"stride", 1}, {"chn", "#chn"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hResUnit, "DBL", { {"ksize", 3}, {"stride", 1}, {"chn", "#chn * 2"} }));
	KERR_CHK(KAI_Network_regist_macro(m_hSession, hResUnit, "res_unit"));

	KERR_CHK(KAI_Network_create(m_hSession, &hResChain, "serial", {}));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hResChain, "DBL", { {"ksize", 3}, {"stride", 2}, {"chn", "#chn*2"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hResChain, "res_unit", { {"repeat", "#num"}, {"set", "#set"}, {"chn", "#chn"} })); // set 옵션이 마지막 실행 때만 적용되는지 확인
	KERR_CHK(KAI_Network_regist_macro(m_hSession, hResChain, "res_chain"));

	KERR_CHK(KAI_Network_create(m_hSession, &hYoloHead, "serial", {}));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloHead, "DBL", { {"ksize", 1}, {"stride", 1}, {"chn", "#chn"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloHead, "DBL", { {"ksize", 3}, {"stride", 1}, {"chn", "#chn*2"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloHead, "DBL", { {"ksize", 1}, {"stride", 1}, {"chn", "#chn"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloHead, "DBL", { {"ksize", 3}, {"stride", 1}, {"chn", "#chn*2"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloHead, "DBL", { {"ksize", 1}, {"stride", 1}, {"chn", "#chn"}, { "set", "#inter"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloHead, "DBL", { {"ksize", 3}, {"stride", 1}, {"chn", "#chn*2"} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hYoloHead, "conv", { {"ksize", 1}, {"stride", 1}, {"chn", "#vec_size"}, {"actfunc", "none"}, {"set", "#map"} }));
	KERR_CHK(KAI_Network_regist_macro(m_hSession, hYoloHead, "yolo_head"));

	KERR_CHK(KAI_Network_create(m_hSession, &hMergeBranch, "serial", {}));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hMergeBranch, "DBL", { {"ksize", 1}, {"stride", 1}, {"chn", "#chn"}, {"get", "#up_sample"} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hMergeBranch, "expand", { {"ratio", 2} }));

	KERR_CHK(KAI_Network_create(m_hSession, &hYoloMerge, "parallel", {}));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hYoloMerge, "pass", { {"get", "#direct"} }));
	KERR_CHK(KAI_Network_append_subnet(m_hSession, hYoloMerge, hMergeBranch));
	KERR_CHK(KAI_Network_regist_macro(m_hSession, hYoloMerge, "yolo_merge"));

	KERR_CHK(KAI_Network_create(m_hSession, &hYoloV3, "serial", {}));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "DBL", { {"ksize", 3}, {"stride", 1}, {"chn", 32} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "res_chain", { {"num", "#num1"}, {"chn", 32}, {"set", ""} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "res_chain", { {"num", "#num2"}, {"chn", 64}, {"set", ""} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "res_chain", { {"num", "#num3"}, {"chn", 128}, {"set", "route_1"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "res_chain", { {"num", "#num4"}, {"chn", 256}, {"set", "route_2"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "res_chain", { {"num", "#num5"}, {"chn", 512}, {"set", ""} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "yolo_head", { {"chn", 512}, {"vec_size", "#vec_size"}, {"inter", "route_3"}, {"map", "feature_map_1"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "yolo_merge", { {"chn", 256}, {"direct", "route_2"}, {"up_sample", "route_3"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "yolo_head", { {"chn", 256}, {"vec_size", "#vec_size"}, {"inter", "inter_1"}, {"map", "feature_map_2"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "yolo_merge", { {"chn", 128}, {"direct", "route_1"}, {"up_sample", "inter_1"} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hYoloV3, "yolo_head", { {"chn", 128}, {"vec_size", "#vec_size"}, {"inter", ""}, {"map", "feature_map_3"} }));
	KERR_CHK(KAI_Network_regist_macro(m_hSession, hYoloV3, "yolo_v3"));

	KaiDict kwArgs;

	kwArgs["vec_size"] = 255;

	if (m_sub_model == "large") {
		kwArgs["num1"] = 1;
		kwArgs["num2"] = 2;
		kwArgs["num3"] = 8;
		kwArgs["num4"] = 8;
		kwArgs["num5"] = 4;
	}
	else if (m_sub_model == "medium") {
		kwArgs["num1"] = 1;
		kwArgs["num2"] = 2;
		kwArgs["num3"] = 4;
		kwArgs["num4"] = 4;
		kwArgs["num5"] = 2;
	}
	else if (m_sub_model == "small") {
		kwArgs["num1"] = 1;
		kwArgs["num2"] = 1;
		kwArgs["num3"] = 1;
		kwArgs["num4"] = 1;
		kwArgs["num5"] = 1;
	}

	KaiList maps{ "feature_map_1", "feature_map_2", "feature_map_3" };
	KaiShape tail_shape{ 85 };

	KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "serial", { {"name", "yolo_v3"}, {"use_output_layer", false} }));
	KERR_CHK(KAI_Network_append_custom_layer(m_hSession, hNetwork, "yolo_v3", kwArgs));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "stack", { {"collect", maps}, {"ignore_input", true}, {"tail_shape", tail_shape} }));

	return hNetwork;
}

void SceneTextDetectMission::m_execute() {
	KString sModelName = "Yolo3 " + m_sub_model;

	printf("\n*** %s Model ***\n\n", sModelName.c_str());

	KHNetwork hNetwork = m_buildNetwork();

	KaiDict kwArgs{ { "name", sModelName}, { "dataset", m_hDataset }, { "network", hNetwork },
					{ "loss_exp", m_hLossExp }, { "accuracy_exp", m_hAccExp }, { "visualize_exp", m_hVisExp }, { "predict_exp", m_hPredExp },
					{ "optimizer", m_hOptimizer }, { "clip_grad", 0.0f } };

	if (m_sub_model == "mini") {
		KaiDict debug_trace;
		debug_trace["phase"] = KaiList{ "forward", "backprop" };
		debug_trace["targets"] = KaiList{ };
		debug_trace["checked"] = KaiList{  };

		kwArgs["debug_trace"] = debug_trace;
	}

	KERR_CHK(KAI_Model_create(m_hSession, &m_hModel, "basic", kwArgs));

	m_reporter.ConnectToKai(m_hSession, m_hModel, KCb_mask_all);

	KERR_CHK(KAI_Model_train(m_hModel, { {"epoch_count", 10}, {"epoch_report", 1}, {"epoch_validate", 5}, {"learning_rate", 0.001f} }));
	KERR_CHK(KAI_Model_test(m_hModel, {}));
	KERR_CHK(KAI_Model_visualize(m_hModel, {}));
}
*/