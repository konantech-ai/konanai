/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "steel.h"
#include "../../../kai_engine/src/nightly/nightly_utils.h"

// One-hot vector conversion settings : Convert from the column 0 to 3-size one-hot vector.
//#define STEEL_MISSION_TO_ONEHOT          KaiList{M,N}	// No use

// Converted data structure : {begin_index, data_count}
#define STEEL_MISSION_INPUT_COLUMNS      KaiList{ {0,27} }
#define STEEL_MISSION_OUTPUT_COLUMNS     KaiList{ {27,7} }

#define STEEL_MISSION_INPUT_SHAPE        KaiShape{27}
#define STEEL_MISSION_OUTPUT_SHAPE       KaiShape{7}

#define STEEL_MISSION_INPUT_SHAPE_1D     STEEL_MISSION_INPUT_SHAPE
#define STEEL_MISSION_OUTPUT_SHAPE_1D    STEEL_MISSION_OUTPUT_SHAPE

// Paths
#define STEEL_MISSION_PATH_DATA     ms_data_root + "chap03/faults.csv"
#define STEEL_MISSION_PATH_CACHE    ms_cache_root + "steel"

SteelMission::SteelMission(KHSession hSession, enum Ken_test_level testLevel) : Mission(hSession, testLevel), m_reporter() {
}

SteelMission::~SteelMission() {
}

void SteelMission::Execute() {
	srand(1234);
	m_createComponents();
	m_execute("mlp");
}

void SteelMission::m_createComponents() {
	KERR_CHK(KAI_Dataset_create(m_hSession, &m_hDataset, "feeding", { {"name", "steel_dataset"}}));

	m_dataFeeder.ConnectToKai(m_hSession, m_hDataset);

	/*// Urban style
	KERR_CHK(KAI_Dataloader_create(m_hSession, &m_hDataloader, "plain", {}));
	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", { {"trace_grad_norm", false}, {"clip_grad", 100000.0f},  {"learning_rate", 0.001f} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", { {"exp", "mean(square(sub(@est,@ans)))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", { {"exp", "sub(1.0, div(sqrt(mean(square(sub(@est,@ans)))), mean(@ans)))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hEstExp, "hungarian", { {"exp", "@est"} }));
	*/

	// SteelBuiltinMission style
	KaiDict kwArgs;
	KaiDict visTerms;

	kwArgs.clear();
	kwArgs["name"] = "steel_adam";
	kwArgs["ro1"] = 0.9f;
	kwArgs["ro2"] = 0.999f;
	kwArgs["epsilon"] = 1.0e-8f;
	kwArgs["learning_rate"] = 0.001f;
	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", kwArgs));

	kwArgs.clear();
	kwArgs["name"] = "steel_loss";
	kwArgs["exp"] = "mean(softmax_cross_entropy_with_logits(@est,@ans))";
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", kwArgs));

	kwArgs.clear();
	kwArgs["name"] = "steel_acc";
	kwArgs["exp"] = "mean(equal(argmax(softmax(@est)),argmax(softmax(@ans))))";
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", kwArgs));

	visTerms["probs"] = "mult(softmax(@est), 100)";
	visTerms["select"] = "argmax(@est)";
	visTerms["answer"] = "argmax(@ans)";

	kwArgs.clear();
	kwArgs["name"] = "steel_visualize";
	kwArgs["terms"] = visTerms;

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hVisExp, "hungarian", kwArgs));

	// Predict
	kwArgs.clear();
	kwArgs["name"] = "steel_predict";
	kwArgs["terms"] = visTerms;
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hPredExp, "hungarian", kwArgs));

	// Set properties to initialize the data feeder
	kwArgs.clear();
	kwArgs["name"] = "steel_data_feeder";

	// If the first row is header, then a value of "header_exist" key is true.
	kwArgs["header_exist"] = true;

	// Normalization
	kwArgs["input_normalize"] = true;
	kwArgs["temp_input_columns"] = 27;	// input_columns 대상을 알려주기 위한 임시조치임, 앞으로 dataset-dataloader 기능을 통합하고 datafeeder 형태만 지원할 예정이므로 입력정규화는 에코시스템이 책임질 부분임

	// One-hot vector conversion settings : Convert from the column 0 to 3-size one-hot vector.
#ifdef STEEL_MISSION_TO_ONEHOT
	KaiList onehotList;
	onehotList.push_back(STEEL_MISSION_TO_ONEHOT);
	kwArgs["to_onehot"] = onehotList;
#endif

	// Define the converted data structure : {begin_index, data_count}
	kwArgs["input_columns"] = STEEL_MISSION_INPUT_COLUMNS;
	kwArgs["output_columns"] = STEEL_MISSION_OUTPUT_COLUMNS;

	// Shuffle option
#if (ACTIVATE_TEST && TEST_DISABLE_SHUFFLE)
	// no shuffle
	kwArgs["data_split"] = "sequential";
#else
	// shuffle (default)
	kwArgs["data_split"] = "random";
#endif

	// Data split ratio (train | test | validate)
	kwArgs["tr_ratio"] = 0.8f;
	kwArgs["te_ratio"] = 0.1f;
	kwArgs["va_ratio"] = 0.1f;

	// Each batch size (train | test | validate)
	kwArgs["tr_batch_size"] = 3;
	kwArgs["te_batch_size"] = 20;
	kwArgs["va_batch_size"] = 5;

	// Whether to use cache
	kwArgs["load_cache"] = false;
	kwArgs["save_cache"] = true;

	m_dataFeeder.loadData(STEEL_MISSION_PATH_DATA, STEEL_MISSION_PATH_CACHE, kwArgs, m_hSession, m_hDataset);
}

KHNetwork SteelMission::m_buildNetwork(KString sCellType) {
	KHNetwork hNetwork = 0;

	// MnistReaderMission style
	//KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "serial", { {"name", "cnn"}, {"use_output_layer", true} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 128} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 32} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 4} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, sCellType, { {"width", 27}, {"input_seq", false} , {"output_seq", true} }));

	// Multi-layer perceptron
	KaiDict kwArgs;
	kwArgs["name"] = "steel_mlp";
	kwArgs["widths"] = KaiList{ 64, 16 };
	kwArgs["actfunc"] = "relu";
	kwArgs["input_shape"] = STEEL_MISSION_INPUT_SHAPE_1D;
	kwArgs["output_shape"] = STEEL_MISSION_OUTPUT_SHAPE_1D;
	kwArgs["use_output_layer"] = true;
	//kwArgs["init_weight"] = "gaussian";
	kwArgs["init_weight"] = "adaptive_gaussian";
	kwArgs["init_std"] = 0.03f;
	KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "mlp", kwArgs));

	return hNetwork;
}

void SteelMission::m_execute(KString sCellType) {
	KString sModelName = "Steel " + sCellType;

	printf("\n*** %s Model ***\n\n", sModelName.c_str());

	KHModel hModel = 0;
	KHNetwork hNetwork = m_buildNetwork(sCellType);

	KERR_CHK(KAI_Model_create(m_hSession, &hModel, "basic",
		{ {"name", sModelName}, {"dataset", m_hDataset}, {"network", hNetwork},
		  {"loss_exp", m_hLossExp}, {"accuracy_exp", m_hAccExp}, {"visualize_exp", m_hVisExp}, {"predict_exp", m_hPredExp},
		  {"optimizer", m_hOptimizer} }));

	// Connect to report
	KaiList targetNames = m_dataFeeder.getOuputFieldNames();

	m_reporter.ConnectToKai(m_hSession, hModel, KCb_mask_all);
	m_reporter.setTargetNames(targetNames);

	//KERR_CHK(KAI_Model_visualize(hModel, {}));
	// Original steel style
	//KERR_CHK(KAI_Model_train(hModel, { {"batch_size", 1}, {"epoch_count", 10}, {"epoch_report", 0}, {"epoch_validate", 2}, {"epoch_visualize", 2}, {"visualize_count", 3} }));

	// Pulsar style
	KERR_CHK(KAI_Model_train(hModel, { {"batch_size", 10}, {"epoch_count", 10}, {"epoch_report", 1}, {"epoch_validate", 5}, {"validate_count", 50}, {"learning_rate", 0.0001f} }));
	KERR_CHK(KAI_Model_test(hModel, {}));
	KERR_CHK(KAI_Model_visualize(hModel, {}));

	// Error
	//KERR_CHK(KAI_Model_visualize(hModel, {}));

	/*// Predict
		kwArgs.clear();

	KaiDict fields;
	KaiList userdata, vector;
	KaiList predict;

	// 원시 csv 파일 헤더명에 따른 필드정보, 하나의 데이터 처리
	kwArgs["input_multiple"] = false;
	kwArgs["input_format"] = "csv_vector";
	kwArgs["visualize"] = true;

	KFloat dat[] = { 42, 50, 270900, 270944, 267, 17, 44, 24220, 76, 108, 1687, 1, 0, 80, 0.0498f, 0.2415f, 0.1818f, 0.0047f, 0.4706f, 1, 1, 2.4265f, 0.9031f, 1.6435f, 0.8182f, -0.2913f, 0.5822f };

	for (KInt n = 0; n < sizeof(dat) / sizeof(dat[0]); n++) {
		vector.push_back(dat[n]);
	}

	kwArgs["userdata"] = vector;

	KERR_CHK(KAI_Model_predict(m_hModel, kwArgs, &predict));
	KERR_CHK(KAI_value_dump(predict, "predict"));
	*/
}
