/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
// Reference : (Confusion matrix, precision, recall, accuracy, F1 score)
// https://sumniya.tistory.com/26

#include "water.h"
#include "../../../kai_engine/src/nightly/nightly_utils.h"

// One-hot vector conversion settings : Convert from the column 0 to 3-size one-hot vector.
//#define WATER_MISSION_TO_ONEHOT          KaiList{M,N}	// No use

// Converted data structure : {begin_index, data_count}
#define WATER_MISSION_INPUT_COLUMNS      KaiList{ {0,9} }
#define WATER_MISSION_OUTPUT_COLUMNS     KaiList{ {9,1} }

#define WATER_MISSION_INPUT_SHAPE        KaiShape{9}
#define WATER_MISSION_OUTPUT_SHAPE       KaiShape{1}

#define WATER_MISSION_INPUT_SHAPE_1D     WATER_MISSION_INPUT_SHAPE
#define WATER_MISSION_OUTPUT_SHAPE_1D    WATER_MISSION_OUTPUT_SHAPE

// Paths
#define WATER_MISSION_PATH_DATA     ms_data_root + "chap02/water_potability.csv"
#define WATER_MISSION_PATH_CACHE    ms_cache_root + "water"

WaterMission::WaterMission(KHSession hSession, enum Ken_test_level testLevel) : Mission(hSession, testLevel) {
}

WaterMission::~WaterMission() {
}

void WaterMission::Execute() {
	srand(1234);
	m_createComponents();
	m_execute("mlp");
}

void WaterMission::m_createComponents() {
	KERR_CHK(KAI_Dataset_create(m_hSession, &m_hDataset, "feeding", { {"name", "water_dataset"}}));

	m_dataFeeder.ConnectToKai(m_hSession, m_hDataset);

	/*// Urban style
	KERR_CHK(KAI_Dataloader_create(m_hSession, &m_hDataloader, "plain", {}));
	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", { {"trace_grad_norm", false}, {"clip_grad", 100000.0f},  {"learning_rate", 0.001f} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", { {"exp", "mean(square(sub(@est,@ans)))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", { {"exp", "sub(1.0, div(sqrt(mean(square(sub(@est,@ans)))), mean(@ans)))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hEstExp, "hungarian", { {"exp", "@est"} }));
	*/

	// WaterBuiltinMission style
	KaiDict kwArgs;

	kwArgs.clear();
	kwArgs["name"] = "water_adam";
	kwArgs["ro1"] = 0.9f;
	kwArgs["ro2"] = 0.999f;
	kwArgs["epsilon"] = 1.0e-8f;
	kwArgs["learning_rate"] = 0.001f;
	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", kwArgs));

	kwArgs.clear();
	kwArgs["name"] = "water_loss";
	kwArgs["exp"] = "mean(sigmoid_cross_entropy_with_logits(@est,@ans))";
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", kwArgs));

	kwArgs.clear();

	KaiDict accTerms;
	accTerms["TP"] = "mean(and(gt(@est,0),gt(@ans,0.5)))";
	accTerms["FN"] = "mean(and(le(@est,0),gt(@ans,0.5)))";
	accTerms["FP"] = "mean(and(gt(@est,0),le(@ans,0.5)))";
	accTerms["TN"] = "mean(and(le(@est,0),le(@ans,0.5)))";

	kwArgs["name"] = "water_acc";
	kwArgs["terms"] = accTerms;

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", kwArgs));

	kwArgs.clear();
	kwArgs["name"] = "water_visualize";
	kwArgs["exp"] = "mult(sigmoid(@est),100)";
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hVisExp, "hungarian", kwArgs));

	// Set properties to initialize the data feeder
	kwArgs.clear();
	kwArgs["name"] = "water_data_feeder";

	// If the first row is header, then a value of "header_exist" key is true.
	kwArgs["header_exist"] = true;

	// Normalization
	kwArgs["input_normalize"] = true;
	kwArgs["temp_input_columns"] = 9;	// input_columns 대상을 알려주기 위한 임시조치임, 앞으로 dataset-dataloader 기능을 통합하고 datafeeder 형태만 지원할 예정이므로 입력정규화는 에코시스템이 책임질 부분임

	// One-hot vector conversion settings : Convert from the column 0 to 3-size one-hot vector.
#ifdef WATER_MISSION_TO_ONEHOT
	KaiList onehotList;
	onehotList.push_back(WATER_MISSION_TO_ONEHOT);
	kwArgs["to_onehot"] = onehotList;
#endif

	// Define the converted data structure : {begin_index, data_count}
	kwArgs["input_columns"] = WATER_MISSION_INPUT_COLUMNS;
	kwArgs["output_columns"] = WATER_MISSION_OUTPUT_COLUMNS;

	// Shuffle option
#if (ACTIVATE_TEST && TEST_DISABLE_SHUFFLE)
	// no shuffle
	kwArgs["data_split"] = "sequential";
#else
	// shuffle (default)
	kwArgs["data_split"] = "random";
#endif

	// Data split ratio (train | test | validate)
	kwArgs["tr_ratio"] = 0.9f;
	kwArgs["te_ratio"] = 0.05f;
	kwArgs["va_ratio"] = 0.05f;

	// Each batch size (train | test | validate)
	kwArgs["tr_batch_size"] = 20;
	kwArgs["te_batch_size"] = 20;
	kwArgs["va_batch_size"] = 20;

	// Whether to use cache
	kwArgs["load_cache"] = false;
	kwArgs["save_cache"] = true;

	m_dataFeeder.loadData(WATER_MISSION_PATH_DATA, WATER_MISSION_PATH_CACHE, kwArgs, m_hSession, m_hDataset);
}

KHNetwork WaterMission::m_buildNetwork(KString sCellType) {
	KHNetwork hNetwork = 0;

	// MnistReaderMission style
	//KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "serial", { {"name", "cnn"}, {"use_output_layer", true} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 128} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 32} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 4} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, sCellType, { {"width", 27}, {"input_seq", false} , {"output_seq", true} }));

	// Multi-layer perceptron
	KaiDict kwArgs;
	kwArgs["name"] = "water_mlp";
	kwArgs["widths"] = KaiList{ 16, 4 };
	kwArgs["actfunc"] = "relu";
	kwArgs["input_shape"] = WATER_MISSION_INPUT_SHAPE_1D;
	kwArgs["output_shape"] = WATER_MISSION_OUTPUT_SHAPE_1D;
	kwArgs["use_output_layer"] = true;
	//kwArgs["init_weight"] = "gaussian";
	kwArgs["init_weight"] = "adaptive_gaussian";
	kwArgs["init_std"] = 0.03f;
	KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "mlp", kwArgs));

	return hNetwork;
}

void WaterMission::m_execute(KString sCellType) {
	KString sModelName = "Water " + sCellType;

	printf("\n*** %s Model ***\n\n", sModelName.c_str());

	KHModel hModel = 0;
	KHNetwork hNetwork = m_buildNetwork(sCellType);

	KERR_CHK(KAI_Model_create(m_hSession, &hModel, "basic",
		{ {"name", sModelName}, {"dataset", m_hDataset},
		  {"network", hNetwork}, {"loss_exp", m_hLossExp}, {"accuracy_exp", m_hAccExp}, {"visualize_exp", m_hVisExp},
		  {"optimizer", m_hOptimizer}, {"clip_grad", 3.141592f} }));

	m_reporter.ConnectToKai(m_hSession, hModel, KCb_mask_all);

	//KERR_CHK(KAI_Model_visualize(hModel, {}));
	KERR_CHK(KAI_Model_train(hModel, { {"epoch_count", 10}, {"epoch_report", 1}, {"epoch_validate", 2}, {"epoch_visualize", 5}, {"validate_count", 3} }));
	KERR_CHK(KAI_Model_test(hModel, {}));
	KERR_CHK(KAI_Model_visualize(hModel, {}));
}
