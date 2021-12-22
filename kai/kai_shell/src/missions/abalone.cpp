/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "abalone.h"
#include "../../../kai_engine/src/nightly/nightly_utils.h"

// One-hot vector conversion settings : Convert from the column 0 to 3-size one-hot vector.
#define ABALONE_MISSION_TO_ONEHOT          KaiList{0,3}

// Converted data structure : {begin_index, data_count}
#define ABALONE_MISSION_INPUT_COLUMNS      KaiList{ {0,10} }
#define ABALONE_MISSION_OUTPUT_COLUMNS     KaiList{ {10,1} }

#define ABALONE_MISSION_INPUT_SHAPE        KaiShape{10}
#define ABALONE_MISSION_OUTPUT_SHAPE       KaiShape{1}

#define ABALONE_MISSION_INPUT_SHAPE_1D     ABALONE_MISSION_INPUT_SHAPE
#define ABALONE_MISSION_OUTPUT_SHAPE_1D    ABALONE_MISSION_OUTPUT_SHAPE

// Paths
#define ABALONE_MISSION_PATH_DATA     ms_data_root + "chap01/abalone.csv"
#define ABALONE_MISSION_PATH_CACHE    ms_cache_root + "abalone"

AbaloneMission::AbaloneMission(KHSession hSession, enum Ken_test_level testLevel) : Mission(hSession, testLevel), m_reporter() {
}

AbaloneMission::~AbaloneMission() {
}

void AbaloneMission::Execute() {
	srand(1234);
	m_createComponents();
	m_execute("mlp");
}

void AbaloneMission::m_createComponents() {
	KERR_CHK(KAI_Dataset_create(m_hSession, &m_hDataset, "feeding", { {"name", "abalone_dataset"}}));

	m_dataFeeder.ConnectToKai(m_hSession, m_hDataset);

	/*// Urban style
	KERR_CHK(KAI_Dataloader_create(m_hSession, &m_hDataloader, "plain", {}));
	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", { {"trace_grad_norm", false}, {"clip_grad", 100000.0f},  {"learning_rate", 0.001f} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", { {"exp", "mean(square(sub(@est,@ans)))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", { {"exp", "sub(1.0, div(sqrt(mean(square(sub(@est,@ans)))), mean(@ans)))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hEstExp, "hungarian", { {"exp", "@est"} }));
	*/

	// AbaloneBuiltinMission style
	KaiDict kwArgs;

	kwArgs.clear();
	kwArgs["name"] = "abalone_adam";
	kwArgs["ro1"] = 0.9f;
	kwArgs["ro2"] = 0.999f;
	kwArgs["epsilon"] = 1.0e-8f;
	kwArgs["learning_rate"] = 0.001f;
	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", kwArgs));

	kwArgs.clear();
	kwArgs["name"] = "abalone_loss";
	kwArgs["exp"] = "mean(square(sub(@est,@ans)))";
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", kwArgs));

	kwArgs.clear();
	kwArgs["name"] = "abalone_acc";
	kwArgs["exp"] = "sub(1.0, div(sqrt(mean(square(sub(@est,@ans)))), mean(@ans)))";
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", kwArgs));

	kwArgs.clear();
	kwArgs["name"] = "abalone_visualize";
	kwArgs["exp"] = "@est";
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hVisExp, "hungarian", kwArgs));

	// Set properties to initialize the data feeder
	kwArgs.clear();
	kwArgs["name"] = "abalone_data_feeder";

	// If the first row is header, then a value of "header_exist" key is true.
	kwArgs["header_exist"] = true;

	// One-hot vector conversion settings : Convert from the column 0 to 3-size one-hot vector.
#ifdef ABALONE_MISSION_TO_ONEHOT
	KaiList onehotList;
	onehotList.push_back(ABALONE_MISSION_TO_ONEHOT);
	kwArgs["to_onehot"] = onehotList;
#endif

	// Define the converted data structure : {begin_index, data_count}
	kwArgs["input_columns"] = ABALONE_MISSION_INPUT_COLUMNS;
	kwArgs["output_columns"] = ABALONE_MISSION_OUTPUT_COLUMNS;

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
	kwArgs["te_ratio"] = 0.15f;
	kwArgs["va_ratio"] = 0.05f;

	// Each batch size (train | test | validate)
	kwArgs["tr_batch_size"] = 10;
	kwArgs["te_batch_size"] = 20;
	kwArgs["va_batch_size"] = 50;

	// Whether to use cache
	kwArgs["load_cache"] = false;
	kwArgs["save_cache"] = true;

	m_dataFeeder.loadData(ABALONE_MISSION_PATH_DATA, ABALONE_MISSION_PATH_CACHE, kwArgs, m_hSession, m_hDataset);
}

KHNetwork AbaloneMission::m_buildNetwork(KString sCellType) {
	KHNetwork hNetwork = 0;

	// MnistReaderMission style
	//KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "serial", { {"name", "cnn"}, {"use_output_layer", true} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 128} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 32} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 4} }));
	//KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, sCellType, { {"width", 27}, {"input_seq", false} , {"output_seq", true} }));

	// Multi-layer perceptron
	KaiDict kwArgs;
	kwArgs["name"] = "abalone_mlp";
	kwArgs["widths"] = KaiList{ 16, 4 };
	kwArgs["actfunc"] = "relu";
	kwArgs["input_shape"] = ABALONE_MISSION_INPUT_SHAPE_1D;
	kwArgs["output_shape"] = ABALONE_MISSION_OUTPUT_SHAPE_1D;
	kwArgs["use_output_layer"] = true;
	//kwArgs["init_weight"] = "gaussian";
	kwArgs["init_weight"] = "adaptive_gaussian";
	kwArgs["init_std"] = 0.03f;
	KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "mlp", kwArgs));

	return hNetwork;
}

void AbaloneMission::m_execute(KString sCellType) {
	KString sModelName = "Abalone " + sCellType;

	printf("\n*** %s Model ***\n\n", sModelName.c_str());

	KHModel hModel = 0;
	KHNetwork hNetwork = m_buildNetwork(sCellType);

	KERR_CHK(KAI_Model_create(m_hSession, &hModel, "basic",
		{ {"name", sModelName}, {"dataset", m_hDataset},
		  {"network", hNetwork}, {"loss_exp", m_hLossExp}, {"accuracy_exp", m_hAccExp}, {"visualize_exp", m_hVisExp},
		  {"optimizer", m_hOptimizer}, {"clip_grad", 0.0f} }));

	m_reporter.ConnectToKai(m_hSession, hModel, KCb_mask_all);

	KERR_CHK(KAI_Model_train(hModel, { {"batch_size", 10}, {"epoch_count", 10}, {"epoch_report", 1}, {"epoch_validate", 5}, {"learning_rate", 0.0001f} }));
	KERR_CHK(KAI_Model_test(hModel, {}));
	KERR_CHK(KAI_Model_visualize(hModel, {}));
}
