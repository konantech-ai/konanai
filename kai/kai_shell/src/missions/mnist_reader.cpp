/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "mnist_reader.h"

MnistReaderMission::MnistReaderMission(KHSession hSession, enum Ken_test_level testLevel) : Mission(hSession, testLevel) {
}

MnistReaderMission::~MnistReaderMission() {
}

void MnistReaderMission::Execute() {
	srand(1234);
	m_createComponents();
	m_execute("gru");
	//m_execute("lstm");
	m_execute("rnn");
	m_execute("gru");
	m_execute("lstm");
}

void MnistReaderMission::m_createComponents() {
	KaiDict accTerms, visTerms;

	accTerms["char"] = "mean(equal(argmax(@est),argmax(@ans)))";
	accTerms["word"] = "mean(equal_col(argmax(@est),argmax(@ans)))";

	visTerms["char_probs"] = "mult(softmax(@est), 100)";
	visTerms["char_select"] = "argmax(@est)";
	visTerms["char_answer"] = "argmax(@ans)";

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", { {"exp", "mean(softmax_cross_entropy_with_logits(@est,@ans))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", { {"terms", accTerms} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hVisExp, "hungarian", { {"terms", visTerms} }));

	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", { {"trace_grad_norm", false}, {"clip_grad", 100000.0f},  {"learning_rate", 0.001f} }));

	KERR_CHK(KAI_Dataset_create(m_hSession, &m_hDataset, "feeding", { {"name", "mnist reader dataset"}, {"visualize_batch", true} }));

	m_dataFeeder.ConnectToKai(m_hSession, m_hDataset);

	m_dataFeeder.loadData(ms_data_root + "mnist", ms_cache_root + "mnist_reader");
}

KHNetwork MnistReaderMission::m_buildNetwork(KString sCellType) {
	KHNetwork hNetwork = 0;

	KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "serial", { {"name", "rnn"}, {"use_output_layer", true} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 128} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 32} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", 4} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, sCellType, { {"width", 27}, {"input_seq", false} , {"output_seq", true}, {"timesteps", 6}, {"forget_bias", 0.5f} }));

	return hNetwork;
}

void MnistReaderMission::m_execute(KString sCellType) {
	KString sModelName = "Mnist Reader " + sCellType;

	printf("\n*** %s Model ***\n\n", sModelName.c_str());

	KHModel hModel = 0;
	KHNetwork hNetwork = m_buildNetwork(sCellType);

	KERR_CHK(KAI_Model_create(m_hSession, &hModel, "basic",
		{ {"name", sModelName}, {"dataset", m_hDataset},
		  {"network", hNetwork}, {"loss_exp", m_hLossExp}, {"accuracy_exp", m_hAccExp}, {"visualize_exp", m_hVisExp},
		  {"optimizer", m_hOptimizer}, {"clip_grad", 0.0f} }));

	m_reporter.ConnectToKai(m_hSession, hModel, KCb_mask_all);

	//KERR_CHK(KAI_Model_visualize(hModel, {}));
	KERR_CHK(KAI_Model_train(hModel, { {"batch_size", 2}, {"epoch_count", 10}, {"epoch_report", 1}, {"epoch_validate", 5}, {"learning_rate", 0.001f} }));
	KERR_CHK(KAI_Model_test(hModel, {}));
	KERR_CHK(KAI_Model_visualize(hModel, {}));
}
