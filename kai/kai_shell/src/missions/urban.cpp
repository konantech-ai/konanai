/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "urban.h"

UrbanSoundMission::UrbanSoundMission(KHSession hSession, enum Ken_test_level testLevel) : Mission(hSession, testLevel), m_urbanFeeder(hSession) {
}

UrbanSoundMission::~UrbanSoundMission() {
}

void UrbanSoundMission::Execute() {
	srand(1234);
	m_createComponents();
	m_execute("lstm", true);
	m_execute("lstm");
	m_execute("rnn");
	m_execute("gru");
}

void UrbanSoundMission::m_createComponents() {
	KaiDict visTerms{ {"probs", "mult(softmax(@est), 100)"}, {"select", "argmax(@est)"}, {"answer", "argmax(@ans)"} };

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", { {"exp", "mean(softmax_cross_entropy_with_logits(@est,@ans))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", { {"exp", "mean(equal(argmax(@est),argmax(@ans)))"} }));
	KERR_CHK(KAI_Expression_create(m_hSession, &m_hVisExp, "hungarian", { {"terms", visTerms} }));

	// 아담 알고리즘을 사용하면 lstm 신경망이 모두 nan 발생으로 폭주함
	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "sgd", { {"trace_grad_norm", false}, {"clip_grad", 100000.0f},  {"learning_rate", 0.001f} }));

	KERR_CHK(KAI_Dataset_create(m_hSession, &m_hDataset, "feeding", { {"name", "urban dataset"} }));

	m_urbanFeeder.ConnectToKai(m_hSession, m_hDataset);
	m_urbanFeeder.loadData(ms_data_root+"urban-sound-classification", ms_cache_root+"urban-sound");
}

KHNetwork UrbanSoundMission::m_buildNetwork(KString sCellType, KBool use_state) {
	KHNetwork hNetwork = 0;

	KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "serial", { {"name", "rnn"}, {"use_output_layer", true} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, sCellType, { {"width", 64}, {"input_seq", true} , {"output_seq", false}, {"use_state", use_state} }));

	return hNetwork;
}

void UrbanSoundMission::m_execute(KString sCellType, KBool use_state) {
	KString sModelName = "Urban Sound " + sCellType;
	if (use_state) sModelName += "(use_state)";

	printf("\n*** %s Model ***\n\n", sModelName.c_str());

	KHModel hModel = 0;
	KHNetwork hNetwork = m_buildNetwork(sCellType, use_state);

	KERR_CHK(KAI_Model_create(m_hSession, &hModel, "basic", 
		{ {"name", sModelName}, {"dataset", m_hDataset},
		  {"network", hNetwork}, {"loss_exp", m_hLossExp}, {"accuracy_exp", m_hAccExp}, {"visualize_exp", m_hVisExp},
		  {"optimizer", m_hOptimizer}, {"clip_grad", 0.0f} }));

	KaiList targetNames = m_urbanFeeder.getTargetNames();

	m_reporter.ConnectToKai(m_hSession, hModel, KCb_mask_all);
	m_reporter.setTargetNames(targetNames);

	KERR_CHK(KAI_Model_train(hModel, { {"epoch_count", 10}, {"epoch_report", 1}, {"epoch_validate", 5}, {"learning_rate", 0.0001f}}));
	KERR_CHK(KAI_Model_test(hModel, {}));
	KERR_CHK(KAI_Model_visualize(hModel, {}));
}
