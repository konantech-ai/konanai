/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "bert.h"

BertMission::BertMission(KHSession hSession, KString sub_model, enum Ken_test_level testLevel) : Mission(hSession, testLevel), m_feeder() {
	m_sub_model = sub_model;

	if (m_sub_model == "ptb_large") {
		m_hidden_size = 1024;
		m_stack_depth = 24;
		m_attention_heads = 16;
		m_max_position = 512;
		m_hidden_ex_size = m_hidden_size * 4;
		m_batch_size = 10;
	}
	else if (m_sub_model == "ptb_small") {
		m_hidden_size = 128;
		m_stack_depth = 3;
		m_attention_heads = 4;
		m_max_position = 512;
		m_hidden_ex_size = m_hidden_size * 4;
		m_batch_size = 10;
	}
	else if (m_sub_model == "eng_mini") {
		m_hidden_size = 10;
		m_stack_depth = 2;
		m_attention_heads = 5;
		m_max_position = 12;
		m_hidden_ex_size = m_hidden_size * 2;
		m_batch_size = 2;
	}
	else THROW(KERR_UNIMPEMENTED_YET);

	m_att_dropout_ratio = 0.0f;		// 드롭아웃은 self_attention 레이어의 로직 점검 및 효율적 랜덤 마스크 생성 처리 후 적용하기로 한다.
	m_hid_dropout_ratio = 0.1f;		// 드롭아웃은 self_attention 레이어의 로직 점검 및 효율적 랜덤 마스크 생성 처리 후 적용하기로 한다.

	m_feeder.setMaxPosition(m_max_position);

	m_voc_count = 0;
}

BertMission::~BertMission() {
}

void BertMission::Execute() {
	//srand(1234);
	m_createComponents();
	m_execute();
}

void BertMission::m_createComponents() {
	KERR_CHK(KAI_Dataset_create(m_hSession, &m_hDataset, "feeding", { {"name", "bert dataset"} }));

	m_feeder.setModel(m_sub_model);
	m_feeder.ConnectToKai(m_hSession, m_hDataset);

	KERR_CHK(KAI_Optimizer_create(m_hSession, &m_hOptimizer, "adam", { {"trace_grad_norm", false}, {"clip_grad", 100000.0f},  {"learning_rate", 0.0001f} }));
	
	KaiDict lossTerms, accTerms, visTerms;

	lossTerms["next_sent"] = "mean(softmax_cross_entropy_with_logits(@est:is_next_sent,@ans:next_sent))";
	lossTerms["masked_word"] = "mean(softmax_cross_entropy_with_logits_idx(@est, @ans:masked_words))";

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hLossExp, "hungarian", { {"terms", lossTerms} }));

	accTerms["next_sent"] = "mean(equal(argmax(@est:is_next_sent),argmax(@ans:next_sent)))";
	accTerms["masked_word"] = "mean(equal(argmax(@est), @ans:masked_words))";

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hAccExp, "hungarian", { {"terms", accTerms} }));

	visTerms["next_sent_probs"] = "mult(softmax(@est:is_next_sent),100)";
	visTerms["next_sent_select"] = "argmax(@est:is_next_sent)";
	visTerms["next_sent_answer"] = "argmax(@ans:next_sent)";
	visTerms["masked_word_probs"] = "mult(softmax(@est),100)";
	visTerms["masked_word_select"] = "argmax(@est)";
	visTerms["masked_word_answer"] = "@ans:masked_words";

	KERR_CHK(KAI_Expression_create(m_hSession, &m_hVisExp, "hungarian", { {"terms", visTerms} }));

	//KERR_CHK(KAI_Expression_create(m_hSession, &m_hPredExp, "hungarian", { {"exp", "softmax(@est)"} }));

	m_feeder.loadData(ms_data_root + "bert", ms_cache_root + "bert");
	m_voc_count = m_feeder.voc_count();
}

KHNetwork BertMission::m_buildNetwork() {
	KHNetwork hNetwork = 0;
	KHNetwork hEncoderLoop = 0;
	KHNetwork hAttentionRes = 0;
	KHNetwork hDenseRes = 0;
	KHNetwork hNextSent = 0;

	/*
		info["hidden_size"] = 128;		// 1024 for BERT-large
		info["stack_depth"] = 3;		// 24 for BERT - large
		info["attention_heads"] = 4;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;

			const char* bert_conf = " \
			[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent']}], \
			 ['batch_normal'], \
			 ['serial', {'repeat':%stack_depth%}, \
			  ['add', \
			   ['serial', \
			     ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}], \
			     ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal'], \
			  ['add', \
			   ['serial', {'wrap_axis':0}, \
			    ['full', {'width':%hidden_size%}], \
			    ['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
			    ['full', {'width':%hidden_size%}], \
			    ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal']], \
			 ['pass', \
			  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
			  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
			  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
			 ['serial', {'wrap_axis':0}, \
			   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

			   */
	KaiList embed_info;
	
	embed_info.push_back(KaiDict{ {"name", "wid"}, {"size", m_voc_count} });
	embed_info.push_back(KaiDict{ {"name", "position"}, {"size", m_max_position} });
	embed_info.push_back(KaiDict{ {"name", "sentence"}, {"size", 3} });

	KERR_CHK(KAI_Network_create(m_hSession, &hAttentionRes, "add", { "subnet", "serial" }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hAttentionRes, "self_attention", { {"multi_heads", m_attention_heads}, {"dropout", m_att_dropout_ratio} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hAttentionRes, "dropout", { {"drop_ratio", m_hid_dropout_ratio} }));

	KERR_CHK(KAI_Network_create(m_hSession, &hDenseRes, "add", { "subnet", "serial" }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hDenseRes, "dense", { {"width", m_hidden_size} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hDenseRes, "dense", { {"width", m_hidden_ex_size}, {"actfunc", "gelu"} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hDenseRes, "dense", { {"width", m_hidden_size} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hDenseRes, "dropout", { {"drop_ratio", m_hid_dropout_ratio} }));

	KERR_CHK(KAI_Network_create(m_hSession, &hEncoderLoop, "serial", { {"repeat", m_stack_depth} }));
	KERR_CHK(KAI_Network_append_subnet(m_hSession, hEncoderLoop, hAttentionRes));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hEncoderLoop, "batchnormal", {}));
	KERR_CHK(KAI_Network_append_subnet(m_hSession, hEncoderLoop, hDenseRes));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hEncoderLoop, "batchnormal", {}));

	KERR_CHK(KAI_Network_create(m_hSession, &hNextSent, "pass", {}));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNextSent, "extract", { {"axis", -1}, {"index", 0}, {"reduce_axis", true } }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNextSent, "dense", { {"width", m_hidden_size}, {"actfunc", "tanh"} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNextSent, "dense", { {"width", 2}, {"actfunc", "none"}, {"set", "is_next_sent"} }));

	KERR_CHK(KAI_Network_create(m_hSession, &hNetwork, "serial", { {"name", "bert"}, {"use_output_layer", false} }));

	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "embed", { {"get", "tokens"}, {"vec_size", m_hidden_size}, {"embed_info", embed_info} }));
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "batchnormal", {}));
	KERR_CHK(KAI_Network_append_subnet(m_hSession, hNetwork, hEncoderLoop));
	KERR_CHK(KAI_Network_append_subnet(m_hSession, hNetwork, hNextSent));
	// 2020 버전에 따르면 "use_bias" 옵션을 false로 세팅함, 이유가 뭘까?
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "dense", { {"width", m_voc_count}, {"actfunc", "none"}, {"use_bias", true}, {"set", "all_words"} })); // 'all_words' 실제로는 전혀 이용 안함
	KERR_CHK(KAI_Network_append_named_layer(m_hSession, hNetwork, "select", { {"selector", "mask_index"} }));

	return hNetwork;
}

void BertMission::m_execute() {
	KString sModelName = "Bert " + m_sub_model;

	printf("\n*** %s Model ***\n\n", sModelName.c_str());

	KHNetwork hNetwork = m_buildNetwork();

	KaiDict kwArgs{ { "name", sModelName}, { "dataset", m_hDataset }, { "network", hNetwork },
					{ "loss_exp", m_hLossExp }, { "accuracy_exp", m_hAccExp }, { "visualize_exp", m_hVisExp }, /* {"predict_exp", m_hPredExp},*/
					{ "optimizer", m_hOptimizer }, { "clip_grad", 0.0f } };
	
	if (m_sub_model == "eng_mini") {
		KaiDict debug_trace;
		debug_trace["phase"] = KaiList{ "forward", "backprop" };
		debug_trace["phase"] = KaiList{ "backprop" };
		debug_trace["targets"] = KaiList{ };
		debug_trace["checked"] = KaiList{ "embed", "batchnormal", "self_attention", "extract", "dense", "select", "dropout", "add_serial", "pass" };
		debug_trace["checked"] = KaiList{ "embed", "select", "dense", "extract", "pass", "batchnormal", "dropout", "add_serial", "self_attention" };

		//kwArgs["debug_trace"] = debug_trace;
	}

	KERR_CHK(KAI_Model_create(m_hSession, &m_hModel, "basic", kwArgs));

	m_reporter.ConnectToKai(m_hSession, m_hModel, KCb_mask_all);
	m_reporter.setFeeder(&m_feeder);

	KERR_CHK(KAI_Model_train(m_hModel, { {"epoch_count", 10}, {"epoch_report", 1}, {"epoch_validate", 5}, {"learning_rate", 0.001f} }));
	KERR_CHK(KAI_Model_test(m_hModel, {}));
	KERR_CHK(KAI_Model_visualize(m_hModel, {}));
}
