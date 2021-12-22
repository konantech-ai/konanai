/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../mission.h"
#include "../data_feeders/bert_feeder.h"
#include "../reporters/bert_reporter.h"
#include "../utils/utils.h"

class BertMission : public Mission {
public:
	BertMission(KHSession hSession, KString sub_model, enum Ken_test_level testLevel);
	virtual ~BertMission();

	virtual void Execute();

protected:
	void m_createComponents();
	void m_execute();
	KHNetwork m_buildNetwork();

protected:
	KString m_sub_model;

	BertFeeder m_feeder;
	BertReporter m_reporter;

protected:
	KInt m_hidden_size;
	KInt m_stack_depth;
	KInt m_attention_heads;
	KInt m_max_position;

	KInt m_hidden_ex_size;

	KInt m_voc_count;

	KFloat m_att_dropout_ratio;
	KFloat m_hid_dropout_ratio;

	KInt m_batch_size;
};
