/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../reporter.h"
#include "../data_feeders/bert_feeder.h"
#include "../utils/utils.h"

class BertReporter : public Reporter {
public:
	BertReporter();
	virtual ~BertReporter();

	void setModel(KString sub_model) { m_sub_model = sub_model; }
	void setFeeder(BertFeeder* pFeeder) { m_pFeeder = pFeeder; }

protected:
	virtual KBool m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy);
	virtual KBool m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy);
	virtual KBool m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy);
	virtual KBool m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy);
	virtual KBool m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vis);

protected:
	int m_version;
	static int ms_checkCode;

	KString m_sub_model;

protected:
	BertFeeder* m_pFeeder;
};
