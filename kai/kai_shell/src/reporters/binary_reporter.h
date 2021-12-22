/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../reporter.h"
#include "../utils/utils.h"

class BinaryReporter : public Reporter {
public:
	BinaryReporter();
	virtual ~BinaryReporter();

protected:
	virtual KBool m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy);
	virtual KBool m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy);
	virtual KBool m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy);
	virtual KBool m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy);
	virtual KBool m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs);
	/*
	virtual KBool m_trainStart(void* pAux, KString sName, KString sTimestamp, KInt epoch_count, KInt data_count);
	virtual KBool m_epochStart(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt batch_size, KaiDict data_index_token);
	virtual KBool m_batchStart(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index);
	virtual KBool m_validateStart(void* pAux, KString sTimestamp, KInt data_count, KInt batch_size);

	virtual KBool m_layerForwardStart(void* pAux, KaiValue info);
	virtual KBool m_layerForwardEnd(void* pAux, KaiValue info);
	virtual KBool m_layerBackpropStart(void* pAux, KaiValue info);
	virtual KBool m_layerBackpropEnd(void* pAux, KaiValue info);
	virtual KBool m_networkStart(void* pAux, KaiValue info);
	virtual KBool m_networkEnd(void* pAux, KaiValue info);
	virtual KBool m_evalLossStart(void* pAux, KaiValue info);
	virtual KBool m_evalLossEnd(void* pAux, KaiValue info);
	virtual KBool m_evalAccStart(void* pAux, KaiValue info);
	virtual KBool m_evalAccEnd(void* pAux, KaiValue info);
	virtual KBool m_paramUpdateStart(void* pAux, KaiValue info);
	virtual KBool m_paramUpdateEnd(void* pAux, KaiValue info);

	virtual KBool m_testStart(void* pAux, KString sName, KString sTimestamp, KInt dat_count, KaiDict data_index_token);
	virtual KBool m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy);

	virtual KBool m_visualizeStart(void* pAux, KString sName, KString sTimestamp, Ken_visualize_mode mode, KInt dat_count, KaiDict data_index_token);

	virtual KBool m_predictStart(void* pAux, KInt nCount);
	virtual KBool m_predictData(void* pAux, KInt nth);
	virtual KBool m_predictEnd(void* pAux);
	*/

protected:
	int m_version;
	static int ms_checkCode;

protected:
	KString m_get_acc_string(KaiDict accuracy);
};
