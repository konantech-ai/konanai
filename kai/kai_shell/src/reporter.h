/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class Reporter {
public:
	Reporter();
	virtual ~Reporter();

	virtual void ConnectToKai(KHSession hSession, KHComponent hComponent, KInt nCbMask);

	void showDataIndices(KBool bShow = true);

protected:
	KHSession m_hSession;
	KBool m_showDataIndices;

protected:
	static KBool ms_cbTrainStart(void* pInst, void* pAux, KString sName, KString sTimestamp, KInt epoch_count, KInt data_count);
	static KBool ms_cbTrainEnd(void* pInst, void* pAux, KString sName, KString sTimestamp);
	static KBool ms_cbEpochStart(void* pInst, void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt batch_size, KaiDict data_index_token);
	static KBool ms_cbEpochEnd(void* pInst, void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy);
	static KBool ms_cbBatchStart(void* pInst, void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index);
	static KBool ms_cbBatchEnd(void* pInst, void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy);
	static KBool ms_cbValidateStart(void* pInst, void* pAux, KString sTimestamp, KInt data_count, KInt batch_size);
	static KBool ms_cbValidateEnd(void* pInst, void* pAux, KString sTimestamp, KaiDict accuracy);

	static KBool ms_cbLayerForwardStart(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbLayerForwardEnd(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbLayerBackpropStart(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbLayerBackpropEnd(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbNetworkStart(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbNetworkEnd(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbEvalLossStart(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbEvalLossEnd(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbEvalAccStart(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbEvalAccEnd(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbParamUpdateStart(void* pInst, void* pAux, KaiValue info);
	static KBool ms_cbParamUpdateEnd(void* pInst, void* pAux, KaiValue info);

	static KBool ms_cbTestStart(void* pInst, void* pAux, KString sName, KString sTimestamp, KInt dat_count, KaiDict data_index_token);
	static KBool ms_cbTestEnd(void* pInst, void* pAux, KString sName, KString sTimestamp, KaiDict accuracy);

	static KBool ms_cbVisualizeStart(void* pInst, void* pAux, KString sName, KString sTimestamp, Ken_visualize_mode mode, KInt dat_count, KaiDict data_index_token);
	static KBool ms_cbVisualizeEnd(void* pInst, void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vis);

	static KBool ms_cbPredictStart(void* pInst, void* pAux, KInt nCount);
	static KBool ms_cbPredictData(void* pInst, void* pAux, KInt nth);
	static KBool ms_cbPredictEnd(void* pInst, void* pAux);

protected:
	virtual KBool m_trainStart(void* pAux, KString sName, KString sTimestamp, KInt epoch_count, KInt data_count);
	virtual KBool m_trainEnd(void* pAux, KString sName, KString sTimestamp);
	virtual KBool m_epochStart(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt batch_size, KaiDict data_index_token);
	virtual KBool m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy);
	virtual KBool m_batchStart(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index);
	virtual KBool m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy);
	virtual KBool m_validateStart(void* pAux, KString sTimestamp, KInt data_count, KInt batch_size);
	virtual KBool m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy);

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
	virtual KBool m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vis);

	virtual KBool m_predictStart(void* pAux, KInt nCount);
	virtual KBool m_predictData(void* pAux, KInt nth);
	virtual KBool m_predictEnd(void* pAux);

protected:
	void m_dump_arr_token(KString sprefix, KString sTitle, KaiDict token_info);

	KFloat* m_download_float_data(KaiDict dat_info, KaiShape& shape);
	KInt* m_download_int_data(KaiDict dat_info, KaiShape& shape);
};

class FloatBuffer {
public:
	FloatBuffer(KHSession hSession, KaiDict dat_info);
	virtual ~FloatBuffer();

	KInt axis(KInt nth);

	KFloat at(KInt n1);
	KFloat at(KInt n1, KInt n2);
	KFloat at(KInt n1, KInt n2, KInt n3);

protected:
	KaiShape m_shape;
	KFloat* m_pBuffer;
};

class IntBuffer {
public:
	IntBuffer(KHSession hSession, KaiDict dat_info);
	virtual ~IntBuffer();

	KInt axis(KInt nth);

	KInt at(KInt n1);
	KInt at(KInt n1, KInt n2);
	KInt at(KInt n1, KInt n2, KInt n3);

protected:
	KaiShape m_shape;
	KInt* m_pBuffer;
};
