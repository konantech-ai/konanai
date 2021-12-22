/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "reporter.h"
#include "utils/utils.h"

#include <algorithm>

Reporter::Reporter() {
	m_hSession = 0;
	m_showDataIndices = false;
}

Reporter::~Reporter() {
}

void Reporter::showDataIndices(KBool bShow) {
	m_showDataIndices = bShow;
}

void Reporter::ConnectToKai(KHSession hSession, KHComponent hComponent, KInt nCbMask) {
	m_hSession = hSession;

	if (nCbMask & KCb_mask_train_start) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_start, this, reinterpret_cast<void*>(ms_cbTrainStart), NULL));
	if (nCbMask & KCb_mask_train_end) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_end, this, reinterpret_cast<void*>(ms_cbTrainEnd), NULL));
	if (nCbMask & KCb_mask_train_epoch_start) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_epoch_start, this,  reinterpret_cast<void*>(ms_cbEpochStart), NULL));
	if (nCbMask & KCb_mask_train_epoch_end) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_epoch_end, this,  reinterpret_cast<void*>(ms_cbEpochEnd), NULL));
	if (nCbMask & KCb_mask_train_batch_start) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_batch_start, this, reinterpret_cast<void*>(ms_cbBatchStart), NULL));
	if (nCbMask & KCb_mask_train_batch_end) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_batch_end, this,  reinterpret_cast<void*>(ms_cbBatchEnd), NULL));
	if (nCbMask & KCb_mask_train_validate_start) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_validate_start, this, reinterpret_cast<void*>(ms_cbValidateStart), NULL));
	if (nCbMask & KCb_mask_train_validate_end) KERR_CHK(KAI_Component_set_train_callback(hComponent, Ken_train_cb_event::train_validate_end, this,  reinterpret_cast<void*>(ms_cbValidateEnd), NULL));
	if (nCbMask & KCb_mask_test_start) KERR_CHK(KAI_Component_set_test_callback(hComponent, Ken_test_cb_event::test_start, this,  reinterpret_cast<void*>(ms_cbTestStart), NULL));
	if (nCbMask & KCb_mask_test_end) KERR_CHK(KAI_Component_set_test_callback(hComponent, Ken_test_cb_event::test_end, this, reinterpret_cast<void*>(ms_cbTestEnd), NULL));
	if (nCbMask & KCb_mask_visualize_start) KERR_CHK(KAI_Component_set_visualize_callback(hComponent, Ken_visualize_cb_event::visualize_start, this, reinterpret_cast<void*>(ms_cbVisualizeStart), NULL));
	if (nCbMask & KCb_mask_visualize_end) KERR_CHK(KAI_Component_set_visualize_callback(hComponent, Ken_visualize_cb_event::visualize_end, this,  reinterpret_cast<void*>(ms_cbVisualizeEnd), NULL));
}

KBool Reporter::ms_cbTrainStart(void* pInst, void* pAux, KString sName, KString sTimestamp, KInt epoch_count, KInt data_count) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_trainStart(pAux, sName, sTimestamp, epoch_count, data_count);
}

KBool Reporter::ms_cbTrainEnd(void* pInst, void* pAux, KString sName, KString sTimestamp) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_trainEnd(pAux, sName, sTimestamp);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbEpochStart(void* pInst, void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt batch_size, KaiDict data_index_token) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_epochStart(pAux, sTimestamp, epoch_count, epoch_index, batch_size, data_index_token);
}

KBool Reporter::ms_cbEpochEnd(void* pInst, void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_epochEnd(pAux, sTimestamp, epoch_count, epoch_index, dat_count, loss, accuracy);
}

KBool Reporter::ms_cbBatchStart(void* pInst, void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_batchStart(pAux, sTimestamp, batch_count, batch_index);
}

KBool Reporter::ms_cbBatchEnd(void* pInst, void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_batchEnd(pAux, sTimestamp, batch_count, batch_index, batch_size, loss, accuracy);
}

KBool Reporter::ms_cbValidateStart(void* pInst, void* pAux, KString sTimestamp, KInt data_count, KInt batch_size) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_validateStart(pAux, sTimestamp, data_count, batch_size);
}

KBool Reporter::ms_cbValidateEnd(void* pInst, void* pAux, KString sTimestamp, KaiDict accuracy) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_validateEnd(pAux, sTimestamp, accuracy);
}

KBool Reporter::ms_cbLayerForwardStart(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbLayerForwardEnd(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbLayerBackpropStart(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbLayerBackpropEnd(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbNetworkStart(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbNetworkEnd(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbEvalLossStart(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbEvalLossEnd(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbEvalAccStart(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbEvalAccEnd(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbParamUpdateStart(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbParamUpdateEnd(void* pInst, void* pAux, KaiValue info) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbTestStart(void* pInst, void* pAux, KString sName, KString sTimestamp, KInt dat_count, KaiDict data_index_token) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_testStart(pAux, sName, sTimestamp, dat_count, data_index_token);
}

KBool Reporter::ms_cbTestEnd(void* pInst, void* pAux, KString sName, KString sTimestamp, KaiDict accuracy) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_testEnd(pAux, sName, sTimestamp, accuracy);
}

KBool Reporter::ms_cbVisualizeStart(void* pInst, void* pAux, KString sName, KString sTimestamp, Ken_visualize_mode mode, KInt dat_count, KaiDict data_index_token) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_visualizeStart(pAux, sName, sTimestamp, mode, dat_count, data_index_token);
}

KBool Reporter::ms_cbVisualizeEnd(void* pInst, void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vis) {
	Reporter* pInstance = (Reporter*)pInst;
	return pInstance->m_visualizeEnd(pAux, sName, sTimestamp, xs, ys, os, vis);
}

KBool Reporter::ms_cbPredictStart(void* pInst, void* pAux, KInt nCount) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbPredictData(void* pInst, void* pAux, KInt nth) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::ms_cbPredictEnd(void* pInst, void* pAux) {
	Reporter* pInstance = (Reporter*)pInst;
	//return pInstance->m_XXX(pAux, xxx);
	THROW(KERR_UNIMPEMENTED_YET);
	return false;
}

KBool Reporter::m_trainStart(void* pAux, KString sName, KString sTimestamp, KInt epoch_count, KInt data_count) {
	printf("> Model %s train started at %s: %lld epoch, %lld data\n", sName.c_str(), sTimestamp.c_str(), epoch_count, data_count);
	return false;
}

KBool Reporter::m_trainEnd(void* pAux, KString sName, KString sTimestamp) {
	printf("> Model %s train ended at %s\n\n", sName.c_str(), sTimestamp.c_str());
	return false;
}

KBool Reporter::m_epochStart(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt batch_size, KaiDict data_index_token) {
	if (m_showDataIndices) {
		printf("\n   train epoch [%lld/%lld] started at %s: batch_size = %lld", epoch_index, epoch_count, sTimestamp.c_str(), batch_size);
		m_dump_arr_token("\n     ", "data_index", data_index_token);
	}
	return false;
}

KBool Reporter::m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy) {
	KFloat f_loss = (KFloat)loss["#default"]; // / (KFloat)dat_count;
	KFloat f_acc = (KFloat)accuracy["#default"]; // / (KFloat)dat_count;
	printf("   train epoch [%lld/%lld] ended at %s: loss = %5.3f, acc = %5.3f\n", epoch_index, epoch_count, sTimestamp.c_str(), f_loss, f_acc);
	return false;
}

KBool Reporter::m_batchStart(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index) {
	return false;
}

KBool Reporter::m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy) {
	int percent = (int)((batch_index + 1) * 100 / batch_count);
	fprintf(stdout, "     [%4lld/%4lld] %02d%% [", batch_index + 1, batch_count, percent);
	int n = 0;
	for (; n < percent; n += 2) printf("#");
	for (; n < 100; n += 2) printf(" ");
	printf("] loss = %5.3f, acc = %5.3f\r", (KFloat)loss["#default"], (KFloat)accuracy["#default"]);
	if (batch_count == batch_index + 1) printf("%c[2K", 27);
	return false;
}

KBool Reporter::m_validateStart(void* pAux, KString sTimestamp, KInt data_count, KInt batch_size) {
	fprintf(stdout, "   * Validate (data_count:%lld, batch_size:%lld) at %s ", data_count, batch_size, sTimestamp.c_str());
	return false;
}

KBool Reporter::m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy) {
	fprintf(stdout, "accuracy = %5.3f\n", (KFloat)accuracy["#default"]);
	return false;
}

KBool Reporter::m_layerForwardStart(void* pAux, KaiValue info) {
	printf("콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_layerForwardEnd(void* pAux, KaiValue info) {
	printf("콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_layerBackpropStart(void* pAux, KaiValue info) {
	printf("콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_layerBackpropEnd(void* pAux, KaiValue info) {
	printf("콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_networkStart(void* pAux, KaiValue info) {
	printf("콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_networkEnd(void* pAux, KaiValue info) {
	printf("콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_evalLossStart(void* pAux, KaiValue info) {
	printf("m_evalLossStart: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_evalLossEnd(void* pAux, KaiValue info) {
	printf("m_evalLossEnd: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_evalAccStart(void* pAux, KaiValue info) {
	printf("m_evalAccStart: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_evalAccEnd(void* pAux, KaiValue info) {
	printf("m_evalAccEnd: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_paramUpdateStart(void* pAux, KaiValue info) {
	printf("m_paramUpdateStart: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_paramUpdateEnd(void* pAux, KaiValue info) {
	printf("m_paramUpdateEnd: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_testStart(void* pAux, KString sName, KString sTimestamp, KInt dat_count, KaiDict data_index_token) {
	printf("> Model %s test started at %s: %lld data\n", sName.c_str(), sTimestamp.c_str(), dat_count);
	//m_dump_arr_token("", "", data_index_token);
	return false;
}

KBool Reporter::m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy) {
	printf("> Model %s test ended at %s: acc = %5.3f\n\n", sName.c_str(), sTimestamp.c_str(), (KFloat)accuracy["#default"]);
	return false;
}

KBool Reporter::m_visualizeStart(void* pAux, KString sName, KString sTimestamp, Ken_visualize_mode mode, KInt dat_count, KaiDict data_index_token) {
	printf("> Model %s visualize started at %s: %lld data\n", sName.c_str(), sTimestamp.c_str(), dat_count);
	return false;
}

KBool Reporter::m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vis) {
	printf("> m_visualizeEnd: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n\n");
	return false;
}

KBool Reporter::m_predictStart(void* pAux, KInt nCount) {
	printf("m_predictStart: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_predictData(void* pAux, KInt nth) {
	printf("m_predictData: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

KBool Reporter::m_predictEnd(void* pAux) {
	printf("m_predictEnd: 콜백 함수가 처리 루틴 지정 없이 호출되었습니다. 프로그램 로직을 확인하세요.\n");
	return false;
}

void Reporter::m_dump_arr_token(KString sprefix, KString sTitle, KaiDict token_info) {
	KBool bFloat = token_info["is_float"];
	KaiShape shape = token_info["shape"];
	KInt nToken = token_info["token"];

	KString sShape = shape.desc();
	KInt nSize = shape.total_size();

	if (bFloat) {
		KFloat* pBuffer = new KFloat[nSize];
		KERR_CHK(KAI_download_float_data(m_hSession, nToken, nSize, pBuffer));
		printf("%sFloat Array %s%s [", sprefix.c_str(), sTitle.c_str(), sShape.c_str());
		if (nSize > 0) printf("%5.3f", pBuffer[0]);
		if (nSize <= 10) {
			for (KInt n = 1; n < nSize; n++) printf(" %5.3f", pBuffer[n]);
		}
		else {
			for (KInt n = 1; n < 5; n++) printf(" %5.3f", pBuffer[n]);
			printf(" ...");
			for (KInt n = nSize - 5; n < nSize; n++) printf(" %5.3f", pBuffer[n]);
		}
		printf("] %lld elements\n", nSize);
		delete[] pBuffer;
	}
	else {
		KInt* pBuffer = new KInt[nSize];
		KERR_CHK(KAI_download_int_data(m_hSession, nToken, nSize, pBuffer));
		printf("%sInt Array %s%s [", sprefix.c_str(), sTitle.c_str(), sShape.c_str());
		if (nSize > 0) printf("%lld", pBuffer[0]);
		if (nSize <= 10) {
			for (KInt n = 1; n < nSize; n++) printf(" %lld", pBuffer[n]);
		}
		else {
			for (KInt n = 1; n < 5; n++) printf(" %lld", pBuffer[n]);
			printf(" ...");
			for (KInt n = nSize - 5; n < nSize; n++) printf(" %lld", pBuffer[n]);
		}
		printf("] %lld elements\n", nSize);
		delete[] pBuffer;
	}
}

KFloat* Reporter::m_download_float_data(KaiDict dat_info, KaiShape& shape) {
	KBool bFloat = dat_info["is_float"];
	
	if (!bFloat) THROW(KERR_REQUEST_FLOAT_DATA_FOR_NON_FLOAT_FIELD);

	shape = dat_info["shape"];

	KInt nSize = shape.total_size();
	KInt nToken = dat_info["token"];

	KFloat* pBuffer = new KFloat[nSize];

	KERR_CHK(KAI_download_float_data(m_hSession, nToken, nSize, pBuffer));

	return pBuffer;
}

//====================================================================================================

FloatBuffer::FloatBuffer(KHSession hSession, KaiDict dat_info) : m_pBuffer(0) {
	KBool bFloat = dat_info["is_float"];

	if (!bFloat) THROW(KERR_REQUEST_FLOAT_DATA_FOR_NON_FLOAT_FIELD);

	m_shape = dat_info["shape"];

	KInt nSize = m_shape.total_size();
	KInt nToken = dat_info["token"];

	m_pBuffer = new KFloat[nSize];

	KERR_CHK(KAI_download_float_data(hSession, nToken, nSize, m_pBuffer));
}

FloatBuffer::~FloatBuffer() {
	delete m_pBuffer;
}

KInt FloatBuffer::axis(KInt nth) {
	return m_shape[nth];
}

KFloat FloatBuffer::at(KInt n1) {
	if (m_shape.size() != 1) THROW(KERR_REQUEST_ARRAY_DATA_WITH_BAD_NUM_IDXS);
	return m_pBuffer[n1];
}

KFloat FloatBuffer::at(KInt n1, KInt n2) {
	if (m_shape.size() != 2) THROW(KERR_REQUEST_ARRAY_DATA_WITH_BAD_NUM_IDXS);
	KInt nx = n1 * m_shape[1] + n2;
	return m_pBuffer[nx];
}

KFloat FloatBuffer::at(KInt n1, KInt n2, KInt n3) {
	if (m_shape.size() != 3) THROW(KERR_REQUEST_ARRAY_DATA_WITH_BAD_NUM_IDXS);
	KInt nx = (n1 * m_shape[1] + n2) * m_shape[2] + n3;
	return m_pBuffer[nx];
}

//====================================================================================================

IntBuffer::IntBuffer(KHSession hSession, KaiDict dat_info) : m_pBuffer(0) {
	KBool bFloat = dat_info["is_float"];

	if (bFloat) THROW(KERR_REQUEST_NON_FLOAT_DATA_FOR_FLOAT_FIELD);

	m_shape = dat_info["shape"];

	KInt nSize = m_shape.total_size();
	KInt nToken = dat_info["token"];

	m_pBuffer = new KInt[nSize];

	KERR_CHK(KAI_download_int_data(hSession, nToken, nSize, m_pBuffer));
}

IntBuffer::~IntBuffer() {
	delete m_pBuffer;
}

KInt IntBuffer::axis(KInt nth) {
	return m_shape[nth];
}

KInt IntBuffer::at(KInt n1) {
	if (m_shape.size() != 1) THROW(KERR_REQUEST_ARRAY_DATA_WITH_BAD_NUM_IDXS);
	return m_pBuffer[n1];
}

KInt IntBuffer::at(KInt n1, KInt n2) {
	if (m_shape.size() != 2) THROW(KERR_REQUEST_ARRAY_DATA_WITH_BAD_NUM_IDXS);
	KInt nx = n1 * m_shape[1] + n2;
	return m_pBuffer[nx];
}

KInt IntBuffer::at(KInt n1, KInt n2, KInt n3) {
	if (m_shape.size() != 3) THROW(KERR_REQUEST_ARRAY_DATA_WITH_BAD_NUM_IDXS);
	KInt nx = (n1 * m_shape[1] + n2) * m_shape[2] + n3;
	return m_pBuffer[nx];
}
