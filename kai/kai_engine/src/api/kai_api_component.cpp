/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../components/component.h"

KAI_API KRetCode KAI_Component_set_datafeed_callback(KHComponent hComponent, Ken_datafeed_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->set_datafeed_callback(cb_event, pCbInst, pCbFunc, pCbAux);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_set_train_callback(KHComponent hComponent, Ken_train_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->set_train_callback(cb_event, pCbInst, pCbReport, pCbAux);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_set_test_callback(KHComponent hComponent, Ken_test_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->set_test_callback(cb_event, pCbInst, pCbReport, pCbAux);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_set_visualize_callback(KHComponent hComponent, Ken_visualize_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->set_visualize_callback(cb_event, pCbInst, pCbReport, pCbAux);

	NO_SESSION_CLOSE();
}

/*
KAI_API KRetCode KAI_Component_set_predict_callback(KHComponent hComponent, void* pCbInst, void* pCbFuncStart, void* pCbFuncData, void* pCbFuncEnd, void* pCbAux) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->set_predict_callback(pCbInst, pCbFuncStart, pCbFuncData, pCbFuncEnd, pCbAux);

	NO_SESSION_CLOSE();
}
*/

KAI_API KRetCode KAI_download_float_data(KHSession hSession, KInt nToken, KInt nSize, KFloat* pBuffer) {
	SESSION_OPEN();

	pSession->download_float_data(nToken, nSize, pBuffer);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_download_int_data(KHSession hSession, KInt nToken, KInt nSize, KInt* pBuffer) {
	SESSION_OPEN();

	pSession->download_int_data(nToken, nSize, pBuffer);

	SESSION_CLOSE();
}
