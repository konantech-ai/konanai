/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

KAI_API KRetCode KAI_OpenCudaOldVersion(KHSession hSession) {
	SESSION_OPEN();

	pSession->OpenCudaOldVersion();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetMissionNames(KHSession hSession, KStrList * pslMissionNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslMissionNames);

	pSession->GetMissionNames(pslMissionNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetPluginTypeNames(KHSession hSession, KStrList * pslPluginTypeNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslPluginTypeNames);

	pSession->GetPluginTypeNames(pslPluginTypeNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetPluginNomNames(KHSession hSession, KString sPluginTypeName, KStrList * pslPluginNomNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslPluginNomNames);

	pSession->GetPluginNomNames(sPluginTypeName, pslPluginNomNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_SetCudaOption(KHSession hSession, KStrList slTokens) {
	SESSION_OPEN();

	pSession->SetCudaOption(slTokens);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetCudaOption(KHSession hSession, int* pnDeviceCnt, int* pnDeviceNum, int* pnBlockSize, KInt* pnAvailMem, KInt* pnUsingMem) {
	SESSION_OPEN();

	pSession->GetCudaOption(pnDeviceCnt, pnDeviceNum, pnBlockSize, pnAvailMem, pnUsingMem);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_SetImageOption(KHSession hSession, KStrList slTokens) {
	SESSION_OPEN();

	pSession->SetImageOption(slTokens);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetImageOption(KHSession hSession, KBool* pbImgScreen, KString* psImgFolder) {
	SESSION_OPEN();

	pSession->GetImageOption(pbImgScreen, psImgFolder);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_SetSelectOption(KHSession hSession, KStrList slTokens) {
	SESSION_OPEN();

	pSession->SetSelectOption(slTokens);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetSelectDesc(KHSession hSession, KString sTypeName, KString* psTypeDesc) {
	SESSION_OPEN();
	POINTER_CHECK(psTypeDesc);

	*psTypeDesc = pSession->GetSelectDesc(sTypeName);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_ExecMission(KHSession hSession, KString sMissionName, KString sExecMode) {
	SESSION_OPEN();

	pSession->ExecMission(sMissionName, sExecMode);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Session_set_callback(KHSession hSession, Ken_session_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux) {
	SESSION_OPEN();

	pSession->RegistCallback(cb_event, pCbInst, pCbFunc, pCbAux);

	SESSION_CLOSE();
}
