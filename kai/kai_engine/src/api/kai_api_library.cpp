/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "kai_api_common.h"

#include "../session/session.h"

#include "../library/library.h"
#include "../library/local_library.h"
#include "../library/public_library.h"
#include "../library/object.h"

#include "../components/kmodel.h"
#include "../components/kdataset.h"
#include "../components/knetwork.h"
#include "../components/kexpression.h"
#include "../components/koptimizer.h"

#include "../utils/kutil.h"
#include "../math/kmath.h"

KAI_API KRetCode KAI_OpenSession(KHSession* phSession) {
	try {
		POINTER_CHECK(phSession);

		KaiSession* pSession = new KaiSession();
		*phSession = (KHSession)pSession;
		return KRetOK;
	}
	catch (...) {
		return KERR_SESSION_CREATE_FAILURE;
	}
}

KAI_API void KAI_GetLastErrorCode(KHSession hSession, KRetCode* pRetCode) {
	KaiSession* pSession = NULL;

	if (pRetCode == NULL) return;

	try {
		pSession = KaiSession::HandleToPointer(hSession);
	}
	catch (...) {
		*pRetCode = KERR_INVALID_SESSION_HANDLE;
		return;
	}

	try {
		*pRetCode = pSession->GetLastErrorCode();
	}
	catch (KaiException ex) {
		*pRetCode = ex.GetErrorCode();
	}
	catch (...) {
		*pRetCode = KERR_UNKNOWN_ERROR;
	}
}

KAI_API void KAI_GetLastErrorMessage(KHSession hSession, KString* psErrMessage) {
	KaiSession* pSession = NULL;

	if (psErrMessage == NULL) return;

	try {
		pSession = KaiSession::HandleToPointer(hSession);
	}
	catch (...) {
		*psErrMessage = KaiException::GetErrorMessage(KERR_INVALID_SESSION_HANDLE);
		return;
	}

	try {
		*psErrMessage = pSession->GetLastErrorMessage();
	}
	catch (KaiException ex) {
		*psErrMessage = ex.GetErrorMessage();
	}
	catch (...) {
		*psErrMessage = KaiException::GetErrorMessage(KERR_UNKNOWN_ERROR);
	}
}

/*
KAI_API KRetCode KAI_CloseSession(KHSession hSession) {
	KalSession* pSession = NULL;
	try {
		pSession = KalSession::HandleToPointer(hSession);
	}
	catch (...) {
		return KERR_INVALID_SESSION_HANDLE;
	}
	try {
		;
		// your code according to the (pSession) here
		return KRetOK;
	}
	catch (KaiException ex) {
		pSession->SetLastError(ex); return ex.GetErrorCode();
	}
	catch (...) {
		return KERR_UNKNOWN_ERROR;
	};
}
*/

KAI_API KRetCode KAI_CloseSession(KHSession hSession) {
	SESSION_OPEN();

	delete pSession;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_debug_session_component_dump(KHSession hSession) {
	SESSION_OPEN();

	pSession->debug_component_dump();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_debug_curr_count(KHSession hSession, KInt* pnObjCnt, KInt* pnValCnt) {
	SESSION_OPEN();

	if (pnObjCnt) *pnObjCnt = KaiObject::ms_debug_count;
	if (pnValCnt) *pnValCnt = KaiValueCore::ms_debug_count;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetVersion(KHSession hSession, KString* psVersion) {
	SESSION_OPEN();
	POINTER_CHECK(psVersion);

	KString sVersion = pSession->GetVersion();
	*psVersion = sVersion;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetLocalLibaryNames(KHSession hSession, KStrList* pslLibNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslLibNames);

	pSession->GetLocalLibaryNames(pslLibNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_DeleteLocalLibary(KHSession hSession, KString sLibName) {
	SESSION_OPEN();

	pSession->DeleteLocalLibary(sLibName);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_DeleteAllLocalLibraries(KHSession hSession) {
	SESSION_OPEN();

	KStrList slLibNames;
	pSession->GetLocalLibaryNames(&slLibNames);

	for (auto it = slLibNames.begin(); it != slLibNames.end(); it++) {
		pSession->DeleteLocalLibary(*it);
	}

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetInstallModels(KHSession hSession, KStrList* pslModuleNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslModuleNames);

	pSession->GetInstallModels(pslModuleNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_CreateLocalLibrary(KHSession hSession, KHLibrary* phLib, KString sLibName, KString sPassword, Ken_inst_mode enumInstallMode, KStrList slModelNames) {
	SESSION_OPEN();
	POINTER_CHECK(phLib);

	KaiLocalLibrary* pLibrary = pSession->CreateLocalLibrary(sLibName, sPassword, enumInstallMode, slModelNames);
	*phLib = (KHLibrary) pLibrary;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_OpenLocalLibrary(KHSession hSession, KHLibrary* phLib, KString sLibName, KString sPassword) {
	SESSION_OPEN();
	POINTER_CHECK(phLib);

	pSession->OpenLocalLibrary(phLib, sLibName, sPassword);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_ConnectToPublicLibrary(KHSession hSession, KHLibrary* phLib, KString sIpAddr, KString sPort, KString sLibName, KString sUserName, KString sPassword) {
	SESSION_OPEN();
	POINTER_CHECK(phLib);

	KaiPublicLibrary* pLibrary = pSession->ConnectToPublicLibrary(sIpAddr, sPort, sLibName, sUserName, sPassword);
	*phLib = (KHLibrary)pLibrary;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetModelProperties(KHSession hSession, KStrList* pslProps) {
	SESSION_OPEN();

	pSession->GetProperties(Ken_component_type::model, pslProps);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetDatasetProperties(KHSession hSession, KStrList* pslProps) {
	SESSION_OPEN();

	pSession->GetProperties(Ken_component_type::dataset, pslProps);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetDataloaderProperties(KHSession hSession, KStrList* pslProps) {
	SESSION_OPEN();

	pSession->GetProperties(Ken_component_type::dataloader, pslProps);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetNetworkProperties(KHSession hSession, KStrList* pslProps) {
	SESSION_OPEN();

	pSession->GetProperties(Ken_component_type::network, pslProps);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetExpressionProperties(KHSession hSession, KStrList* pslProps) {
	SESSION_OPEN();

	pSession->GetProperties(Ken_component_type::expression, pslProps);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_GetOptimizerProperties(KHSession hSession, KStrList* pslProps) {
	SESSION_OPEN();

	pSession->GetProperties(Ken_component_type::optimizer, pslProps);

	SESSION_CLOSE();
}

// 참고: 아래의 KAI_LocalLib_SetName() 함수 정의에서 매크로를 펼친 형태임
/*
KAI_API KRetCode KAI_LocalLib_SetName(KHSession hSession, KHLibrary hLib, KString sNewLibName) {
	KalSession* pSession = NULL;
	try {
		pSession = KalSession::HandleToPointer(hSession);
	}
	catch (...) {
		return KERR_INVALID_SESSION_HANDLE;
	}
	try {
		;
		KaiLocalLibrary* pLib = KaiLocalLibrary::HandleToPointer(hLib);
		// your code according to the (pSession, pLib) here
		return KRetOK;
	}
	catch (KaiException ex) {
		pSession->SetLastError(ex); return ex.GetErrorCode();
	}
	catch (...) {
		return KERR_UNKNOWN_ERROR;
	};
}
*/

KAI_API KRetCode KAI_LocalLib_setName(KHSession hSession, KHLibrary hLib, KString sNewLibName) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLocalLibrary, hLib, pLib);

	pLib->setName(sNewLibName);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_LocalLib_installModels(KHSession hSession, KHLibrary hLib, KStrList slModels) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLocalLibrary, hLib, pLib);

	pLib->installModels(slModels);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_LocalLib_save(KHSession hSession, KHLibrary hLib) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLocalLibrary, hLib, pLib);

	pLib->save();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_LocalLib_destory(KHSession hSession, KHLibrary hLib) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLocalLibrary, hLib, pLib);

	pLib->destory();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_LocalLib_close(KHSession hSession, KHLibrary hLib, bool bSave) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLocalLibrary, hLib, pLib);

	pLib->close(bSave);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_PubLib_login(KHSession hSession, KHLibrary hLib, KString sUserName, KString sPassword) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiPublicLibrary, hLib, pLib);

	pLib->login(sUserName, sPassword);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_PubLib_logout(KHSession hSession, KHLibrary hLib) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiPublicLibrary, hLib, pLib);

	pLib->logout();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_PubLib_close(KHSession hSession, KHLibrary hLib) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiPublicLibrary, hLib, pLib);

	pLib->close();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_getVersion(KHSession hSession, KHLibrary hLib, KString* psVersion) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(psVersion);

	KString sVersion = pLib->GetVersion();
	*psVersion = sVersion;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_getName(KHSession hSession, KHLibrary hLib, KString * psLibName) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(psLibName);

	KString sLibName = pLib->getName();
	*psLibName = sLibName;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_changePassword(KHSession hSession, KHLibrary hLib, KString sOldPassword, KString sNewPassword) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->changePassword(sOldPassword, sNewPassword);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_getCurrPath(KHSession hSession, KHLibrary hLib, KPathString* psCurrPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(psCurrPath);

	KPathString sCurrPath = pLib->getCurrPath();
	*psCurrPath = sCurrPath;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_setCurrPath(KHSession hSession, KHLibrary hLib, KPathString sCurrPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->setCurrPath(sCurrPath);
	
	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_createFolder(KHSession hSession, KHLibrary hLib, KPathString sNewPath, bool bThrowOnExist) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->createFolder(sNewPath, bThrowOnExist);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_renameFolder(KHSession hSession, KHLibrary hLib, KPathString sFolderPath, KString sNewName) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->renameFolder(sFolderPath, sNewName);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_moveFolder(KHSession hSession, KHLibrary hLib, KPathString sOldPath, KPathString sNewPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->moveFolder(sOldPath, sNewPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_deleteFolder(KHSession hSession, KHLibrary hLib, KPathString sFolderPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->deleteFolder(sFolderPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_listFolders(KHSession hSession, KHLibrary hLib, KPathStrList* pslSubFolders, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pslSubFolders);

	pLib->listSubFolders(pslSubFolders, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_list(KHSession hSession, KHLibrary hLib, KJsonStrList* pjlComponents, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pjlComponents);

	pLib->list(pjlComponents, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_listModels(KHSession hSession, KHLibrary hLib, KJsonStrList* pslModelInfo, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pslModelInfo);

	pLib->listComponents(Ken_component_type::model, pslModelInfo, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_listDatasets(KHSession hSession, KHLibrary hLib, KJsonStrList* pslDatasetInfo, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pslDatasetInfo);

	pLib->listComponents(Ken_component_type::dataset, pslDatasetInfo, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_listDataloaders(KHSession hSession, KHLibrary hLib, KJsonStrList* pslDataloaderInfo, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pslDataloaderInfo);

	pLib->listComponents(Ken_component_type::dataloader, pslDataloaderInfo, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_listNetworks(KHSession hSession, KHLibrary hLib, KJsonStrList* pslNetworkInfo, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pslNetworkInfo);

	pLib->listComponents(Ken_component_type::network, pslNetworkInfo, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_listExpressions(KHSession hSession, KHLibrary hLib, KJsonStrList* pslExpressionInfo, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pslExpressionInfo);

	pLib->listComponents(Ken_component_type::expression, pslExpressionInfo, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_listOptimizers(KHSession hSession, KHLibrary hLib, KJsonStrList* pslOptimizerInfo, KPathString sPath, bool recursive) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(pslOptimizerInfo);

	pLib->listComponents(Ken_component_type::optimizer, pslOptimizerInfo, sPath, recursive);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_set(KHSession hSession, KHLibrary hLib, KPathString sComponentlPath, KString sProperty, KString sValue) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->setProperty(sComponentlPath, sProperty, sValue);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_move(KHSession hSession, KHLibrary hLib, KPathString sComponentlPath, KPathString sDestFolder) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->moveComponent(sComponentlPath, sDestFolder);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_rename(KHSession hSession, KHLibrary hLib, KPathString sModelPath, KString sNewName) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->renameComponent(sModelPath, sNewName);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Lib_delete(KHSession hSession, KHLibrary hLib, KPathString sModelPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);

	pLib->deleteComponent(sModelPath);

	SESSION_CLOSE();
}

//----------------------------------------------------------------------------------------------------------------

KAI_API KRetCode KAI_Component_bind(KHSession hSession, KHComponent hComponent1, KHComponent hComponent2, KString sRelation) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiComponent, hComponent1, pComponent1);
	HANDLE_OPEN(KaiComponent, hComponent2, pComponent2);

	pComponent1->bind(pComponent2, sRelation, true, true);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_regist(KHSession hSession, KHLibrary hLib, KHComponent hComponent, KPathString sNewComponentPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->regist(pLib, sNewComponentPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_touch(KHSession hSession, KHComponent hComponent) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->touch();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_set(KHSession hSession, KHComponent hComponent, KString sProperty, KString sValue) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->set_property(sProperty, sValue);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_update(KHSession hSession, KHComponent hComponent) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->update();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_close(KHSession hSession, KHComponent hComponent) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->close();

	SESSION_CLOSE();
}


KAI_API KRetCode KAI_Component_get_property(KHComponent hComponent, KString sKey, KaiValue* pvValue) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);
	POINTER_CHECK(pvValue);

	*pvValue = pComponent->get_property(sKey);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_get_str_property(KHComponent hComponent, KString sKey, KString* psValue) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);
	POINTER_CHECK(psValue);

	KaiValue value = pComponent->get_property(sKey);
	*psValue = value.desc();

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_get_int_property(KHComponent hComponent, KString sKey, KInt * pnValue) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);
	POINTER_CHECK(pnValue);

	*pnValue = pComponent->get_property(sKey);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Object_get_type(KHObject hObject, Ken_object_type* pObj_type) {
	NO_SESSION_OPEN();
	POINTER_CHECK(pObj_type);

	*pObj_type = ((KaiObject*) hObject)->get_type();

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_set_property(KHSession hSession, KHComponent hComponent, KaiDict kwArgs) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->set_property_dict(kwArgs);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Component_dump_property(KHComponent hComponent, KString sKey, KString sTitle) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->dump_property(sKey, sTitle);

	NO_SESSION_CLOSE();
}


KAI_API KRetCode KAI_Component_dump_binding_blocks(KHComponent hComponent) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiComponent, hComponent, pComponent);

	pComponent->get_session()->dump_binding_blocks(pComponent);

	NO_SESSION_CLOSE();
}

//----------------------------------------------------------------------------------------------------------------

KAI_API KRetCode KAI_Model_download(KHSession hSession, KHLibrary hLib, KHModel * phModel, KPathString sModelPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(phModel);

	KaiComponentInfo* pComponentInfo = pLib->seekComponent(sModelPath);
	KaiModel* pModel = new KaiModel(pSession, pLib, pComponentInfo);
	*phModel = (KHModel)pModel;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_regist(KHSession hSession, KHLibrary hLib, KHModel hModel, KPathString sNewModelPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	HANDLE_OPEN(KaiModel, hModel, pModel);

	pModel->regist(pLib, sNewModelPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_touch(KHSession hSession, KHModel hModel) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiModel, hModel, pModel);

	pModel->touch();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_set(KHSession hSession, KHModel hModel, KString sProperty, KString sValue) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiModel, hModel, pModel);

	pModel->set_property(sProperty, sValue);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_update(KHSession hSession, KHModel hModel) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiModel, hModel, pModel);

	pModel->update();

	SESSION_CLOSE();
}

//----------------------------------------------------------------------------------------------------------------

KAI_API KRetCode KAI_Dataset_download(KHSession hSession, KHLibrary hLib, KHDataset* phDataset, KPathString sDatasetPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(phDataset);

	KaiComponentInfo* pComponentInfo = pLib->seekComponent(sDatasetPath);
	KaiDataset* pDataset = new KaiDataset(pSession, pLib, pComponentInfo);
	*phDataset = (KHDataset)pDataset;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Dataset_regist(KHSession hSession, KHLibrary hLib, KHDataset hDataset, KPathString sNewDatasetPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	HANDLE_OPEN(KaiDataset, hDataset, pDataset);

	pDataset->regist(pLib, sNewDatasetPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Dataset_touch(KHSession hSession, KHDataset hDataset) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiDataset, hDataset, pDataset);

	pDataset->touch();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Dataset_set(KHSession hSession, KHDataset hDataset, KString sProperty, KString sValue) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiDataset, hDataset, pDataset);

	pDataset->set_property(sProperty, sValue);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Dataset_update(KHSession hSession, KHDataset hDataset) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiDataset, hDataset, pDataset);

	pDataset->update();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Dataset_close(KHSession hSession, KHDataset hDataset) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(KaiDataset, hDataset, pDataset);

	pDataset->close();

	SESSION_CLOSE();
}

//----------------------------------------------------------------------------------------------------------------

KAI_API KRetCode KAI_Network_download(KHSession hSession, KHLibrary hLib, KHNetwork* phNetwork, KPathString sNetworkPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(phNetwork);

	KaiComponentInfo* pComponentInfo = pLib->seekComponent(sNetworkPath);
	KaiNetwork* pNetwork = new KaiNetwork(pSession, pLib, pComponentInfo);
	*phNetwork = (KHNetwork)pNetwork;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_regist(KHSession hSession, KHLibrary hLib, KHNetwork hNetwork, KPathString sNewNetworkPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);

	pNetwork->regist(pLib, sNewNetworkPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_touch(KHSession hSession, KHNetwork hNetwork) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);

	pNetwork->touch();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_set(KHSession hSession, KHNetwork hNetwork, KString sProperty, KString sValue) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);

	pNetwork->set_property(sProperty, sValue);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_update(KHSession hSession, KHNetwork hNetwork) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);

	pNetwork->update();

	SESSION_CLOSE();
}

//----------------------------------------------------------------------------------------------------------------

KAI_API KRetCode KAI_Expression_download(KHSession hSession, KHLibrary hLib, KHExpression* phExpression, KPathString sExpressionPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(phExpression);

	KaiComponentInfo* pComponentInfo = pLib->seekComponent(sExpressionPath);
	KaiExpression* pExpression = new KaiExpression(pSession, pLib, pComponentInfo);
	*phExpression = (KHExpression)pExpression;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Expression_regist(KHSession hSession, KHLibrary hLib, KHExpression hExpression, KPathString sNewExpressionPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	HANDLE_OPEN(KaiExpression, hExpression, pExpression);

	pExpression->regist(pLib, sNewExpressionPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Expression_touch(KHSession hSession, KHExpression hExpression) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiExpression, hExpression, pExpression);

	pExpression->touch();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Expression_set(KHSession hSession, KHExpression hExpression, KString sProperty, KString sValue) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiExpression, hExpression, pExpression);

	pExpression->set_property(sProperty, sValue);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Expression_update(KHSession hSession, KHExpression hExpression) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiExpression, hExpression, pExpression);

	pExpression->update();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Expression_close(KHSession hSession, KHExpression hExpression) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(KaiExpression, hExpression, pExpression);

	pExpression->close();

	SESSION_CLOSE();
}

//----------------------------------------------------------------------------------------------------------------

KAI_API KRetCode KAI_Optimizer_download(KHSession hSession, KHLibrary hLib, KHOptimizer* phOptimizer, KPathString sOptimizerPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	POINTER_CHECK(phOptimizer);

	KaiComponentInfo* pComponentInfo = pLib->seekComponent(sOptimizerPath);
	KaiOptimizer* pOptimizer = new KaiOptimizer(pSession, pLib, pComponentInfo);
	*phOptimizer = (KHOptimizer)pOptimizer;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Optimizer_regist(KHSession hSession, KHLibrary hLib, KHOptimizer hOptimizer, KPathString sNewOptimizerPath) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLibrary, hLib, pLib);
	HANDLE_OPEN(KaiOptimizer, hOptimizer, pOptimizer);

	pOptimizer->regist(pLib, sNewOptimizerPath);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Optimizer_touch(KHSession hSession, KHOptimizer hOptimizer) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiOptimizer, hOptimizer, pOptimizer);

	pOptimizer->touch();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Optimizer_set(KHSession hSession, KHOptimizer hOptimizer, KString sProperty, KString sValue) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiOptimizer, hOptimizer, pOptimizer);

	pOptimizer->set_property(sProperty, sValue);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Optimizer_update(KHSession hSession, KHOptimizer hOptimizer) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiOptimizer, hOptimizer, pOptimizer);

	pOptimizer->update();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Optimizer_close(KHSession hSession, KHOptimizer hOptimizer) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(KaiOptimizer, hOptimizer, pOptimizer);

	pOptimizer->close();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_value_dump(KaiValue vData, KString sTitle) {
	NO_SESSION_OPEN();

	KString desc = vData.desc();
	logger.Print("%s %s", sTitle.c_str(), desc.c_str());

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Util_fft(KHSession hSession, KFloat* pWave, KFloat* pFTT, KInt file_count, KInt fetch_width, KInt step_width, KInt step_cnt, KInt fft_width, KInt freq_cnt) {
	SESSION_OPEN();

	KaiMath* pMath = KaiMath::Allocate(NULL);
	pMath->fft(pWave, pFTT, file_count, fetch_width, step_width, step_cnt, fft_width, freq_cnt);

	SESSION_CLOSE();
}
