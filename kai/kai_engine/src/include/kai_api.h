/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#ifdef KAI2021_WINDOWS
#ifdef KAI_EXPORTS
#ifdef KAI_API
#undef KAI_API
#endif
#define KAI_API __declspec(dllexport)
#else
#define KAI_API __declspec(dllimport)
#endif
#else
#ifdef KAI_API
#undef KAI_API
#endif
#define KAI_API __attribute__((__visibility__("default")))
#endif

#include "kai_errors.h"
#include "kai_types.h"
#include "kai_value.hpp"
#include "kai_callback.h"

extern "C" KAI_API KRetCode KAI_OpenSession(KHSession * phSession);
extern "C" KAI_API KRetCode KAI_CloseSession(KHSession hSession);

extern "C" KAI_API void KAI_GetLastErrorCode(KHSession hSession, KRetCode* pRetCode);
extern "C" KAI_API void KAI_GetLastErrorMessage(KHSession hSession, KString * psErrMessage);

extern "C" KAI_API KRetCode KAI_GetVersion(KHSession hSession, KString * psVersion);
extern "C" KAI_API KRetCode KAI_GetLocalLibaryNames(KHSession hSession, KStrList * pslLibNames);
extern "C" KAI_API KRetCode KAI_DeleteLocalLibary(KHSession hSession, KString sLibName);
extern "C" KAI_API KRetCode KAI_GetInstallModels(KHSession hSession, KStrList * pslModuleNames);

extern "C" KAI_API KRetCode KAI_CreateLocalLibrary(KHSession hSession, KHLibrary * phLib, KString sLibName, KString sPassword, Ken_inst_mode enumInstallMode, KStrList slModelNames= KStrList());
extern "C" KAI_API KRetCode KAI_OpenLocalLibrary(KHSession hSession, KHLibrary * phLib, KString sLibName, KString sPassword);
extern "C" KAI_API KRetCode KAI_ConnectToPublicLibrary(KHSession hSession, KHLibrary * phLib, KString sIpAddr, KString sPort, KString sLibName, KString sUserName, KString sPassword);
extern "C" KAI_API KRetCode KAI_DeleteAllLocalLibraries(KHSession hSession);

extern "C" KAI_API KRetCode KAI_GetModelProperties(KHSession hSession, KStrList* pslProps);
extern "C" KAI_API KRetCode KAI_GetDatasetProperties(KHSession hSession, KStrList* pslProps);
extern "C" KAI_API KRetCode KAI_GetDataloaderProperties(KHSession hSession, KStrList* pslProps);
extern "C" KAI_API KRetCode KAI_GetNetworkProperties(KHSession hSession, KStrList* pslProps);
extern "C" KAI_API KRetCode KAI_GetExpressionProperties(KHSession hSession, KStrList* pslProps);
extern "C" KAI_API KRetCode KAI_GetOptimizerProperties(KHSession hSession, KStrList* pslProps);

extern "C" KAI_API KRetCode KAI_LocalLib_setName(KHSession hSession, KHLibrary hLib, KString sNewLibName);
extern "C" KAI_API KRetCode KAI_LocalLib_installModels(KHSession hSession, KHLibrary hLib, KStrList slModels);
extern "C" KAI_API KRetCode KAI_LocalLib_save(KHSession hSession, KHLibrary hLib);
extern "C" KAI_API KRetCode KAI_LocalLib_destory(KHSession hSession, KHLibrary hLib);
extern "C" KAI_API KRetCode KAI_LocalLib_close(KHSession hSession, KHLibrary hLib, bool bSave);

extern "C" KAI_API KRetCode KAI_PubLib_login(KHSession hSession, KHLibrary hLib, KString sUserName, KString sPassword);
extern "C" KAI_API KRetCode KAI_PubLib_logout(KHSession hSession, KHLibrary hLib);
extern "C" KAI_API KRetCode KAI_PubLib_close(KHSession hSession, KHLibrary hLib);

extern "C" KAI_API KRetCode KAI_Lib_getVersion(KHSession hSession, KHLibrary hLib, KString* psVersion);
extern "C" KAI_API KRetCode KAI_Lib_getName(KHSession hSession, KHLibrary hLib, KString * psLibName);
extern "C" KAI_API KRetCode KAI_Lib_changePassword(KHSession hSession, KHLibrary hLib, KString sOldPassword, KString sNewPassword);

extern "C" KAI_API KRetCode KAI_Lib_getCurrPath(KHSession hSession, KHLibrary hLib, KPathString * psCurrPath);
extern "C" KAI_API KRetCode KAI_Lib_setCurrPath(KHSession hSession, KHLibrary hLib, KPathString sCurrPath);
extern "C" KAI_API KRetCode KAI_Lib_createFolder(KHSession hSession, KHLibrary hLib, KPathString sNewPath, bool bThrowOnExist);
extern "C" KAI_API KRetCode KAI_Lib_renameFolder(KHSession hSession, KHLibrary hLib, KPathString sFolderPath, KString sNewName);
extern "C" KAI_API KRetCode KAI_Lib_moveFolder(KHSession hSession, KHLibrary hLib, KPathString sFolderPath, KPathString sDestPath);
extern "C" KAI_API KRetCode KAI_Lib_deleteFolder(KHSession hSession, KHLibrary hLib, KPathString sFolderPath);
extern "C" KAI_API KRetCode KAI_Lib_listFolders(KHSession hSession, KHLibrary hLib, KPathStrList * pslSubFolders, KPathString sPath, bool recursive);

extern "C" KAI_API KRetCode KAI_Lib_list(KHSession hSession, KHLibrary hLib, KJsonStrList * pjlComponents, KPathString sPath, bool recursive);

extern "C" KAI_API KRetCode KAI_Lib_listModels(KHSession hSession, KHLibrary hLib, KJsonStrList * pslModelInfo, KPathString sPath, bool recursive);
extern "C" KAI_API KRetCode KAI_Lib_listDatasets(KHSession hSession, KHLibrary hLib, KJsonStrList * pslDatasetInfo, KPathString sPath, bool recursive);
extern "C" KAI_API KRetCode KAI_Lib_listDataloaders(KHSession hSession, KHLibrary hLib, KJsonStrList * pslDataloaderInfo, KPathString sPath, bool recursive);
extern "C" KAI_API KRetCode KAI_Lib_listNetworks(KHSession hSession, KHLibrary hLib, KJsonStrList * pslNetworkInfo, KPathString sPath, bool recursive);
extern "C" KAI_API KRetCode KAI_Lib_listExpressions(KHSession hSession, KHLibrary hLib, KJsonStrList * pslExpressionInfo, KPathString sPath, bool recursive);
extern "C" KAI_API KRetCode KAI_Lib_listOptimizers(KHSession hSession, KHLibrary hLib, KJsonStrList * pslOptimizerInfo, KPathString sPath, bool recursive);

extern "C" KAI_API KRetCode KAI_Lib_set(KHSession hSession, KHLibrary hLib, KPathString sComponentlPath, KString sProperty, KString sValue);
extern "C" KAI_API KRetCode KAI_Lib_move(KHSession hSession, KHLibrary hLib, KPathString sComponentlPath, KPathString sDestFolder);
extern "C" KAI_API KRetCode KAI_Lib_rename(KHSession hSession, KHLibrary hLib, KPathString sComponentlPath, KString sNewName);
extern "C" KAI_API KRetCode KAI_Lib_delete(KHSession hSession, KHLibrary hLib, KPathString sComponentlPath);

extern "C" KAI_API KRetCode KAI_Session_set_callback(KHSession hSession, Ken_session_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux);

extern "C" KAI_API KRetCode KAI_Component_bind(KHSession hSession, KHComponent hComponent1, KHComponent hComponent2, KString sRelation);
extern "C" KAI_API KRetCode KAI_Component_regist(KHSession hSession, KHLibrary hLib, KHComponent hComponent, KPathString sNewComponentPath);
extern "C" KAI_API KRetCode KAI_Component_touch(KHSession hSession, KHComponent hComponent);
extern "C" KAI_API KRetCode KAI_Component_set(KHSession hSession, KHComponent hComponent, KString sProperty, KString sValue);
extern "C" KAI_API KRetCode KAI_Component_update(KHSession hSession, KHComponent hComponent);
extern "C" KAI_API KRetCode KAI_Component_close(KHSession hSession, KHComponent hComponent);

extern "C" KAI_API KRetCode KAI_Component_get_property(KHComponent hComponent, KString sKey, KaiValue * pvValue);
extern "C" KAI_API KRetCode KAI_Component_get_str_property(KHComponent hComponent, KString sKey, KString * psValue);
extern "C" KAI_API KRetCode KAI_Component_get_int_property(KHComponent hComponent, KString sKey, KInt * pnValue);

extern "C" KAI_API KRetCode KAI_Component_set_property(KHSession hSession, KHComponent hComponent, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Component_dump_property(KHComponent hComponent, KString sKey, KString sTitle);
extern "C" KAI_API KRetCode KAI_Component_dump_binding_blocks(KHComponent hComponent);

extern "C" KAI_API KRetCode KAI_Component_set_datafeed_callback(KHComponent hComponent, Ken_datafeed_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux);
extern "C" KAI_API KRetCode KAI_Component_set_train_callback(KHComponent hComponent, Ken_train_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux);
extern "C" KAI_API KRetCode KAI_Component_set_test_callback(KHComponent hComponent, Ken_test_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux);
extern "C" KAI_API KRetCode KAI_Component_set_visualize_callback(KHComponent hComponent, Ken_visualize_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux);
//extern "C" KAI_API KRetCode KAI_Component_set_predict_callback(KHComponent hComponent, void* pCbInst, void* pCbFuncStart, void* pCbFuncData, void* pCbFuncEnd, void* pCbAux);

extern "C" KAI_API KRetCode KAI_value_dump(KaiValue vData, KString sTile);

extern "C" KAI_API KRetCode KAI_Model_get_builtin_names(KHSession hSession, KStrList * pslNames);
extern "C" KAI_API KRetCode KAI_Model_create(KHSession hSession, KHModel * phModel, KString sBuiltin, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Model_download(KHSession hSession, KHLibrary hLib, KHModel * phModel, KPathString sModelPath);
extern "C" KAI_API KRetCode KAI_Model_regist(KHSession hSession, KHLibrary hLib, KHModel hModel, KPathString sNewModelPath);
extern "C" KAI_API KRetCode KAI_Model_touch(KHSession hSession, KHModel hModel);
extern "C" KAI_API KRetCode KAI_Model_set(KHSession hSession, KHModel hModel, KString sProperty, KString sValue);
extern "C" KAI_API KRetCode KAI_Model_update(KHSession hSession, KHModel hModel);
extern "C" KAI_API KRetCode KAI_Model_close(KHSession hSession, KHModel hModel);

extern "C" KAI_API KRetCode KAI_Model_train(KHModel hModel, KaiDict kwArgs, KBool bAsync=false);
extern "C" KAI_API KRetCode KAI_Model_test(KHModel hModel, KaiDict kwArgs, KBool bAsync = false);
extern "C" KAI_API KRetCode KAI_Model_visualize(KHModel hModel, KaiDict kwArgs, KBool bAsync = false);
extern "C" KAI_API KRetCode KAI_Model_predict(KHModel hModel, KaiDict kwArgs, KaiList* pdResult);
extern "C" KAI_API KRetCode KAI_Model_get_trained_epoch_count(KHModel hModel, KInt* pnEpochCount);
extern "C" KAI_API KRetCode KAI_Model_get_instance(KHModel hModel, KHModelInstance* phModelInst, KaiDict kwArgs);

extern "C" KAI_API KRetCode KAI_Model_Instance_train(KHModelInstance hModelInst, KaiDict kwArgs, KBool bAsync = false);
extern "C" KAI_API KRetCode KAI_Model_Instance_test(KHModelInstance hModelInst, KaiDict kwArgs, KBool bAsync = false);
extern "C" KAI_API KRetCode KAI_Model_Instance_visualize(KHModelInstance hModelInst, KaiDict kwArgs, KBool bAsync = false);
extern "C" KAI_API KRetCode KAI_Model_Instance_predict(KHModelInstance hModelInst, KaiDict kwArgs, KaiList* pdResult);
extern "C" KAI_API KRetCode KAI_Model_Instance_get_trained_epoch_count(KHModelInstance hModelInst, KInt * pnEpochCount);

extern "C" KAI_API KRetCode KAI_Dataset_get_builtin_names(KHSession hSession, KStrList * pslNames);
extern "C" KAI_API KRetCode KAI_Dataset_create(KHSession hSession, KHDataset* phDataset, KString sBuiltin, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Dataset_download(KHSession hSession, KHLibrary hLib, KHDataset * phDataset, KPathString sDatasetPath);
extern "C" KAI_API KRetCode KAI_Dataset_regist(KHSession hSession, KHLibrary hLib, KHDataset hDataset, KPathString sNewDatasetPath);
extern "C" KAI_API KRetCode KAI_Dataset_touch(KHSession hSession, KHDataset hDataset);
extern "C" KAI_API KRetCode KAI_Dataset_set(KHSession hSession, KHDataset hDataset, KString sProperty, KString sValue);
extern "C" KAI_API KRetCode KAI_Dataset_update(KHSession hSession, KHDataset hDataset);
extern "C" KAI_API KRetCode KAI_Dataset_close(KHSession hSession, KHDataset hDataset);

extern "C" KAI_API KRetCode KAI_Dataset_read_file(KHDataset hDataset, KString sDataFilePath);

extern "C" KAI_API KRetCode KAI_Network_get_builtin_names(KHSession hSession, KStrList * pslNames);
extern "C" KAI_API KRetCode KAI_Network_create(KHSession hSession, KHNetwork * phNetwork, KString sBuiltin, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Network_download(KHSession hSession, KHLibrary hLib, KHNetwork * phNetwork, KPathString sNetworkPath);
extern "C" KAI_API KRetCode KAI_Network_regist(KHSession hSession, KHLibrary hLib, KHNetwork hNetwork, KPathString sNewNetworkPath);
extern "C" KAI_API KRetCode KAI_Network_touch(KHSession hSession, KHNetwork hNetwork);
extern "C" KAI_API KRetCode KAI_Network_set(KHSession hSession, KHNetwork hNetwork, KString sProperty, KString sValue);
extern "C" KAI_API KRetCode KAI_Network_update(KHSession hSession, KHNetwork hNetwork);
extern "C" KAI_API KRetCode KAI_Network_close(KHSession hSession, KHNetwork hNetwork);

extern "C" KAI_API KRetCode KAI_Network_regist_macro(KHSession hSession, KHNetwork hNetwork, KString sMacroName);

extern "C" KAI_API KRetCode KAI_Network_append_layer(KHSession hSession, KHNetwork hNetwork, KHLayer hLayer);
extern "C" KAI_API KRetCode KAI_Network_append_named_layer(KHSession hSession, KHNetwork hNetwork, KString sLayerName, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Network_append_custom_layer(KHSession hSession, KHNetwork hNetwork, KString sLayerName, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Network_append_subnet(KHSession hSession, KHNetwork hNetwork, KHNetwork hSubnet);

extern "C" KAI_API KRetCode KAI_Network_get_layer_count(KHSession hSession, KHNetwork hNetwork, KInt * pnLayerCount);
extern "C" KAI_API KRetCode KAI_Network_get_nth_layer(KHSession hSession, KHNetwork hNetwork, KInt nth, KHLayer * phLayer);

extern "C" KAI_API KRetCode KAI_Expression_get_builtin_names(KHSession hSession, KStrList * pslNames);
extern "C" KAI_API KRetCode KAI_Expression_create(KHSession hSession, KHExpression * phExpression, KString sBuiltin, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Expression_download(KHSession hSession, KHLibrary hLib, KHExpression * phExpression, KPathString sExpressionPath);
extern "C" KAI_API KRetCode KAI_Expression_regist(KHSession hSession, KHLibrary hLib, KHExpression hExpression, KPathString sNewExpressionPath);
extern "C" KAI_API KRetCode KAI_Expression_touch(KHSession hSession, KHExpression hExpression);
extern "C" KAI_API KRetCode KAI_Expression_set(KHSession hSession, KHExpression hExpression, KString sProperty, KString sValue);
extern "C" KAI_API KRetCode KAI_Expression_update(KHSession hSession, KHExpression hExpression);
extern "C" KAI_API KRetCode KAI_Expression_close(KHSession hSession, KHExpression hExpression);

//extern "C" KAI_API KRetCode KAI_Exp_get_operator(KHExpression hExpression, KString* psOpCode, KString * psOpAux);
//extern "C" KAI_API KRetCode KAI_Exp_get_operand_count(KHExpression hExpression, KInt* pnOpndCnt);
//extern "C" KAI_API KRetCode KAI_Exp_get_nth_operand(KHExpression hExpression, KInt nth, KHExpression *phOperand);

extern "C" KAI_API KRetCode KAI_Optimizer_get_builtin_names(KHSession hSession, KStrList * pslNames);
extern "C" KAI_API KRetCode KAI_Optimizer_create(KHSession hSession, KHOptimizer * phOptimizer, KString sBuiltin, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Optimizer_download(KHSession hSession, KHLibrary hLib, KHOptimizer * phOptimizer, KPathString sOptimizerPath);
extern "C" KAI_API KRetCode KAI_Optimizer_regist(KHSession hSession, KHLibrary hLib, KHOptimizer hOptimizer, KPathString sNewOptimizerPath);
extern "C" KAI_API KRetCode KAI_Optimizer_touch(KHSession hSession, KHOptimizer hOptimizer);
extern "C" KAI_API KRetCode KAI_Optimizer_set(KHSession hSession, KHOptimizer hOptimizer, KString sProperty, KString sValue);
extern "C" KAI_API KRetCode KAI_Optimizer_update(KHSession hSession, KHOptimizer hOptimizer);
extern "C" KAI_API KRetCode KAI_Optimizer_close(KHSession hSession, KHOptimizer hOptimizer);

extern "C" KAI_API KRetCode KAI_Layer_create(KHSession hSession, KHLayer * phLayer, KString sBuiltin, KaiDict kwArgs);
extern "C" KAI_API KRetCode KAI_Layer_close(KHSession hSession, KHLayer hLayer);

extern "C" KAI_API KRetCode KAI_Object_get_type(KHObject hObject, Ken_object_type * pObj_type);

extern "C" KAI_API KRetCode KAI_Array_get_int_data(KHSession hSession, KHObject hObject, KInt nStart, KInt nCount, KInt* pBuffer);
extern "C" KAI_API KRetCode KAI_Array_get_float_data(KHSession hSession, KHObject hObject, KInt nStart, KInt nCount, KFloat * pBuffer);

extern "C" KAI_API KRetCode KAI_download_float_data(KHSession hSession, KInt nToken, KInt nSize, KFloat * pBuffer);
extern "C" KAI_API KRetCode KAI_download_int_data(KHSession hSession, KInt nToken, KInt nSize, KInt * pBuffer);

extern "C" KAI_API KRetCode KAI_debug_session_component_dump(KHSession hSession);
extern "C" KAI_API KRetCode KAI_debug_curr_count(KHSession hSession, KInt * pnObjCnt, KInt * pnValCnt);

extern "C" KAI_API KRetCode KAI_Util_fft(KHSession hSession, KFloat* pWave, KFloat* pFTT, KInt file_count, KInt fetch_width, KInt step_width, KInt step_cnt, KInt fft_width, KInt freq_cnt);