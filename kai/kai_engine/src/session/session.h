/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../include/kai_types.h"
#include "../include/kai_value.hpp"

#include "../session/kcommon.h"
#include "../session/callback_manager.h"

#include "../utils/kexception.h"

#include <string.h>
#include <iostream>
#include <fstream>
//#include <filesystem>

class KaiLocalLibrary;
class KaiPublicLibrary;

class KaiComponent;
class KaiModel;
class KaiNetwork;

#include <map>

typedef std::vector<KaiLocalLibrary*> LocalLibList;

struct _bind_pair {
	KaiValue binding;	// 실제로는 KaiComponent*만을 저장하지만 KaiValue에 저장함으로써 자동 메모리 관리가 가능해진다.
	KaiValue bound;		// 실제로는 KaiComponent*만을 저장하지만 KaiValue에 저장함으로써 자동 메모리 관리가 가능해진다.
	KString relation;
};

class KaiSession : public KaiObject {
public:
	KaiSession();
	virtual ~KaiSession();

	static KaiSession* HandleToPointer(KHObject hObject);

	void SetLastError(KaiException ex);
	
	KRetCode GetLastErrorCode();
	KString GetLastErrorMessage();
	
	KString GetVersion();
	
	void GetLocalLibaryNames(KStrList* pslLibNames);
	void DeleteLocalLibary(KString sLibName, bool bClose=false);
	void GetInstallModels(KStrList* pslModuleNames);

	void GetProperties(Ken_component_type componentType, KStrList* pslProps);

	bool CloseLocalLibary(KaiLocalLibrary* pLib);

	KaiLocalLibrary* CreateLocalLibrary(KString sLibName, KString sPassword, Ken_inst_mode enumInstallMode, KStrList slModelNames);
	void OpenLocalLibrary(KHLibrary* phLib, KString sLibName, KString sPassword);

	KaiPublicLibrary* ConnectToPublicLibrary(KString sIpAddr, KString sPort, KString sLibName, KString sUserName, KString sPassword);
	
	KString GetLocalLibFilePath(KString sLibName);

	void regist(KaiComponent* pComponent);
	void unregist(KaiComponent* pComponent);

	void reportBinding(KaiComponent* pBinding, KaiComponent* pBound, KString relation, bool bExternal, bool bUnique);
	void reportUnbinding(KaiComponent* pBinding, KaiComponent* pBound);

	void dump_binding_blocks(KaiComponent* pBinding);

	KaiComponent* get_bound_component(KaiComponent* pBinding, KString sKey, KaiComponent* pDefault=NULL);

	KBool areCleanBoundComponents(KaiComponent* pBinding);

	void debug_component_dump();
	static KInt debug_get_curr_obj_count() { return ms_debug_count; }

	void regist_macro(KaiNetwork* pNetwork, KString sMacroName);
	KaiNetwork* get_macro(KString sMacroName);

	KString desc();

	void download_float_data(KInt nToken, KInt nSize, KFloat* pBuffer);
	void download_int_data(KInt nToken, KInt nSize, KInt* pBuffer);

	KInt nDebugObjectCount;
	KInt nDebugValueCount;

protected:
	bool m_checkLocalLibExist(KString sLibName="");
	void m_dump_binding_blocks(KaiComponent* pBinding, int depth);

protected:
	LocalLibList m_openedLocalLibs;

	int m_checkCode;
	KString m_sLocalLibPath;

	KaiException m_lastError;

	KaiList m_instList;

	KaiDict m_macro;

	std::vector<_bind_pair> m_bindList;

	static KString ms_sKaiVersion;
	static int ms_checkCode;

// 여기부터는 임시 기능인 Shell 지원 기능. 차차 삭제될 예정임
public:
	void OpenCudaOldVersion();
	void GetMissionNames(KStrList *pslMissionNames);
	void RegistCallback(Ken_session_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux);
	void GetPluginTypeNames(KStrList* pslPluginTypeNames);
	void GetPluginNomNames(KString sPluginTypeName, KStrList* pslPluginNomNames);
	void SetCudaOption(KStrList tokens);
	void GetCudaOption(int* pnDeviceCnt, int* pnDeviceNum, int* pnBlockSize, KInt * pnAvailMem, KInt * pnUsingMem);
	void SetImageOption(KStrList tokens);
	void GetImageOption(KBool * pbImgScreen, KString * psImgFolder);
	void SetSelectOption(KStrList slTokens);

	KString GetSelectDesc(KString sTypeName);

	void ExecMission(KString sMissionName, KString sExecMode);

protected:
};
