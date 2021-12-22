/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "../session/session.h"
#include "../components/knetwork.h"
#include "../library/local_library.h"
#include "../gpu_cuda/cuda_manager.h"
#include "../exec/callback.h"
#include "../utils/klogger.h"
#include "../utils/kutil.h"

//hs.cho
#ifdef KAI2021_WINDOWS
#include <io.h>
#include <direct.h>
#else
#include <limits.h>
#include "../nightly/findfirst.h"
#define _MAX_PATH PATH_MAX
#define _fullpath(absPath,relPath,size) realpath(relPath, absPath)
#define _mkdir(filepath)  mkdir(filepath, 0777)
#define sprintf_s snprintf
#endif



#include <sys/types.h>
#include <sys/stat.h>


//#include <filesystem>

KString KaiSession::ms_sKaiVersion = "0.0.0.1";

int KaiSession::ms_checkCode = 15967575;

KaiSession::KaiSession() : KaiObject(Ken_object_type::session), m_lastError(), m_instList(), m_bindList(), m_macro() {
	logger.open("kai", "session");
	
	cuda_man.OpenCuda();

	char fullpath[_MAX_PATH];
	_fullpath(fullpath, ".\\LocalLib", _MAX_PATH);
	
	m_sLocalLibPath = fullpath;
	
	if (!m_checkLocalLibExist()) {
		if (_mkdir(m_sLocalLibPath.c_str()) != 0) throw KaiException(KERR_FAIL_TO_CREATE_LOCAL_LIB_FOLDER, m_sLocalLibPath);
	}
	
	m_checkCode = ms_checkCode;

	nDebugObjectCount = 0;
	nDebugValueCount = 0;
}

KaiSession::~KaiSession() {
	m_checkCode = 0;

	for (auto it = m_openedLocalLibs.begin(); it != m_openedLocalLibs.end(); it++) {
		KaiLocalLibrary* pLib = *it;
		delete pLib;
	}
}

KaiSession* KaiSession::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Session");
	KaiSession* pSession = (KaiSession*)hObject;
	if (pSession->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Session");
	return pSession;
}

void KaiSession::SetLastError(KaiException ex) {
	m_lastError = ex;
}

KRetCode KaiSession::GetLastErrorCode() {
	return m_lastError.GetErrorCode();
}

KString KaiSession::GetLastErrorMessage() {
	return m_lastError.GetErrorMessage();
}

KString KaiSession::GetVersion() {
	return ms_sKaiVersion;
}

void KaiSession::RegistCallback(Ken_session_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux) {
	if (cb_event == Ken_session_cb_event::print) logger.setCallback(pCbInst, (KCbPrint*)pCbFunc);
	else throw KaiException(KERR_UNKNOWN_CALLBACK_TYPE);
}

void KaiSession::download_float_data(KInt nToken, KInt nSize, KFloat* pBuffer) {
	KaiCallbackTransaction::_ArrToken* pToken = (KaiCallbackTransaction::_ArrToken*)nToken;
	pToken->m_pTransaction->download_float_data(pToken, nSize, pBuffer);
}

void KaiSession::download_int_data(KInt nToken, KInt nSize, KInt* pBuffer) {
	KaiCallbackTransaction::_ArrToken* pToken = (KaiCallbackTransaction::_ArrToken*)nToken;
	pToken->m_pTransaction->download_int_data(pToken, nSize, pBuffer);
}

void KaiSession::GetLocalLibaryNames(KStrList* pslLibNames) {
	pslLibNames->clear();

	KString sLibPath = m_sLocalLibPath + "\\*.map";

	_finddata_t fd;
	intptr_t hObject;
	long long result = 1;

	hObject = _findfirst(sLibPath.c_str(), &fd);

	if (hObject == -1) return;

	while (result != -1) {
		KString fname = fd.name;
		if (fname .size() > 4 && fname.substr(fname.size()-4) == ".map") {
			pslLibNames->push_back(fname.substr(0, fname.size() - 4));
		}
		result = _findnext(hObject, &fd);
	}

	_findclose(hObject);
}

void KaiSession::DeleteLocalLibary(KString sLibName, bool bClose) {
	for (auto it = m_openedLocalLibs.begin(); it != m_openedLocalLibs.end(); it++) {
		KaiLocalLibrary* pLib = *it;
		if (pLib->getName() == sLibName) {
			if (bClose) {
				m_openedLocalLibs.erase(it);
				delete pLib;
				break;
			}
			else {
				throw KaiException(KERR_OPENED_LOCAL_LIB, sLibName);
			}
		}
	}

	if (!m_checkLocalLibExist(sLibName)) throw KaiException(KERR_UNKNOWN_LOCAL_LIB_NAME, sLibName);

	KString sLibPath = GetLocalLibFilePath(sLibName);
	kutil.remove_all(sLibPath);
}

bool KaiSession::CloseLocalLibary(KaiLocalLibrary* pLib) {
	auto it = std::find(m_openedLocalLibs.begin(), m_openedLocalLibs.end(), pLib);
	
	if  (it == m_openedLocalLibs.end()) return false;

	m_openedLocalLibs.erase(it);
	delete pLib;
	return true;
}

KString KaiSession::GetLocalLibFilePath(KString sLibName) {
	KString sLibPath = m_sLocalLibPath + "\\" + sLibName + ".map";
	return sLibPath;
}

void KaiSession::GetInstallModels(KStrList* pslModuleNames) {
	KaiLocalLibrary::GetInstallModels(pslModuleNames);
}

void KaiSession::GetProperties(Ken_component_type componentType, KStrList* pslProps) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiLocalLibrary* KaiSession::CreateLocalLibrary(KString sLibName, KString sPassword, Ken_inst_mode enumInstallMode, KStrList slModelNames) {
	if (m_checkLocalLibExist(sLibName)) throw KaiException(KERR_EXISTING_LOCAL_LIB_NAME, sLibName);

	KaiLocalLibrary* pLib = new KaiLocalLibrary(this, sLibName, sPassword, enumInstallMode, slModelNames);
	m_openedLocalLibs.push_back(pLib);

	return pLib;
}

void KaiSession::OpenLocalLibrary(KHLibrary* phLib, KString sLibName, KString sPassword) {
	if (!m_checkLocalLibExist(sLibName)) throw KaiException(KERR_UNKNOWN_LOCAL_LIB_NAME, sLibName);

	bool bAlreadyOpened = false;

	for (auto it2 = m_openedLocalLibs.begin(); it2 != m_openedLocalLibs.end(); it2++) {
		KaiLocalLibrary* pLib = *it2;
		if (pLib->getName() == sLibName) bAlreadyOpened = true;
	}

	KaiLocalLibrary* pLib = new KaiLocalLibrary(this, sLibName, sPassword);
	m_openedLocalLibs.push_back(pLib);

	*phLib = (KHLibrary)pLib;

	if (bAlreadyOpened) throw KaiException(KWARN_ALREADY_OPENED_LOCAL_LIB, sLibName);
}

KaiPublicLibrary* KaiSession::ConnectToPublicLibrary(KString sIpAddr, KString sPort, KString sLibName, KString sUserName, KString sPassword) {
	throw KaiException(KERR_WILL_BE_SUPPORTED_IN_NEXT_VERSION);
}

void KaiSession::debug_component_dump() {
	logger.Print("Components:");
	int n = 0, m = 0;
	for (auto& it : m_instList) {
		KaiComponent* pComponent = (KaiComponent*)(KHObject)it;
		logger.Print("    Component-%d: %s [ref:%d]", n++, pComponent->get_desc().c_str(), pComponent->get_ref_count());
	}

	logger.Print("Binding Relations:");
	for (auto& it : m_bindList) {
		KaiComponent* pBinding = (KaiComponent*)(KHObject)it.binding;
		KaiComponent* pBound = (KaiComponent*)(KHObject)it.bound;
		logger.Print("    Pair-%d: (%s, %s, %s)", m++, pBinding->get_desc().c_str(), it.relation.c_str(), pBound->get_desc().c_str());
	}
}

void KaiSession::regist(KaiComponent* pComponent) {
	m_instList.push_back(pComponent);
}

void KaiSession::unregist(KaiComponent* pComponent) {
	auto it = m_instList.find((KHComponent)pComponent);
	if (it != m_instList.end()) m_instList.erase(it);

	KInt n = 0;

	while (n < (KInt)m_bindList.size()) {
		_bind_pair bindInfo = m_bindList[n];
		if (bindInfo.binding == pComponent) {
			auto it = m_bindList.begin() + n;
			m_bindList.erase(it);
		}
		else if (bindInfo.bound == pComponent) {
			auto it = m_bindList.begin() + n;
			m_bindList.erase(it);
		}
		else {
			n++;
		}
	}
}

void KaiSession::reportBinding(KaiComponent* pBinding, KaiComponent* pBound, KString relation, bool bExternal, bool bUnique) {
	pBound->fetch();

	if (bUnique) {
		for (auto it = m_bindList.begin(); it != m_bindList.end(); it++) {
			if (pBinding == (*it).binding && relation == (*it).relation) {
				m_bindList.erase(it);
				pBound->destroy();
				break;
			}
		}
	}

	struct _bind_pair componentPair;

	componentPair.binding = pBinding;
	componentPair.bound = pBound;
	componentPair.relation = relation;

	m_bindList.push_back(componentPair);
}

void KaiSession::reportUnbinding(KaiComponent* pBinding, KaiComponent* pBound) {
	for (auto it = m_bindList.begin(); it != m_bindList.end(); it++) {
		if (pBinding == (*it).binding && pBound == (*it).bound) {
			m_bindList.erase(it);
			pBound->destroy();
			break;
		}
	}
}

KaiComponent* KaiSession::get_bound_component(KaiComponent* pBinding, KString sKey, KaiComponent* pDefault) {
	for (auto it = m_bindList.begin(); it != m_bindList.end(); it++) {
		if (pBinding == (*it).binding && sKey == (*it).relation) {
			return (KaiComponent*)(KHObject)((*it).bound);
		}
	}

	return pDefault;
}

void KaiSession::dump_binding_blocks(KaiComponent* pBinding) {
	m_dump_binding_blocks(pBinding, 0);
	logger.Print("%s:", pBinding->get_desc().c_str());
}

void KaiSession::m_dump_binding_blocks(KaiComponent* pBinding, int depth) {
	for (auto it = m_bindList.begin(); it != m_bindList.end(); it++) {
		if (pBinding == (*it).binding) {
			logger.Print("%*s ==> (%s) : %s", 2*depth, "", (*it).relation.c_str(), ((KaiComponent*)(KHObject)((*it).bound))->get_desc().c_str());
			m_dump_binding_blocks((KaiComponent*)(KHObject)(*it).bound, depth+1);
		}
	}
}

KBool KaiSession::areCleanBoundComponents(KaiComponent* pBinding) {
	for (auto it = m_bindList.begin(); it != m_bindList.end(); it++) {
		if (pBinding == (*it).binding) {
			KaiComponent* pBound = (KaiComponent*)(KHObject)(*it).bound;
			if (pBound->isDirty()) {
				return false;
			}
		}
	}

	return true;
}

bool KaiSession::m_checkLocalLibExist(KString sLibName) {
	KString sLibPath = m_sLocalLibPath;
	if (sLibName != "") sLibPath += "\\" + sLibName + ".map";

	struct stat info;

	if (stat(sLibPath.c_str(), &info) != 0) return false;

	if (sLibName == "") {
		if (!(info.st_mode & S_IFDIR)) throw KaiException(KERR_BROKEN_LOCAL_LIB_FOLDER);
	}
	else {
		if (info.st_mode & S_IFDIR) throw KaiException(KERR_BROKEN_LOCAL_LIB_FILE, sLibName);
	}

	return true;
}

KString KaiSession::desc() {
	char buf[128];

	sprintf_s(buf,128, "<Session 0x%llx>", (KInt)(KHObject)this);

	return buf;
}

void KaiSession::regist_macro(KaiNetwork* pNetwork, KString sMacroName) {
	m_macro[sMacroName] = (KHObject)pNetwork;
}

KaiNetwork* KaiSession::get_macro(KString sMacroName) {
	if (m_macro.find(sMacroName) == m_macro.end()) throw KaiException(KERR_UNREGISTED_MACRO_WAS_CALLED, sMacroName);
	return (KaiNetwork*)(KHObject)m_macro[sMacroName];
}
