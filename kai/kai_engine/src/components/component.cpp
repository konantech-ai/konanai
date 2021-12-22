/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "component.h"
#include "../library/library.h"
#include "../utils/kutil.h"
#include "../utils/klogger.h"

int KaiComponent::ms_checkCodeComponent = 14239174;
int KaiComponent::ms_nNextSeqID = 1;

KaiComponent::KaiComponent(KaiSession* pSession, Ken_component_type ctype, Ken_object_type otype) : KaiObject(otype) {
	throw KaiException(KERR_PRECIATED_FUNCTION);

	m_componentType = ctype;

	if (pSession) pSession->regist(this);

	m_propDict["version"] = 1;

	m_bDirty = false;
	m_nComponentSeqId = ms_nNextSeqID++;
}

KaiComponent::KaiComponent(KaiSession* pSession, Ken_component_type ctype, Ken_object_type otype, KaiDict kwArgs) : KaiObject(otype) {
	m_pSession = pSession;

	if (pSession) pSession->regist(this);

	m_componentType = ctype;

	m_pOwnLibrary = NULL;
	m_nComponentId = 0;
	m_nComponentSeqId = ms_nNextSeqID++;

	set_property_dict(kwArgs);

	m_checkCode = 0; // temp: 파생클래스 생성자에서 올바른 값 지정할 것

	m_propDict["version"] = 1;

	m_bDirty = false;
	m_checkCodeComponent = ms_checkCodeComponent;
}

KaiComponent::KaiComponent(KaiSession* pSession, Ken_component_type ctype, Ken_object_type otype, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiObject(otype) {
	m_pSession = pSession;

	m_componentType = ctype;

	m_pOwnLibrary = pLib;
	m_nComponentId = pComponentInfo->getComponentId();
	m_nComponentSeqId = ms_nNextSeqID++;

	m_bDirty = false;

	KaiDict& info = pComponentInfo->m_propDict;
	
	for (auto it = info.begin(); it != info.end(); it++) {
		m_propDict[it->first] = it->second;
	}

	m_checkCode = 0; // temp: 파생클래스 생성자에서 올바른 값 지정할 것

	if (m_propDict.find("version") == m_propDict.end()) m_propDict["version"] = 1;

	m_bDirty = false;
	m_checkCodeComponent = ms_checkCodeComponent;
}

KaiComponent::~KaiComponent() {
	//if (m_pSession) m_pSession->unregist(this);
}

void KaiComponent::close() {
	if (m_pSession) m_pSession->unregist(this);
	m_pSession = NULL;
	destroy();
	//delete this;
}

void KaiComponent::destroy_without_report() {
	m_pSession = NULL;
	destroy();
	//delete this;
}

KaiComponent* KaiComponent::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Component");

	KaiComponent* pComponent = (KaiComponent*)hObject;

	if (pComponent->m_checkCodeComponent != ms_checkCodeComponent) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Component");
	if (pComponent->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Component");

	return pComponent;
}

KaiComponent* KaiComponent::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Component");

	KaiComponent* pComponent = (KaiComponent*)hObject;

	if (pComponent->m_checkCodeComponent != ms_checkCodeComponent) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Component");

	return pComponent;
}

void KaiComponent::refreshDirty() {
	if (this == NULL) return;
	if (!m_bDirty) return;
	m_propDict["version"] = (KInt)m_propDict["version"] + 1;
	m_bDirty = false;
}

void KaiComponent::bind(KaiComponent* pComponent, KString relation, bool bExternal, bool bUnique) {
	m_bDirty = true;
	m_pSession->reportBinding(this, pComponent, relation, bExternal, bUnique);
}

void KaiComponent::unbind(KaiComponent* pComponent, bool bReport) {
	m_bDirty = true;
	if (bReport) m_pSession->reportUnbinding(this, pComponent);
}

void KaiComponent::regist(KaiLibrary* pLib, KString sNewComponentlPath) {
	KaiComponentInfo* pComponentInfo = pLib->createComponent(sNewComponentlPath, m_componentType);

	KaiDict& info = pComponentInfo->m_propDict;

	for (auto it = m_propDict.begin(); it != m_propDict.end(); it++) {
		info[it->first] = it->second;
	}
}

void KaiComponent::touch() {
	m_bDirty = true;
}

void KaiComponent::set_property(KString sProperty, KaiValue vValue) {
	m_bDirty = true;
	m_propDict[sProperty] = vValue;
}

void KaiComponent::update() {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiValue KaiComponent::get_property(KString sKey) {
	if (m_propDict.find(sKey) == m_propDict.end()) throw KaiException(KERR_KEY_NOT_FOUND_ON_GET_POPERTY);
	return m_propDict[sKey];
}

KaiValue KaiComponent::get_property(KString sKey, KaiValue def) {
	if (m_propDict.find(sKey) != m_propDict.end()) return m_propDict[sKey];
	return def;
}

KaiValue KaiComponent::get_info_prop(KaiDict info, KString sKey, KaiValue def) {
	if (info.find(sKey) != info.end()) return info[sKey];
	return def;
}

KaiValue KaiComponent::get_set_info_prop(KaiDict info, KString sKey, KaiValue def) {
	if (info.find(sKey) != info.end()) return info[sKey];
	info[sKey] = def;
	return def;
}

KaiValue KaiComponent::m_seek_property(KString sKey, KaiValue def, KaiDict kwArgs1, KaiDict kwArgs2) {
	if (kwArgs1.find(sKey) != kwArgs1.end()) return kwArgs1[sKey];
	if (kwArgs2.find(sKey) != kwArgs2.end()) return kwArgs2[sKey];
	if (m_propDict.find(sKey) != m_propDict.end()) return m_propDict[sKey];
	return def;
}

KString KaiComponent::get_desc() {
	return (KString)m_propDict["desc"];
}

void KaiComponent::set_property_dict(KaiDict kwArgs) {
	m_bDirty = true;
	for (auto it = kwArgs.begin(); it != kwArgs.end(); it++) {
		m_propDict[it->first] = it->second;
	}
}

void KaiComponent::dump_property(KString sKey, KString sTitle) {
	KString sDesc;
	if (m_propDict.find(sKey) == m_propDict.end()) {
		sDesc = "<undefined>";
	}
	else {
		sDesc = m_propDict[sKey].desc();
	}
	logger.Print("%s %s", sTitle.c_str(), sDesc.c_str());
}

void KaiComponent::collect_properties(KaiComponent* pInst, KString sCompName){
	if (this == NULL) return;

	KaiDict clone;

	for (auto& it : m_propDict) {
		if (it.first == "#callback") {
			m_pushCallbackPropert(pInst, it.second);
		}
		else if (it.first[0] == '#') {
			printf("special property in %s: %s\n", sCompName.c_str(), it.first.c_str());
		}
		else {
			clone[it.first] = it.second;
			pInst->set_property(it.first, it.second);
		}
	}

	pInst->set_property(sCompName, clone);
}

KaiDict KaiComponent::m_copy_properties() {
	KaiDict clone;

	for (auto& it : m_propDict) {
		clone[it.first] = it.second;
	}

	return clone;
}

/*
KaiDict KaiComponent::copy_properties() {
	KaiDict clone;

	for (auto& it : m_propDict) {
		clone[it.first] = it.second;
	}

	return clone;
}

void KaiComponent::push_properties(KaiDict& dict) {
	for (auto& it : m_propDict) {
		dict[it.first] = it.second;
	}
}
*/

KaiDict KaiComponent::resolve_copy_properties(KaiDict call_info) {
	KaiDict clone;

	for (auto& it : call_info) {
		clone[it.first] = it.second;
	}

	for (auto& it : m_propDict) {
		if (it.second.type() == Ken_value_type::string && ((KString)it.second)[0] == '#') {
			KString arg_name = ((KString)it.second).substr(1);
			//if (call_info.find(arg_name) == call_info.end()) throw KaiException(KERR_FAIL_TO_RESOLVE_MACRO_ARGUMENT, arg_name);
			if (call_info.find(arg_name) == call_info.end()); // 매크로 인자가 해소되지 않을 경우 성급한 예외 처리 대신 정보 없음으로 처리해 디폴트값 액세스가 이루어지도록 한다.
			else clone[it.first] = call_info[arg_name];
		}
		else {
			clone[it.first] = it.second;
		}
	}

	return clone;
}

void KaiComponent::m_set_default(KString sKey, KaiValue kDefault) {
	m_bDirty = true;
	if (m_propDict.find(sKey) == m_propDict.end()) {
		m_propDict[sKey] = kDefault;
	}
}

void KaiComponent::set_datafeed_callback(Ken_datafeed_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux) {
	m_save_cb_info(Ken_cb_family::datafeed, (int)cb_event, pCbInst, pCbFunc, pCbAux);
}

void KaiComponent::set_train_callback(Ken_train_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux) {
	m_save_cb_info(Ken_cb_family::train, (int)cb_event, pCbInst, pCbReport, pCbAux);
}

void KaiComponent::set_test_callback(Ken_test_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux) {
	m_save_cb_info(Ken_cb_family::test, (int)cb_event, pCbInst, pCbReport, pCbAux);
}

void KaiComponent::set_visualize_callback(Ken_visualize_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux) {
	m_save_cb_info(Ken_cb_family::visualize, (int)cb_event, pCbInst, pCbReport, pCbAux);
}

/*
void KaiComponent::set_visualize_callback(void* pCbInst, void* pCbFuncStart, void* pCbFuncData, void* pCbFuncEnd, void* pCbAux) {
	m_save_cb_info(Ken_cb_family::visualize, 0, pCbInst, pCbFuncStart, pCbAux);
	m_save_cb_info(Ken_cb_family::visualize, 1, pCbInst, pCbFuncData, pCbAux);
	m_save_cb_info(Ken_cb_family::visualize, 2, pCbInst, pCbFuncEnd, pCbAux);
}

void KaiComponent::set_predict_callback(void* pCbInst, void* pCbFuncStart, void* pCbFuncData, void* pCbFuncEnd, void* pCbAux) {
	m_save_cb_info(Ken_cb_family::predict, 0, pCbInst, pCbFuncStart, pCbAux);
	m_save_cb_info(Ken_cb_family::predict, 1, pCbInst, pCbFuncData, pCbAux);
	m_save_cb_info(Ken_cb_family::predict, 2, pCbInst, pCbFuncEnd, pCbAux);
}
*/

void KaiComponent::m_pushCallbackPropert(KaiComponent* pDest, KaiDict cbFamilies) {
	for (auto& it1 : cbFamilies) {
		KString sFamilyKey = it1.first;
		KaiDict cb_events = it1.second;

		for (auto& it2 : cb_events) {
			KString sEventKey = it2.first;
			KaiList cb_functions = it2.second;
			for (auto& it3 : cb_functions) {
				KaiList cb_info = it3;
				pDest->m_regist_cb_info(sFamilyKey, sEventKey, cb_info);
			}
		}
	}
}

void KaiComponent::m_regist_cb_info(KString sFamilyKey, KString sEventKey, KaiList cb_info) {
	KaiDict cb_families;
	KaiDict cb_events;
	KaiList cb_functions;

	if (m_propDict.find("#callback") != m_propDict.end()) cb_families = m_propDict["#callback"];
	if (cb_families.find(sFamilyKey) != cb_families.end()) cb_events = cb_families[sFamilyKey];
	if (cb_events.find(sEventKey) != cb_events.end()) cb_functions = cb_events[sEventKey];

	cb_functions.push_back(cb_info);

	cb_events[sEventKey] = cb_functions;
	cb_families[sFamilyKey] = cb_events;

	m_propDict["#callback"] = cb_families;
}

void KaiComponent::m_save_cb_info(Ken_cb_family cb_family, int cb_event, void* pCbInst, void* pCbFunc, void* pCbAux) {
	KString sFamilyKey = std::to_string((int)cb_family);
	KString sEventKey = std::to_string(cb_event);

	KaiList cb_info{ (KInt)pCbFunc, (KInt)pCbInst, (KInt)pCbAux };

	m_regist_cb_info(sFamilyKey, sEventKey, cb_info);
	/*
	KaiDict cb_families;
	KaiDict cb_events;
	KaiList cb_functions;

	if (m_propDict.find("#callback") != m_propDict.end()) cb_families = m_propDict["#callback"];
	if (cb_families.find(sFamilyKey) != cb_families.end()) cb_events = cb_families[sFamilyKey];
	if (cb_events.find(sEventKey) != cb_events.end()) cb_functions = cb_events[sEventKey];

	cb_functions.push_back(cb_info);

	cb_events[sEventKey] = cb_functions;
	cb_families[sFamilyKey] = cb_events;

	m_propDict["#callback"] = cb_families;
	*/
}
