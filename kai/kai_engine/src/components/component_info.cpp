/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "component_info.h"
#include "../library/local_folder.h"
#include "../utils/kutil.h"
#include "../utils/kutil.h"

KString KaiComponentInfo::ms_componentTypeName[] = { "model", "dataset", "dataloader", "network", "expression", "optimizer" };

KaiComponentInfo::KaiComponentInfo(LocalFolder* pFolder, KString sName, Ken_component_type componentType, int nComponentId) : m_propDict() {
	m_nComponentId = nComponentId;
	m_pOwnFolder = pFolder;
	m_sComponentName = sName;
	m_componentType = componentType;

	m_checkCode = 0;
}

KaiComponentInfo::~KaiComponentInfo() {
	m_checkCode = 0;
}

/*
KaiComponentInfo* KaiComponentInfo::Create(LocalFolder* pFolder, KString sName, Ken_component_type componentType) {
	switch (componentType) {
	case Ken_component_type::model:
		return new KaiModel(pFolder, sName);
	case Ken_component_type::dataset:
		return new KaiDataset(pFolder, sName);
	case Ken_component_type::dataloader:
		return new KaiDataloader(pFolder, sName);
	case Ken_component_type::network:
		return new KaiNetwork(pFolder, sName);
	case Ken_component_type::expression:
		return new KaiExpression(pFolder, sName);
	case Ken_component_type::optimizer:
		return new KaiOptimizer(pFolder, sName);
	default:
		throw KaiException(KERR_UNKNOWN_COMPONENT_TYPE, sName);
	}
}
*/

KaiSession* KaiComponentInfo::getSession() {
	return m_pOwnFolder->getSession();
}

void KaiComponentInfo::typeCheck(Ken_component_type componentType) {
	if (m_componentType != componentType) throw KaiException(KERR_COMPONENT_TYPE_MISMATCH, m_sComponentName);
}

void KaiComponentInfo::setInitialProps(KString sProps) {
	m_propDict["init"] = sProps;
}

void KaiComponentInfo::setProperty(KString sKey, KString sValue) {
	m_propDict[sKey] = sValue;
}

void KaiComponentInfo::setProperty(KString sKey, int nValue) {
	m_propDict[sKey] = std::to_string(nValue);
}

void KaiComponentInfo::serialize(FILE* fid) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	Util::save_int(fid, (int) m_propDict.size());

	for (auto it = m_propDict.begin(); it != m_propDict.end(); it++) {
		Util::save_str(fid, it->first);
		kutil.save_value(fid, it->second);
	}
	*/
}

void KaiComponentInfo::unserialize(FILE* fid) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	int nCnt = Util::read_int(fid);

	for (int n = 0; n < nCnt; n++) {
		KString sKey = Util::read_str(fid);
		KaiValue vValue = kutil.read_value(fid);

		m_propDict[sKey] = vValue;
	}
	*/
}

void KaiComponentInfo::moveFolder(LocalFolder* pDestFolder) {
	if (m_pOwnFolder == pDestFolder) return;

	pDestFolder->checkDuplicatedName(m_componentType, m_sComponentName);

	m_pOwnFolder->unregistComponent(this);
	m_pOwnFolder = pDestFolder;
	m_pOwnFolder->registComponent(m_sComponentName, this);
}

void KaiComponentInfo::rename(KString sNewName) {
	m_pOwnFolder->checkProperName(sNewName);
	m_pOwnFolder->checkDuplicatedName(m_componentType, sNewName);
	m_sComponentName = sNewName;
}

void KaiComponentInfo::destory() {
	m_pOwnFolder->unregistComponent(this);
	delete this;
}

KJsonString KaiComponentInfo::getDescription(KString sPrefix) {
	KJsonString desc = sPrefix + m_sComponentName + "(" + std::to_string(m_nComponentId) + ") ";

	desc += ms_componentTypeName[(int)m_componentType];

	for (auto it = m_propDict.begin(); it != m_propDict.end(); it++) {
		desc += " " + it->first + ":" + it->second.desc();
	}

	return desc;
}

KJsonString KaiComponentInfo::m_dict_to_json(std::map<std::string, std::string> info) {
	KString prefix = "[";
	KJsonString desc;

	for (auto it = info.begin(); it != info.end(); it++) {
		desc += prefix + "\"" + m_jsonEscape(it->first) + "\":\"" + m_jsonEscape(it->second) + "\"";
		prefix = ";";
	}

	return desc + "]";
}

KString KaiComponentInfo::m_jsonEscape(KString str) {
	KString sEscapedStr;

	for (int n = 0; n < str.size(); n++) {
		int ch = str[n];
		if (ch == ' ') sEscapedStr += "\\b";
		else if (ch == '\n') sEscapedStr += "\\n";
		else if (ch == '\r') sEscapedStr += "\\r";
		else if (ch == '\t') sEscapedStr += "\\t";
		else if (ch == '\"') sEscapedStr += "\\\"";
		else if (ch == '\\') sEscapedStr += "\\\\";
		else sEscapedStr += ch;
	}

	return sEscapedStr;
}
