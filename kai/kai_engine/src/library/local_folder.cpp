/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "local_folder.h"
#include "local_library.h"
#include "../components/component.h"
#include "../utils/kutil.h"

LocalFolder::LocalFolder(KaiLocalLibrary* pLibrary, LocalFolder* pParent, KString sName) :m_pChildren(), m_pComponents() {
	m_pLibrary = pLibrary;
	m_pParent = pParent;
	m_sName = sName;
}

LocalFolder::~LocalFolder() {
	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		delete* it;
	}

	for (auto it = m_pComponents.begin(); it != m_pComponents.end(); it++) {
		delete* it;
	}
}

KaiSession* LocalFolder::getSession() {
	return m_pLibrary->getSession();
}

bool LocalFolder::hasNamedChild(KString sFolderName) {
	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		LocalFolder* pChild = *it;
		if (pChild->getName() == sFolderName) return true;
	}

	return false;
}

void LocalFolder::checkProperName(KString sComponentName) {
	if (sComponentName == "") throw KaiException(KERR_EMPTY_STRING_IS_NOT_A_PROPER_NAME, sComponentName);

	for (int n = 0; n < (int)sComponentName.size(); n++) {
		int ch = sComponentName[n];
		if (isalnum(ch)) continue;
		if (strchr("-_,.<>?:;!@#$%^&*()+", ch)) continue;
		throw KaiException(KERR_BAD_CHAR_USED_FOR_NAME, sComponentName.substr(n));
	}
}

void LocalFolder::checkDuplicatedName(enum class Ken_component_type componentType, KString sComponentName) {
	for (auto it = m_pComponents.begin(); it != m_pComponents.end(); it++) {
		KaiComponentInfo* pComponent = *it;
		if (pComponent->getName() == sComponentName && pComponent->getComponentType() == componentType) {
			throw KaiException(KERR_COMPONENT_ALREADY_EXIST, sComponentName);
		}
	}
}

KaiComponentInfo* LocalFolder::seekDuplicatedComponent(enum class Ken_component_type componentType, KString sComponentName) {
	for (auto it = m_pComponents.begin(); it != m_pComponents.end(); it++) {
		KaiComponentInfo* pComponent = *it;
		if (pComponent->getName() == sComponentName && pComponent->getComponentType() == componentType) {
			return pComponent;
		}
	}

	return NULL;
}

KaiComponentInfo* LocalFolder::addComponent(enum class Ken_component_type componentType, KString sComponentName, bool bThrowOnExist) {
	checkProperName(sComponentName);
	KaiComponentInfo* pNewComponent = seekDuplicatedComponent(componentType, sComponentName);
	if (pNewComponent == NULL) {
		pNewComponent = new KaiComponentInfo(this, sComponentName, componentType, m_pLibrary->issueComponentId());
		m_pComponents.push_back(pNewComponent);
	}
	return pNewComponent;
}

void LocalFolder::moveFolder(LocalFolder* pNewParent) {
	m_pParent->m_unregist(this);
	m_pParent = pNewParent;
	m_pParent->m_regist(this);
}

void LocalFolder::destroy() {
	m_pParent->m_unregist(this);
	delete this;
}

void LocalFolder::listAllComponents(KJsonStrList* pjlComponents, bool recursive) {
	m_listComponents(pjlComponents, recursive, true, Ken_component_type::model, ""); // Ken_component_type::model will be ignored
}

void LocalFolder::listComponents(KJsonStrList* pjlComponents, bool recursive, Ken_component_type componentType) {
	m_listComponents(pjlComponents, recursive, false, componentType, "");
}

void LocalFolder::listFolders(KPathStrList* pslSubFolders, bool recursive) {
	m_listFolders(pslSubFolders, recursive, "");
}

void LocalFolder::registComponent(KString sComponentName, KaiComponentInfo* pComponent) {
	checkProperName(sComponentName);
	checkDuplicatedName(pComponent->getComponentType(), sComponentName);

	pComponent->setOwnFolder(this);
	pComponent->setName(sComponentName);

	m_pComponents.push_back(pComponent);
}

void LocalFolder::unregistComponent(KaiComponentInfo* pComponent) {
	auto it = std::find(m_pComponents.begin(), m_pComponents.end(), pComponent);
	if (it == m_pComponents.end()) throw KaiException(KEER_NOT_EXISTING_COMPONENT, pComponent->getName());
	m_pComponents.erase(it);
}

KPathString LocalFolder::getFolderPath() {
	KPathString sParentPath;
	if (m_pParent) sParentPath = m_pParent->getFolderPath();
	return sParentPath + m_sName + "/";
}

LocalFolder* LocalFolder::seekFolder(KStrList slPieces) {
	LocalFolder* pFolder = this;

	for (auto it = slPieces.begin(); it != slPieces.end(); it++) {
		KString sPiece = *it;
		if (sPiece == "..") {
			pFolder = pFolder->getParent();
			if (pFolder == NULL) throw KaiException(KERR_ROOT_FOLDER_HAS_NO_PARENT);
		}
		else if (sPiece == "."); // do nothing
		else if (pFolder->m_hasChildFolder(sPiece)) pFolder = pFolder->m_getNamedChild(sPiece);
		else throw KaiException(KERR_SUBFOLDER_NOT_EXIST, pFolder->getFolderPath(), sPiece);
	}

	return pFolder;
}

LocalFolder* LocalFolder::createSubFolder(KString sNewFolder, bool bThrowOnExist) {
	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		if ((*it)->getName() == sNewFolder) {
			if (bThrowOnExist) throw KaiException(KERR_FOLDER_ALREADY_EXIST, sNewFolder);
			return *it;
		}
	}

	LocalFolder* pFolder = new LocalFolder(m_pLibrary, this, sNewFolder);
	m_pChildren.push_back(pFolder);
	return pFolder;
}

KaiComponentInfo* LocalFolder::getNamedComponent(KString sComponentName) { //, Ken_component_type componentType) {
	for (auto it = m_pComponents.begin(); it != m_pComponents.end(); it++) {
		KaiComponentInfo* pComponent = *it;
		if (pComponent->getName() == sComponentName) return pComponent;
	}
	throw KaiException(KERR_COMPONENT_NOT_FOUND, getFolderPath(), sComponentName);
}

bool LocalFolder::isAncestorOf(LocalFolder* pFolder) {
	pFolder = pFolder->getParent();

	while (pFolder != NULL) {
		if (pFolder == this) return true;
		pFolder = pFolder->getParent();
	}

	return false;
}

void LocalFolder::m_regist(LocalFolder* pFolder) {
	m_pChildren.push_back(pFolder);
}

void LocalFolder::m_unregist(LocalFolder* pFolder) {
	auto it = std::find(m_pChildren.begin(), m_pChildren.end(), pFolder);
	m_pChildren.erase(it);
}

void LocalFolder::m_listComponents(KJsonStrList* pjlComponents, bool recursive, bool bAllType, Ken_component_type componentType, KString sFolderPath) {
	for (auto it = m_pComponents.begin(); it != m_pComponents.end(); it++) {
		KaiComponentInfo* pComponent = *it;
		if (!bAllType && pComponent->getComponentType() != componentType) continue;
		KJsonString componentDesc = pComponent->getDescription(sFolderPath);
		pjlComponents->push_back(componentDesc);
	}

	if (!recursive) return;

	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		LocalFolder* pChild = *it;
		pChild->m_listComponents(pjlComponents, recursive, bAllType, componentType, pChild->getName() + "/");
	}
}

void LocalFolder::m_listFolders(KPathStrList* pslSubFolders, bool recursive, KString sFolderPath) {
	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		LocalFolder* pChild = *it;
		pslSubFolders->push_back(sFolderPath + pChild->getName());
		if (!recursive) continue;
		pChild->m_listFolders(pslSubFolders, recursive, pChild->getName() + "/");
	}
}

bool LocalFolder::m_hasChildFolder(KString sFolderName) {
	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		LocalFolder* pChild = *it;
		if (pChild->getName() == sFolderName) return true;
	}
	return false;
}

LocalFolder* LocalFolder::m_getNamedChild(KString sFolderName) {
	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		LocalFolder* pChild = *it;
		if (pChild->getName() == sFolderName) return pChild;
	}
	return NULL;
}

void LocalFolder::serialize(FILE* fid) {
	kutil.save_int(fid, m_pComponents.size());
	for (auto it = m_pComponents.begin(); it != m_pComponents.end(); it++) {
		KaiComponentInfo* pComponent = *it;
		kutil.save_int(fid, pComponent->getComponentId());
		kutil.save_str(fid, pComponent->getName());
		kutil.save_int(fid, (KInt) pComponent->getComponentType());
		pComponent->serialize(fid);
	}
	kutil.save_int(fid, m_pChildren.size());
	for (auto it = m_pChildren.begin(); it != m_pChildren.end(); it++) {
		LocalFolder* pChild = *it;
		kutil.save_str(fid, pChild->getName());
		pChild->serialize(fid);
	}
}

//void LocalFolder::unserialize(FILE* fid, KBindTable& bindTab, KComponentMap& componentMap) {
void LocalFolder::unserialize(FILE * fid) {
	KInt nComponentCount = kutil.read_int(fid);

	for (KInt n = 0; n < nComponentCount; n++) {
		int nComponentId = (int) kutil.read_int(fid);
		KString sComponentName = kutil.read_str(fid);
		Ken_component_type componentType = (Ken_component_type) kutil.read_int(fid);
		KaiComponentInfo* pComponent = new KaiComponentInfo(this, sComponentName, componentType, nComponentId);
		pComponent->unserialize(fid);
		m_pComponents.push_back(pComponent);
	}

	KInt nChildCount = kutil.read_int(fid);

	for (KInt n = 0; n < nChildCount; n++) {
		KString sFolderName = kutil.read_str(fid);
		LocalFolder* pFolder = new LocalFolder(m_pLibrary, this, sFolderName);
		pFolder->unserialize(fid);
		m_pChildren.push_back(pFolder);
	}
}