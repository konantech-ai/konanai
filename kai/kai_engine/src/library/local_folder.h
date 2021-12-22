/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/session.h"
#include "../components/component_info.h"

class LocalFolder {
public:
	LocalFolder(KaiLocalLibrary* pLibrary, LocalFolder* pParent, KString sName);
	virtual ~LocalFolder();

	KaiSession* getSession();
	KString getName() { return m_sName; }
	LocalFolder* getParent() { return m_pParent; }

	void serialize(FILE* fid);
	void unserialize(FILE* fid);

	void rename(KString sNewName) {m_sName = sNewName; }
	void moveFolder(LocalFolder* pNewParent);
	void destroy();
	void listAllComponents(KJsonStrList* pjlComponents, bool recursive);
	void listComponents(KJsonStrList* pjlComponents, bool recursive, Ken_component_type componentType);
	void listFolders(KPathStrList* pslSubFolders, bool recursive);
	void registComponent(KString sComponentName, KaiComponentInfo* pComponent);
	void unregistComponent(KaiComponentInfo* pComponent);

	KPathString getFolderPath();

	LocalFolder* seekFolder(KStrList slPieces);
	LocalFolder* createSubFolder(KString sFolderName, bool bThrowOnExist);

	KaiComponentInfo* addComponent(enum class Ken_component_type componentType, KString name, bool bThrowOnExist);
	KaiComponentInfo* getNamedComponent(KString sComponentName); // , Ken_component_type componentType);
	KaiComponentInfo* seekDuplicatedComponent(enum class Ken_component_type componentType, KString sComponentName); // , Ken_component_type componentType);

	void checkProperName(KString sName);
	void checkDuplicatedName(enum class Ken_component_type componentType, KString sName);

	bool isAncestorOf(LocalFolder* pFolder);
	bool hasNamedChild(KString sFolderName);

protected:
	void m_regist(LocalFolder* pFolder);
	void m_unregist(LocalFolder* pFolder);

	void m_listComponents(KJsonStrList* pjlComponents, bool recursive, bool bAllType, Ken_component_type componentType, KString sFolderPath);
	void m_listFolders(KPathStrList* pslSubFolders, bool recursive, KString sFolderPath);

	bool m_hasChildFolder(KString sFolderName);

	LocalFolder* m_getNamedChild(KString sFolderName);

protected:
	KaiLocalLibrary* m_pLibrary;
	LocalFolder* m_pParent;
	KString m_sName;
	std::vector<LocalFolder*> m_pChildren;
	std::vector<KaiComponentInfo*> m_pComponents;
};
