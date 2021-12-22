/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../session/session.h"
#include "../components/component.h"

class KaiLibrary {
public:
	KaiLibrary(KaiSession* pSession, KString sLibName);
	virtual ~KaiLibrary();

	static KaiLibrary* HandleToPointer(KHObject hObject, KaiSession* pSession);

	KaiSession* getSession() { return m_pOwnSession; }
	KString getName() { return m_sLibName; }

	virtual KString GetVersion() = 0;

	virtual KPathString getCurrPath() = 0;
	
	virtual void changePassword(KString sOldPassword, KString sNewPassword) = 0;

	virtual void setCurrPath(KPathString KPathString) = 0;
	virtual void createFolder(KPathString sNewPath, bool bThrowOnExist) = 0;
	virtual void renameFolder(KPathString sOldPath, KString sNewName) = 0;
	virtual void moveFolder(KPathString sOldPath, KPathString sNewPath) = 0;
	virtual void deleteFolder(KPathString sFolderPath) = 0;
	virtual void list(KJsonStrList* pjlComponents, KPathString sPath, bool recursive) = 0;
	virtual void listSubFolders(KPathStrList* pslSubFolders, KPathString sPath, bool recursive) = 0;

	//virtual KaiComponent* downloadComponent(Ken_component_type componentType, KPathString sComponentPath) = 0;
	virtual KaiComponentInfo* seekComponent(KPathString sPath) = 0;
	virtual KaiComponentInfo* createComponent(KPathString sPath, Ken_component_type componentType) = 0;

	virtual void listComponents(Ken_component_type componentType, KStrList* pslComponentNames, KPathString sPath, bool recursive) = 0;
	virtual void setProperty(KPathString sComponentlPath, KPathString sProperty, KString sValue) = 0;
	virtual void moveComponent(KPathString sCurComponentPath, KPathString sDestFolder) = 0;
	virtual void renameComponent(KPathString sCurComponentPath, KString sNewName) = 0;
	virtual void deleteComponent(KPathString sComponentlPath) = 0;

	virtual void registComponent(Ken_component_type componentType, KPathString sNewComponentPath, KaiComponent* pComponent) = 0;
	virtual void updateComponent(Ken_component_type componentType, KPathString sComponentPath, KaiComponent* pComponent) = 0;

protected:
	KaiSession* m_pOwnSession;
	KString m_sLibName;

	int m_checkCode;

	static int ms_checkCodeLocal;
	static int ms_checkCodePublic;
};
