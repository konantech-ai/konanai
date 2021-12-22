/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "library.h"
#include "local_folder.h"

class KaiSession;

struct _InitModelInfo {
	const char* modelName;
	bool isStandatd;
	const char* modelPath;
	const char* dataset;
	const char* dataloader;
	const char* network;
	const char* expression;
	const char* optimizer;
};

struct _InitComponentInfo {
	const char* componentName;
	const char* componentProps;
};

struct _BindInfo {
	int subject, object;
	_BindInfo(int subj, int obj) { subject = subj; object = obj; }
};

struct BindInfo {
	KaiComponentInfo* m_pSubject;
	KaiComponentInfo* m_pObject;
};

typedef std::vector<_BindInfo> _KBindTable;
typedef std::vector<BindInfo> KBindTable;

class KaiLocalLibrary : public KaiLibrary {
public:
	KaiLocalLibrary(KaiSession* pSession, KString sLibName, KString sPassword);
	KaiLocalLibrary(KaiSession* pSession, KString sLibName, KString sPassword, Ken_inst_mode enumInstallMode, KStrList slModelNames);
	virtual ~KaiLocalLibrary();

	static KaiLocalLibrary* HandleToPointer(KHObject hObject, KaiSession* pSession);

	int issueComponentId();

	static void GetInstallModels(KStrList* pslModuleNames);

	void setName(KString sNewLibName);
	void installModels(KStrList slModels);
	void save();
	void destory();

	KString GetVersion();
	void close(bool bSave);

	KPathString getCurrPath();

	void changePassword(KString sOldPassword, KString sNewPassword);

	void setCurrPath(KPathString sCurrPath);
	void createFolder(KPathString sNewPath, bool bThrowOnExist);
	void renameFolder(KPathString sOldPath, KString sNewName);
	void moveFolder(KPathString sOldPath, KPathString sNewPath);
	void deleteFolder(KPathString sFolderPath);
	void list(KJsonStrList* pjlComponents, KPathString sPath, bool recursive);
	void listSubFolders(KPathStrList* pslSubFolders, KPathString sPath, bool recursive);

	//KaiComponent* downloadComponent(Ken_component_type componentType, KPathString sComponentPath); // ÇÊ¿ä?

	KaiComponentInfo* seekComponent(KPathString sPath);
	KaiComponentInfo* createComponent(KPathString sPath, Ken_component_type componentType);

	void listComponents(Ken_component_type componentType, KJsonStrList* pslComponentNames, KPathString sPath, bool recursive);
	void setProperty(KPathString sComponentlPath, KPathString sProperty, KString sValue);
	void moveComponent(KPathString sCurComponentPath, KPathString sDestFolder);
	void renameComponent(KPathString sCurComponentPath, KString sNewName);
	void deleteComponent(KPathString sComponentath);

	void registComponent(Ken_component_type componentType, KPathString sNewComponentPath, KaiComponent* pComponent);
	void updateComponent(Ken_component_type componentType, KPathString sComponentPath, KaiComponent* pComponent);

protected:
	void m_load_library_file();
	void m_create_library_file(Ken_inst_mode inst_mode, KStrList slModelNames);
	void m_save_library_file(bool bMandatory);
	void m_remove_old_library_file(KString sOldLibName);
	void m_checkModelNames(KStrList slModels);
	void m_installModels(KStrList slModels);
	void m_installModel(int nth);

	LocalFolder* m_seekFolder(KPathString sPath, KString* pLastPiece=NULL);
	LocalFolder* m_seekOrCreateFolder(KPathString sPath);

	KStrList m_splitPathPieces(KPathString sFolderPath, KString* pLastPiece, bool*pbIsAbsolutePath);

	bool ms_isValidLibName(KString sLibName);
	bool ms_isValidPassword(KString sPassword);

protected:
	KString m_sPassword;
	KString m_sVersion;

	bool m_bTouched;
	bool m_bDestroyed;

	LocalFolder* m_pRootFolder;
	LocalFolder* m_pCurrFolder;

	int m_nNextComponentId;

	KBindTable m_bindTable;

	static int ms_checkCodeSave;

	static _InitModelInfo ms_initModuleNames[];
	static _InitComponentInfo ms_iniComponentInfo[];

	//static std::map<KString, std::map<KString, KString>> ms_initComponentInfoMap;
	static std::map<KString, KString> ms_initComponentInfoMap;
};
