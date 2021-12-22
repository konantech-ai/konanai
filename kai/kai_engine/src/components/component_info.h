/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "../session/session.h"

class LocalFolder;
class KaiComponentInfo;
class KaiComponent;

typedef std::map<int, KaiComponentInfo*> KComponentMap;

class KaiComponentInfo {
public:
	KaiComponentInfo(LocalFolder* pFolder, KString sName, Ken_component_type componentType, int nComponentId);
	virtual ~KaiComponentInfo();

	//static KaiComponent* Create(LocalFolder* pFolder, KString sName, Ken_component_type componentType);

	int getComponentId() { return m_nComponentId; }
	//void setComponentId(int nComponentId) { m_nComponentId = nComponentId; }

	KString getName() { return m_sComponentName; }
	Ken_component_type getComponentType() { return m_componentType; }
	KaiSession* getSession();

	void setInitialProps(KString props);
	void setProperty(KString key, KString value);
	void setProperty(KString key, int value);

	void serialize(FILE* fid);
	void unserialize(FILE* fid);

	void setOwnFolder(LocalFolder* pFolder) { m_pOwnFolder = pFolder; }
	void setName(KString sComponentName) { m_sComponentName = m_sComponentName; }

	void typeCheck(Ken_component_type componentType);
	void moveFolder(LocalFolder* pDestFolder);
	void rename(KString sNewName);

	void update(KaiComponent* pComponent);
	void destory();

	KJsonString getDescription(KString sPrefix);

	//void bindRelation(KaiComponentInfo* pObjectComponent, int relation);

protected:
	KJsonString m_dict_to_json(std::map<std::string, std::string> info);
	KString m_jsonEscape(KString str);

protected:
	friend class KaiComponent;

	int m_nComponentId;

	LocalFolder* m_pOwnFolder;
	KString m_sComponentName;
	Ken_component_type m_componentType;

	KaiDict m_propDict;

	int m_checkCode;

	static KString ms_componentTypeName[];
};
