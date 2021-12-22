/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/common.h"

#define IPLUGIN(type,name,func) IPLUGREG name##_##type(type, #name, func)

#define IPL_DATABASE		1
#define IPL_OPTIMIZER		2
#define IPL_PIPELINE		3
#define IPL_MULTI_BOARD		4

typedef void (*IPL_SELECT_FUNC)(Dict initArgs);

/*
class IPLUGREG {
public:
	IPLUGREG(int type, string name, IPL_SELECT_FUNC func);
	virtual ~IPLUGREG();
};
*/

class InternalPluginComponent {
public:
	InternalPluginComponent() {}
	virtual ~InternalPluginComponent() {}
};

class InternalPluginMemo {
protected:
	friend class InternalPluginManager;
	int m_type;
	string m_name;
	IPL_SELECT_FUNC m_func;
};

class InternalPluginManager {
public:
	InternalPluginManager();
	virtual ~InternalPluginManager();

	void getTypeNames(KStrList* pTypeNames);
	void getNomNames(string type_name, KStrList* pNomNames);

	//vector<string> getTypeNames(); // depreciated
	//vector<string> getTypeNoms(string type_name); // depreciated

	string introduce(string type_name);

	void set_plugin_component(string type_name, string component_name);

protected:
	map<string, vector<string>> m_components;
};

extern InternalPluginManager int_plugin_man;