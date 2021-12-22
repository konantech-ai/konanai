/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../core/common.h"
#include "../int_plugin/internal_plugin.h"
#include "../int_plugin/optimizer.cuh"
#include "../../src/include/kai_api.h"

/*
IPLUGREG::IPLUGREG(int type, string name, IPL_SELECT_FUNC func) {
	InternalPluginManager::regist(type, name, func);
}

IPLUGREG::~IPLUGREG() {
}
*/

//const char* InternalPluginManager::ms_type_names[] = { "database", "optimizer", "pipeline", "multi_board" };

InternalPluginManager int_plugin_man; 

InternalPluginManager::InternalPluginManager() {
	vector<string> database_noms;
	vector<string> optimizer_noms;
	vector<string> pipeline_noms;
	vector<string> multi_board_noms;

	optimizer_noms.push_back("sgd");
	optimizer_noms.push_back("adam");

	if (database_noms.size() > 0) m_components["database"] = database_noms;
	if (optimizer_noms.size() > 0) m_components["optimizer"] = optimizer_noms;
	if (pipeline_noms.size() > 0) m_components["pipeline"] = pipeline_noms;
	if (multi_board_noms.size() > 0) m_components["multi_board"] = multi_board_noms;

	Optimizer::set_plugin_component("adam");
}

InternalPluginManager::~InternalPluginManager() {
}

/*
void InternalPluginManager::regist(int type, string name, IPL_SELECT_FUNC func) {
	InternalPluginMemo memo;

	memo.m_type = type;
	memo.m_name = name;
	memo.m_func = func;

	ms_info.push_back(memo);

	std::vector<int>::iterator it;
	it = find(ms_types.begin(), ms_types.end(), type);

	if (it == ms_types.end()) {
		ms_types.push_back(type);
	}
}

vector<string> InternalPluginManager::fetch_type(int type) {
	vector<string> names;

	for (int n = 0; n < (int)ms_info.size(); n++) {
		if (ms_info[n].m_type == type) {
			names.push_back(ms_info[n].m_name);
		}
	}

	return names;
}
*/

void InternalPluginManager::getTypeNames(KStrList* pTypeNames) {
	pTypeNames->clear();

	for (auto& it : m_components) {
		pTypeNames->push_back(it.first);
	}
}

void InternalPluginManager::getNomNames(string type_name, KStrList* pNomNames) {
	*pNomNames = m_components[type_name];
}

/*
// depreciated
vector<string> InternalPluginManager::getTypeNames() {
	vector<string> names;

	for (auto& it : m_components) {
		names.push_back(it.first);
	}

	return names;
}

// depreciated
vector<string> InternalPluginManager::getTypeNoms(string type_name) {
	return m_components[type_name];
}
*/

string InternalPluginManager::introduce(string type_name) {
	if (type_name == "optimizer") return Optimizer::introduce_curr_instance();
	throw KaiException(KERR_INPROPER_SELECT_TYPE_NAME);
}

void InternalPluginManager::set_plugin_component(string type_name, string component_name) {
	if (type_name == "optimizer") Optimizer::set_plugin_component(component_name);
	else throw KaiException(KERR_INPROPER_SELECT_TYPE_NAME);
}
