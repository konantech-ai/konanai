/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include <string>
#include <vector>
#include <map>

typedef long long int KInt;
typedef long long int KBool;
typedef float KFloat;
typedef std::string KString;

enum class Ken_object_type { none, session, library, model, model_instance, dataset, dataloader, network, layer, optimizer, expression, farray, narray, _exp_node };
enum class Ken_component_type { model, model_instance, dataset, dataloader, network, layer, optimizer, expression };
enum class Ken_actfunc { none = 0, relu = 1, sigmoid = 2, tanh = 3, leaky_relu = 4, gelu = 5, custom = 99 };
enum class Ken_inst_mode { none, standard, full, custom };
enum class Ken_test_level { core, brief, detail, full };
enum class Ken_value_type { none, kint, kfloat, string, list, dict, shape, object };
enum class Ken_arr_data_type { kint, kfloat };
enum class Ken_visualize_mode { train, visualize, predict };
enum class KBool3 { on, off, unknown };

class KaiObject {
public:
	KaiObject(Ken_object_type type) { m_nRefCount = 1; m_object_type = type; ms_debug_count++; }
	virtual ~KaiObject() { ms_debug_count--; }
	Ken_object_type get_type() { return m_object_type; }
	void destroy() { if (this && --m_nRefCount <= 0) delete this; }
	KaiObject* fetch() { m_nRefCount++; return this; }
	virtual KString desc() = 0;
	static KInt ms_debug_count;
protected:
	int m_nRefCount;
	Ken_object_type m_object_type;
	friend class KValue;
};

typedef KaiObject* KHObject;

typedef KHObject KHSession;
typedef KHObject KHLibrary;

typedef KHObject KHComponent;

typedef KHComponent KHModel;
typedef KHComponent KHModelInstance;
typedef KHComponent KHDataset;
typedef KHComponent KHNetwork;
typedef KHComponent KHLayer;
typedef KHComponent KHExpression;
typedef KHComponent KHOptimizer;

//typedef std::vector<KInt> KaiShape;

typedef KHObject KHArray;

typedef KHArray KHNArray;
typedef KHArray KHFArray;

typedef std::string KPathString;
typedef std::string KJsonString;

class KaiValue;
class KaiShape;

//typedef std::vector<KaiValue> KaiList;
//typedef std::map<KString, KaiValue> KaiDict;

typedef std::vector<KString> KStrList;
typedef std::vector<KString> KPathStrList;
typedef std::vector<KString> KJsonStrList;

typedef std::vector<KInt> KIntList;
