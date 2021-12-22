/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../session/kcommon.h"
#include "../include/kai_api.h"

KInt KaiValueCore::ms_debug_count =  0;
KInt KaiObject::ms_debug_count = 0;

KaiValueCore::~KaiValueCore() {
	switch (m_type) {
	case Ken_value_type::list:
		m_value.m_pList->destroy();
		break;
	case Ken_value_type::dict:
		m_value.m_pDict->destroy();
		break;
	case Ken_value_type::shape:
		m_value.m_pShape->destroy();
		break;
	case Ken_value_type::object:
		m_value.m_pObject->destroy();
		break;
	}
	ms_debug_count--;
}

KaiValue::KaiValue(KaiList lValue) {
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::list;
	m_core->m_value.m_pList = lValue.fetch();
}

KaiValue::KaiValue(KaiDict dValue) {
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::dict;
	m_core->m_value.m_pDict = dValue.fetch();
}

KaiValue::KaiValue(KaiShape sValue) {
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::shape;
	m_core->m_value.m_pShape = sValue.fetch();
}

KaiValue::KaiValue(KHObject hValue) {
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::object;
	m_core->m_value.m_pObject = hValue->fetch();
}

KaiValue& KaiValue::operator =(KaiList lValue) {
	m_core->destroy();
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::list;
	m_core->m_value.m_pList = lValue.fetch();
	return *this;
}

KaiValue& KaiValue::operator =(KaiDict dValue) {
	m_core->destroy();
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::dict;
	m_core->m_value.m_pDict = dValue.fetch();
	return *this;
}

KaiValue& KaiValue::operator =(KaiShape sValue) {
	m_core->destroy();
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::shape;
	m_core->m_value.m_pShape = sValue.fetch();
	return *this;
}

KaiValue& KaiValue::operator =(KHObject hValue) {
	m_core->destroy();
	m_core = new KaiValueCore();
	m_core->m_type = Ken_value_type::object;
	m_core->m_value.m_pObject = hValue->fetch();
	return *this;
}

bool KaiValue::operator ==(const KaiValue& vValue) const {
	Ken_value_type vtype = vValue.type();
	if (type() == Ken_value_type::kint) {
		if (vtype == Ken_value_type::kint) return m_core->m_value.m_int == vValue.m_core->m_value.m_int;
	}
	switch (m_core->m_type) {
	case Ken_value_type::kint:
		if (vtype == Ken_value_type::kint) return m_core->m_value.m_int == (KInt)vValue;
		if (vtype == Ken_value_type::kfloat) return (KFloat)m_core->m_value.m_int == (KFloat)vValue;
		break;
	case Ken_value_type::kfloat:
		if (vtype == Ken_value_type::kint) return m_core->m_value.m_float == (KFloat)vValue;
		if (vtype == Ken_value_type::kfloat) return m_core->m_value.m_float == (KFloat)vValue;
		break;
	case Ken_value_type::none:
		if (vtype == Ken_value_type::none) return true;
		break;
	case Ken_value_type::dict:
		if (vtype == Ken_value_type::dict) return m_core->m_value.m_pDict == vValue.m_core->m_value.m_pDict;
		break;
	case Ken_value_type::list:
		if (vtype == Ken_value_type::list) return m_core->m_value.m_pList == vValue.m_core->m_value.m_pList;
		break;
	case Ken_value_type::shape:
		if (vtype == Ken_value_type::shape) return m_core->m_value.m_pShape == vValue.m_core->m_value.m_pShape;
		break;
	case Ken_value_type::object:
		if (vtype == Ken_value_type::object) return m_core->m_value.m_pObject == vValue.m_core->m_value.m_pObject;
		break;
	}
	return false;
}

bool KaiValue::operator ==(const bool bValue) const {
	if (type() == Ken_value_type::kint) return m_core->m_value.m_int == (KInt)bValue;
	if (type() == Ken_value_type::kfloat) return m_core->m_value.m_float == (KFloat)bValue;
	return false;
}

bool KaiValue::operator ==(int nValue) const {
	if (type() == Ken_value_type::kint) return m_core->m_value.m_int == (KInt)nValue;
	if (type() == Ken_value_type::kfloat) return m_core->m_value.m_float == (KFloat)nValue;
	return false;
}

bool KaiValue::operator ==(KInt nValue) const {
	if (type() == Ken_value_type::kint) return m_core->m_value.m_int == nValue;
	if (type() == Ken_value_type::kfloat) return m_core->m_value.m_float == (KFloat)nValue;
	return false;
}

bool KaiValue::operator ==(KString sValue) const {
	if (type() != Ken_value_type::string) return false;
	return m_core->m_string == sValue;
}

bool KaiValue::operator ==(const char* sValue) const {
	if (type() != Ken_value_type::string) return false;
	return m_core->m_string == (KString) sValue;
}

bool KaiValue::operator ==(KFloat fValue) const {
	if (type() == Ken_value_type::kfloat) return m_core->m_value.m_float == fValue;
	if (type() == Ken_value_type::kint) return m_core->m_value.m_int == fValue;
	return false;
}

bool KaiValue::operator ==(KaiList lValue) const {
	if (type() != Ken_value_type::list) return false;
	return m_core->m_value.m_pList == lValue.m_core; // 내용이 아닌 포인터 수준의 비교임, 내용으로 들어가야 할까?
}

bool KaiValue::operator ==(KaiDict dValue) const {
	if (type() != Ken_value_type::dict) return false;
	return m_core->m_value.m_pDict == dValue.m_core; // 내용이 아닌 포인터 수준의 비교임, 내용으로 들어가야 할까?
}

bool KaiValue::operator ==(KaiShape sValue) const {
	if (type() != Ken_value_type::shape) return false;
	return m_core->m_value.m_pShape == sValue.m_core; // 내용이 아닌 포인터 수준의 비교임, 내용으로 들어가야 할까?
}

bool KaiValue::operator ==(KHObject hValue) const {
	if (type() != Ken_value_type::object) return false;
	return m_core->m_value.m_pObject == hValue;
}

KaiValue::operator int() const {
	if (type() == Ken_value_type::kint) return (int)m_core->m_value.m_int;
	if (type() == Ken_value_type::kfloat) return (int)m_core->m_value.m_float;
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KaiValue::operator KInt() const {
	if (type() == Ken_value_type::kint) return m_core->m_value.m_int;
	if (type() == Ken_value_type::kfloat) return (KInt)m_core->m_value.m_float;
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KaiValue::operator KString() const {
	if (type() == Ken_value_type::string) return m_core->m_string;
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KaiValue::operator KFloat() const {
	if (type() == Ken_value_type::kint) return (KFloat)m_core->m_value.m_int;
	if (type() == Ken_value_type::kfloat) return m_core->m_value.m_float;
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KaiValue::operator KaiList() {
	if (type() == Ken_value_type::list) return KaiList(m_core->m_value.m_pList);
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KaiValue::operator KaiDict() {
	if (type() == Ken_value_type::dict) return KaiDict(m_core->m_value.m_pDict);
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KaiValue::operator KaiShape() {
	if (type() == Ken_value_type::shape) return KaiShape(m_core->m_value.m_pShape);
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KaiValue::operator KHObject() const {
	if (type() == Ken_value_type::object) return m_core->m_value.m_pObject;
	THROW(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
}

KBool KaiValue::is_farray() {
	if (m_core->m_type != Ken_value_type::object) return false;
	return m_core->m_value.m_pObject->get_type() == Ken_object_type::farray;
}

KBool KaiValue::is_narray() {
	if (m_core->m_type != Ken_value_type::object) return false;
	return m_core->m_value.m_pObject->get_type() == Ken_object_type::narray;
}

static KString encode_esc(KString str) {
	if (str.find('\'') == string::npos && str.find('\\') == string::npos) return str;

	size_t pos = str.find('\\');
	while (pos != string::npos) {
		str = str.substr(0, pos - 1) + "\\" + str.substr(pos);
		pos = str.find('\\', pos + 2);
	}

	pos = str.find('\'');
	while (pos != string::npos) {
		str = str.substr(0, pos - 1) + "\\" + str.substr(pos);
		pos = str.find('\'', pos + 2);
	}

	return str;
}

static KString list_desc(KaiListCore* pList) {
}

KString KaiValue::desc() const {
	switch (type()) {
	case Ken_value_type::none:
		return "None";
	case Ken_value_type::kint:
		return to_string(m_core->m_value.m_int);
	case Ken_value_type::kfloat:
		return to_string(m_core->m_value.m_float);
	case Ken_value_type::string:
		return "'" + encode_esc(m_core->m_string) + "'";
	case Ken_value_type::list:
		{
			KaiListCore* pList = m_core->m_value.m_pList;
			KString desc, delimeter = "[";
			KInt size = pList->m_list.size();
			if (size == 0) desc = delimeter;
			for (KInt n = 0; n < size; n++) { desc += delimeter; delimeter = ",";  desc += pList->m_list[n].desc(); }
			return desc + "]";
		}
	case Ken_value_type::dict:
		{
			std::map<KString, KaiValue>& dict = m_core->m_value.m_pDict->m_dict;
			KString desc, delimeter = "{";
			if (dict.size() == 0) desc = delimeter;
			for (KaiDictIter it = dict.begin(); it != dict.end(); it++) { desc += delimeter; delimeter = ",";  desc += it->first + ":" + it->second.desc(); }
			return desc + "}";
		}
	case Ken_value_type::shape:
		{
			KaiShapeCore* pShape = m_core->m_value.m_pShape;
			KString desc, delimeter = "<";
			KInt size = pShape->m_shape.size();
			if (size == 0) desc = delimeter;
			for (KInt n = 0; n < size; n++) { desc += delimeter; delimeter = ",";  desc += to_string(pShape->m_shape[n]); }
			return desc + ">";
		}
	case Ken_value_type::object:
		return m_core->m_value.m_pObject->desc();
	}
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "unknown kvalue type");
}

KBool KaiList::find_string(KString sValue) const {
	for (auto& it : m_core->m_list) {
		if (it.type() == Ken_value_type::string && (KString)it == sValue) return true;
	}
	return false;
}
