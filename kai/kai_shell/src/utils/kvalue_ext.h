/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

/*
#include "../common.h"

class KValueException {
public:
	KValueException(int nErrCode) { m_nErrCode = nErrCode; }
	~KValueException() { }
	int m_nErrCode;
};

class KaiList;
class KaiDict;
class KaiShape;

class KaiListCore;
class KaiDictCore;
class KaiShapeCore;

class KaiValueCore {
protected:
	int m_nRefCnt;
	Ken_value_type m_type;
	union value_union {
		KInt m_int;
		KFloat m_float;
		KaiListCore* m_pList;
		KaiDictCore* m_pDict;
		KaiShapeCore* m_pShape;
		KaiObject* m_pObject;
	} m_value;
	KString m_string;

	friend class KaiValue;

protected:
	KaiValueCore() {
		m_type = Ken_value_type::none;
		m_value.m_int = 0;
		m_nRefCnt = 1;
	}
	~KaiValueCore();
	void destroy() { if (--m_nRefCnt <= 0) delete this; }
};

class KaiValue {
public:
	KaiValue() { m_core = new KaiValueCore(); }
	KaiValue(const KaiValue& src) { m_core = src.m_core; m_core->m_nRefCnt++; }
	KaiValue(int nValue) { m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::kint; m_core->m_value.m_int = (KInt)nValue; }
	KaiValue(KInt nValue) { m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::kint; m_core->m_value.m_int = nValue; }
	KaiValue(KString sValue) { m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::string; m_core->m_string = sValue; }
	KaiValue(const char* sValue) { m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::string; m_core->m_string = sValue; }
	KaiValue(KFloat fValue) { m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::kfloat; m_core->m_value.m_float = fValue; }
	KaiValue(KaiList& lValue);
	KaiValue(KaiDict& dValue);
	KaiValue(KaiShape& sValue);
	KaiValue(KHObject hValue);

	~KaiValue() { m_core->destroy(); }

	KaiValue& operator =(const KaiValue& src) {
		if (&src != this) {
			if (--m_core->m_nRefCnt <= 0) delete m_core;
			m_core = src.m_core;
			m_core->m_nRefCnt++;
		}
		return *this;
	}

	KaiValue& operator =(int nValue) { m_core->destroy(); m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::kint; m_core->m_value.m_int = (KInt)nValue; return *this; }
	KaiValue& operator =(KInt nValue) { m_core->destroy(); m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::kint; m_core->m_value.m_int = nValue; return *this; }
	KaiValue& operator =(KString sValue) { m_core->destroy(); m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::string; m_core->m_string = sValue; return *this; }
	KaiValue& operator =(const char* sValue) { m_core->destroy(); m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::string; m_core->m_string = sValue; return *this; }
	KaiValue& operator =(KFloat fValue) { m_core->destroy(); m_core = new KaiValueCore(); m_core->m_type = Ken_value_type::kfloat; m_core->m_value.m_float = fValue; return *this; }
	KaiValue& operator =(KaiList& lValue);
	KaiValue& operator =(KaiDict& dValue);
	KaiValue& operator =(KaiShape& sValue);
	KaiValue& operator =(KHObject hValue);

	bool operator ==(const KaiValue& vValue) const;
	bool operator ==(const bool sValue) const;
	bool operator ==(int nValue) const;
	bool operator ==(KInt nValue) const;
	bool operator ==(KString sValue) const;
	bool operator ==(const char* sValue) const;
	bool operator ==(KFloat fValue) const;
	bool operator ==(KaiList lValue) const;
	bool operator ==(KaiDict dValue) const;
	bool operator ==(KaiShape sValue) const;
	bool operator ==(KHObject hValue) const;

	Ken_value_type type() const { return m_core->m_type; }

	operator int() const;
	operator KInt() const;
	operator KString() const;
	operator KFloat() const;
	operator KaiList();
	operator KaiDict();
	operator KaiShape();
	operator KHObject() const;

	KString desc() const;

protected:
	KaiValueCore* m_core;
};

typedef std::map<std::string, KaiValue>::iterator KaiDictIter;
typedef std::vector<KaiValue>::iterator KaiListIter;
typedef std::initializer_list<KaiValue>::iterator _initIt;
typedef std::initializer_list<std::initializer_list<KaiValue>>::iterator _initIt2;

class KaiListCore {
	int m_nRefCnt;
	std::vector<KaiValue> m_list;
	void destroy() { if (--m_nRefCnt <= 0) delete this; }
	friend class KaiList;
	friend class KaiValue;
	friend class KaiValueCore;
};

class KaiList {
public:
	KaiList() { m_core = new KaiListCore(); m_core->m_nRefCnt = 1; }
	KaiList(const KaiList& src) { m_core = src.m_core; m_core->m_nRefCnt++; }
	KaiList(KaiListCore* core) { m_core = core; m_core->m_nRefCnt++; }
	KaiList(std::initializer_list<KaiValue> list) { m_core = new KaiListCore(); m_core->m_nRefCnt = 1; for (KaiValue ax : list) m_core->m_list.push_back(ax); }
	~KaiList() { if (--m_core->m_nRefCnt <= 0) delete m_core; }

	KaiList& operator =(const KaiList& src) {
		if (&src != this) {
			if (--m_core->m_nRefCnt <= 0) delete m_core;
			m_core = src.m_core;
			m_core->m_nRefCnt++;
		}
		return *this;
	}

	KaiListCore* fetch() { m_core->m_nRefCnt++; return m_core; }

	KInt size() const { return (KInt)m_core->m_list.size(); }

	void clear() { m_core->m_list.clear(); }
	void push_back(KaiValue value) { m_core->m_list.push_back(value); }
	void erase(KaiListIter it) { m_core->m_list.erase(it); }

	KaiValue& operator [](KInt nIndex) { return m_core->m_list[nIndex]; }
	KaiValue operator [](KInt nIndex) const { return m_core->m_list[nIndex]; }

	KaiListIter begin() const { return m_core->m_list.begin(); }
	KaiListIter end() const { return m_core->m_list.end(); }

	KaiListIter find(KaiValue value) const { return std::find(m_core->m_list.begin(), m_core->m_list.end(), value); }

	KString desc() const {
		KString desc, delimeter = "[";
		for (KInt n = 0; n < size(); n++) { desc += delimeter; delimeter = ",";  desc += m_core->m_list[n].desc(); }
		return desc + "]";
	}

protected:
	KaiListCore* m_core;
	friend class KaiValue;
};

class KaiDictCore {
	int m_nRefCnt;
	std::map<KString, KaiValue> m_dict;
	void destroy() { if (--m_nRefCnt <= 0) delete this; }
	friend class KaiDict;
	friend class KaiValue;
	friend class KaiValueCore;
};

class KaiDict {
public:
	KaiDict() { m_core = new KaiDictCore(); m_core->m_nRefCnt = 1; }
	KaiDict(const KaiDict& src) { m_core = src.m_core; m_core->m_nRefCnt++; }
	KaiDict(KaiDictCore* core) { m_core = core; m_core->m_nRefCnt++; }
	KaiDict(std::initializer_list<KaiValue> list) {
		m_core = new KaiDictCore();
		m_core->m_nRefCnt = 1;
		for (_initIt it = list.begin(); it != list.end(); ) {
			KString k = (KString)(KaiValue)*it; it++;
			KaiValue v = *it; it++;
			m_core->m_dict[k] = v;
		}
	}
	KaiDict(std::initializer_list<std::initializer_list<KaiValue>> list) {
		m_core = new KaiDictCore();
		m_core->m_nRefCnt = 1;
		for (_initIt2 it2 = list.begin(); it2 != list.end(); ) {
			std::initializer_list<KaiValue> term = *it2;
			_initIt it = term.begin();
			KString k = (KString)(KaiValue)*it; it++;
			KaiValue v = *it;
			m_core->m_dict[k] = v;
		}
	}
	~KaiDict() { if (--m_core->m_nRefCnt <= 0) delete m_core; }

	KaiDict& operator =(const KaiDict& src) {
		if (&src != this) {
			if (--m_core->m_nRefCnt <= 0) delete m_core;
			m_core = src.m_core;
			m_core->m_nRefCnt++;
		}
		return *this;
	}

	KaiDictCore* fetch() { m_core->m_nRefCnt++; return m_core; }

	KInt size() const { return (KInt)m_core->m_dict.size(); }

	void clear() { m_core->m_dict.clear(); }
	void erase(KString sKey) { m_core->m_dict.erase(sKey); }

	KaiValue& operator [](KString sKey) { return m_core->m_dict[sKey]; }
	KaiValue operator [](KString sKey) const { return m_core->m_dict[sKey]; }

	KaiDictIter begin() const { return m_core->m_dict.begin(); }
	KaiDictIter end() const { return m_core->m_dict.end(); }

	KaiDictIter find(KString sKey) const { return m_core->m_dict.find(sKey); }

	KString desc() const {
		KString desc, delimeter = "{";
		for (KaiDictIter it = begin(); it != end(); it++) { desc += delimeter; delimeter = ",";  desc += it->first + ":" + it->second.desc(); }
		return desc + "}";
	}

protected:
	KaiDictCore* m_core;
	friend class KaiValue;
};

class KaiShapeCore {
	int m_nRefCnt;
	std::vector<KInt> m_shape;
	void destroy() { if (--m_nRefCnt <= 0) delete this; }
	friend class KaiShape;
	friend class KaiValue;
	friend class KaiValueCore;
};

class KaiShape {
public:
	KaiShape() { m_core = new KaiShapeCore(); m_core->m_nRefCnt = 1; }
	KaiShape(const KaiShape& src) { m_core = src.m_core; m_core->m_nRefCnt++; }
	KaiShape(KaiShapeCore* core) { m_core = core; m_core->m_nRefCnt++; }
	KaiShape(KInt dim, KInt* ax_size) { m_core = new KaiShapeCore(); m_core->m_nRefCnt = 1; for (KInt n = 0; n < dim; n++) m_core->m_shape.push_back(ax_size[n]); }
	KaiShape(std::initializer_list<KInt> list) { m_core = new KaiShapeCore(); m_core->m_nRefCnt = 1; for (KInt ax : list) m_core->m_shape.push_back(ax); }
	~KaiShape() { if (--m_core->m_nRefCnt <= 0) delete m_core; }

	KaiShape& operator =(const KaiShape& src) {
		if (&src != this) {
			if (--m_core->m_nRefCnt <= 0) delete m_core;
			m_core = src.m_core;
			m_core->m_nRefCnt++;
		}
		return *this;
	}

	KaiShapeCore* fetch() { m_core->m_nRefCnt++; return m_core; }

	KInt size() const { return (KInt)m_core->m_shape.size(); }
	KInt total_size() const { if (size() == 0) return 0; KInt prod = 1; for (KInt ax : m_core->m_shape) prod *= ax; return prod; }

	KaiShape copy() { KaiShape shape; for (KInt ax : m_core->m_shape) shape.m_core->m_shape.push_back(ax); return shape; }
	KaiShape replace_end(KInt axn) { if (size() <= 0) THROW(KERR_REPLACE_REQUEST_ON_EMPTY_SHAPE); KaiShape shape = copy(); shape.m_core->m_shape[size() - 1] = axn; return shape; }
	KaiShape insert_head(KInt axn) { KaiShape shape; shape.m_core->m_shape.push_back(axn); for (KInt ax : m_core->m_shape) shape.m_core->m_shape.push_back(ax); return shape; }
	KaiShape replace_tail(KInt tlen, KaiShape newShape) {
		KInt sz1 = size(), sz2 = newShape.size();
		if (sz1 < tlen) THROW(KERR_REPLACE_REQUEST_ON_EMPTY_SHAPE);
		KaiShape shape;
		for (KInt n = 0; n < sz1 - tlen; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		for (KInt n = 0; n < sz2; n++) shape.m_core->m_shape.push_back(newShape[n]);
		return shape;
	}

	KInt& operator [](KInt ax) { if (ax < 0 || ax >= size()) THROW(KERR_SHAPE_AXIS_OUT_OF_RANGE); return m_core->m_shape[ax]; }
	KInt operator [](KInt ax) const { if (ax < 0 || ax >= size()) THROW(KERR_SHAPE_AXIS_OUT_OF_RANGE); return m_core->m_shape[ax]; }

	bool operator ==(KaiShape& shape) const { KInt sz = size(); if (sz != shape.size()) return false; for (KInt n = 0; n < sz; n++) if (m_core->m_shape[n] != shape[n]) return false; return true; }
	bool operator !=(KaiShape& shape) const { KInt sz = size(); if (sz != shape.size()) return true; for (KInt n = 0; n < sz; n++) if (m_core->m_shape[n] != shape[n]) return true; return false; }

	KString desc() const {
		KString desc, delimeter = "<";
		for (KInt n = 0; n < size(); n++) { desc += delimeter; delimeter = ",";  desc += to_string(m_core->m_shape[n]); }
		return desc + ">";
	}

protected:
	KaiShapeCore* m_core;
	friend class KaiValue;
};

*/