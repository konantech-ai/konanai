/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "value.h"
#include "array.h"
#include "log.h"
#include "util.h"

#include <stdio.h>
#include <cstdarg>

const Value None;

//#define TEMP_DIM_SAVE_MISS

mutex_wrap ValueCore::ms_mu_ref(true);

ValueCore::ValueCore() {
	ms_mu_ref.lock();
	m_type = vt::none;
	m_value.m_int = 0;
	m_nRefCount = 1;
	ms_mu_ref.unlock();
}

void ValueCore::destroy() {
	ms_mu_ref.lock();
	bool del = (--m_nRefCount <= 0);
	ms_mu_ref.unlock();
	if (del) delete this;
}

ValueCore::~ValueCore() {
	if (m_type == vt::list) delete (List*)m_value.m_pData;
	else if (m_type == vt::dict) delete (Dict*)m_value.m_pData;
	else if (m_type == vt::shape) delete (Shape*)m_value.m_pData;
	else if (m_type == vt::farray) delete (Array<float>*)m_value.m_pData;
	else if (m_type == vt::barray) delete (Array<bool>*)m_value.m_pData;
	else if (m_type == vt::narray) delete (Array<int>*)m_value.m_pData;
	else if (m_type == vt::n64array) delete (Array<int64>*)m_value.m_pData;
#ifdef _DEBUG
	else if (m_type == vt::string); // m_string은 따로 삭제할 필요가 없음
	else if (m_type == vt::none); // 삭제할 정보 자체가 없음
	else if (m_type == vt::kfloat); // 단순 필드는 m_value 자동 삭제로 충분
	else if (m_type == vt::kint); // 단순 필드는 m_value 자동 삭제로 충분
	else if (m_type == vt::int64); // 단순 필드는 m_value 자동 삭제로 충분
	else if (m_type == vt::kbool); // 단순 필드는 m_value 자동 삭제로 충분
	else assert(0);
#endif
}

Value::Value(const Value& src) {
	ValueCore::ms_mu_ref.lock();
	m_core = src.m_core;
	m_core->m_nRefCount++;
	ValueCore::ms_mu_ref.unlock();
}

Value& Value::operator = (const Value& src) {
	if (this == &src) return *this;
	m_core->destroy();
	ValueCore::ms_mu_ref.lock();
	m_core = src.m_core;
	m_core->m_nRefCount++;
	ValueCore::ms_mu_ref.unlock();
	return *this;
}

Value::Value(List list) {
	m_core = new ValueCore();
	m_core->m_type = vt::list;

	List* pList = new List();
	*pList = list;
	m_core->m_value.m_pData = (void*) pList;
}

Value::Value(Dict dict) {
	m_core = new ValueCore();
	m_core->m_type = vt::dict;

	Dict* pDict = new Dict();
	*pDict = dict;
	m_core->m_value.m_pData = (void*)pDict;
}

Value::Value(Array<float> array) {
	m_core = new ValueCore();
	m_core->m_type = vt::farray;

	Array<float>* pArray = new Array<float>();
	*pArray = array;
	m_core->m_value.m_pData = (void*)pArray;
}

Value::Value(Array<int> array) {
	m_core = new ValueCore();
	m_core->m_type = vt::narray;

	Array<int>* pArray = new Array<int>();
	*pArray = array;
	m_core->m_value.m_pData = (void*)pArray;
}

Value::Value(Array<int64> array) {
	m_core = new ValueCore();
	m_core->m_type = vt::n64array;

	Array<int64>* pArray = new Array<int64>();
	*pArray = array;
	m_core->m_value.m_pData = (void*)pArray;
}

Value::Value(Array<bool> array) {
	m_core = new ValueCore();
	m_core->m_type = vt::barray;

	Array<bool>* pArray = new Array<bool>();
	*pArray = array;
	m_core->m_value.m_pData = (void*)pArray;
}

Value::Value(Shape shape) {
	m_core = new ValueCore();
	m_core->m_type = vt::shape;

	Shape* pShape = new Shape();
	*pShape = shape;
	m_core->m_value.m_pData = (void*)pShape;
}

Value::Value(const char* str) {
	m_core = new ValueCore();
	m_core->m_type = vt::string;
	m_core->m_string = (string) str;
}

Value::Value(string str) {
	m_core = new ValueCore();
	m_core->m_type = vt::string;
	m_core->m_string = str;
}

Value::Value(int val) {
	m_core = new ValueCore();
	m_core->m_type = vt::kint;
	m_core->m_value.m_int = val;
}

Value::Value(int64 val) {
	m_core = new ValueCore();
	m_core->m_type = vt::int64;
	m_core->m_value.m_int64 = val;
}

Value::Value(float val) {
	m_core = new ValueCore();
	m_core->m_type = vt::kfloat;
	m_core->m_value.m_float = val;
}

Value::Value(bool val) {
	m_core = new ValueCore();
	m_core->m_type = vt::kbool;
	m_core->m_value.m_bool = val;
}

Value::operator List() {
	assert(m_core->m_type == vt::list);
	List list = *(List*)m_core->m_value.m_pData;
	return list;
}

Value::operator Dict() {
	assert(m_core->m_type == vt::dict);
	Dict dict = *(Dict*)m_core->m_value.m_pData;
	return dict;
}

#include <iostream>

Value::operator Array<float>() {
	if (m_core->m_type != vt::farray) {
		std::thread::id this_id = std::this_thread::get_id();
		std::cout << "Value::operator Array<float>() error in thread " << this_id << "...\n";
		throw KaiException(KERR_ASSERT);
	}
	Array<float> array = *(Array<float>*)m_core->m_value.m_pData;
	return array;
}

Value::operator Array<int>() {
	assert(m_core->m_type == vt::narray);
	Array<int> array = *(Array<int>*)m_core->m_value.m_pData;
	return array;
}

Value::operator Array<int64>() {
	assert(m_core->m_type == vt::n64array);
	Array<int64> array = *(Array<int64>*)m_core->m_value.m_pData;
	return array;
}

Value::operator Array<bool>() {
	assert(m_core->m_type == vt::barray);
	Array<bool> array = *(Array<bool>*)m_core->m_value.m_pData;
	return array;
}

Value::operator Shape() {
	assert(m_core->m_type == vt::shape);
	Shape shape = *(Shape*)m_core->m_value.m_pData;
	return shape;
}

Value::operator string() {
	assert(m_core->m_type == vt::string);
	return m_core->m_string;
}

Value::operator float() {
	if (m_core->m_type == vt::kint) return (float)m_core->m_value.m_int;
	assert(m_core->m_type == vt::kfloat);
	return m_core->m_value.m_float;
}

Value::operator int() {
	if (m_core->m_type != vt::kint) {
		std::thread::id this_id = std::this_thread::get_id();
		std::cout << "Value::operator int() error in thread " << this_id << "...\n";
		throw KaiException(KERR_ASSERT);
	}
	return m_core->m_value.m_int;
}

Value::operator int64() {
	if (m_core->m_type == vt::kint) {
		return (int64) m_core->m_value.m_int;
	}
	assert(m_core->m_type == vt::int64);
	return m_core->m_value.m_int64;
}

Value::operator bool() {
	assert(m_core->m_type == vt::kbool);
	return m_core->m_value.m_bool;
}

Value Value::decode_fmt_list(const char* fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	string result = Util::str_format(fmt, ap);
	va_end(ap);

	StringLookAheader aheader(result);
	return parse_json(aheader);
}

Value Value::parse_json_file(string file_path) {
	FileLookAheader aheader(file_path);
	return parse_json(aheader);
}

Value Value::parse_json(LookAheader& aheader) {
	int ch = aheader.look();

	if (ch == '[') return decode_list(aheader);
	else if (ch == '{') return decode_dict(aheader);
	else if ((ch >= '0' && ch <= '9') || ch == '-') return decode_number(aheader);
	else if (ch == '\'' || ch == '"') return decode_string(aheader);
	else return decode_bool(aheader);
}

List Value::parse_list(const char* exp) {
	if (exp == NULL || exp[0] == 0) return List();

	StringLookAheader aheader(exp);
	return decode_list(aheader);
}

Dict Value::parse_dict(const char* exp) {
	if (exp == NULL || exp[0] == 0) return Dict();

	StringLookAheader aheader(exp);
	return decode_dict(aheader);
}

List Value::decode_list(LookAheader& aheader) {
	List list;

	if (aheader.at_end()) return list;

	aheader.check('[');
	int ch = aheader.look();

	if (ch != ']') {
		while (true) {
			Value value = parse_json(aheader);
			list.push_back(value);
			if (aheader.look() == ']') break;
			aheader.check(',');
			ch = aheader.look();
		}
	}
	
	aheader.check(']');
	return list;
}

Dict Value::decode_dict(LookAheader& aheader) {
	Dict dict;

	if (aheader.at_end()) return dict;

	aheader.check('{');
	int ch = aheader.look();

	if (ch != '}') {
		while (true) {
			string key = decode_string(aheader);
			aheader.check(':');
			Value value = parse_json(aheader);
			dict[key] = value;
			if (aheader.look() == '}') break;
			aheader.check(',');
			ch = aheader.look();;
		}
	}
	
	aheader.check('}');
	return dict;
}

Value Value::decode_number(LookAheader& aheader) {
	int value = 0, sign = 1;

	if (aheader.pass('-')) sign = -1;

	while (aheader.look() >= '0' && aheader.look() <= '9') {
			value = value * 10 + (aheader.get() - '0');
	}

	if (aheader.pass('.')) {
		float fvalue = (float) value, unit = (float) 0.1;
		while (aheader.look() >= '0' && aheader.look() <= '9') {
			fvalue = fvalue + (float)(aheader.get() - '0') * unit;
			unit *= (float) 0.1;
		}
		return (float) sign * fvalue;
	}
	else if (aheader.look() == 'e' || aheader.look() == 'E') {
		throw KaiException(KERR_ASSERT);
	}

	return sign * value;
}

Value Value::decode_string(LookAheader& aheader) {
	int quote = aheader.get();
	if (quote != '\'' && quote != '"') {
		aheader.report_err("missing quote for string");
		throw KaiException(KERR_ASSERT);
	}
	return aheader.substr(quote);
	/*
	const char* from = ++rest;
	while (*rest++ != quote) {
		assert(*rest != 0);
	}
	return string(from, rest-from-1);
	*/
}

Value Value::decode_bool(LookAheader& aheader) {
	if (aheader.next("True")) return true;
	else if (aheader.next("False")) return false;
	throw KaiException(KERR_ASSERT);
	return false;
}

void Value::update_dict(Dict& dest, const char* exp) {
	Dict src = parse_dict(exp);

	for (Dict::iterator it = src.begin(); it != src.end(); it++) {
		string key = it->first;
		dest[key] = src[key];
	}
}

Dict Value::merge_dict(Dict src1, const char* exp) {
	Dict merged;
	Dict src2 = parse_dict(exp);

	for (Dict::iterator it = src1.begin(); it != src1.end(); it++) {
		string key = it->first;
		merged[key] = src1[key];
	}

	for (Dict::iterator it = src2.begin(); it != src2.end(); it++) {
		string key = it->first;
		merged[key] = src2[key];
	}

	return merged;
}

Value Value::copy(Value src) {
	switch (src.type()) {
	case vt::kbool:
		return (bool)src;
	case vt::kint:
		return (int)src;
	case vt::int64:
		return (int64)src;
	case vt::kfloat:
		return (float)src;
	case vt::string:
		return (string)src;
	case vt::list:
	{	List temp = src;
		return Value::copy_list(temp);
	}
	case vt::dict:
	{
		Dict temp = src;
		return Value::copy_dict(temp);
	}
	case vt::farray:
	{
		Array<float> temp = src;
		return temp.deepcopy();
	}
	case vt::narray:
	{
		Array<int> temp = src;
		return temp.deepcopy();
	}
	case vt::n64array:
	{
		Array<int64> temp = src;
		return temp.deepcopy();
	}
	case vt::barray:
	{	
		Array<bool> temp = src;
		return temp.deepcopy();
	}
	case vt::shape:
	{
		Shape temp = src;
		return Shape(temp);
	}
	default:
		throw KaiException(KERR_ASSERT);
	}

	return 0;
}

Dict Value::copy_dict(Dict src) {
	Dict clone;

	for (Dict::iterator it = src.begin(); it != src.end(); it++) {
		string key = it->first;
		clone[key] = Value::copy(src[key]);
	}

	return clone;
}

List Value::copy_list(List src) {
	List clone;

	for (List::iterator it = src.begin(); it != src.end(); it++) {
		clone.push_back(Value::copy(*it));
	}

	return clone;
}

string Value::m_encode_esc(string str) {
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

void Value::dict_accumulate(Dict& dest, Dict src) {
	for (Dict::iterator it = src.begin(); it != src.end(); it++) {
		string key = it->first;
		float value = it->second;

		if (key[0] == '#') continue;

		if (dest.find(key) != dest.end()) {
			float old_value = dest[key];
			dest[key] = old_value + value;
		}
		else {
			dest[key] = value;
		}
	}

	if (dest.find("#count#") != dest.end()) {
		int count = dest["#count#"];
		dest["#count#"] = count + 1;
	}
	else {
		dest["#count#"] = 1;
	}
}

Dict Value::dict_mean_reset(Dict& dict, int count) {
	if (count < 0) count = dict["#count#"];

	Dict means;

	for (Dict::iterator it = dict.begin(); it != dict.end(); it++) {
		string key = it->first;
		float value = it->second;

		if (key == "#count#") continue;

		means[key] = (count > 0) ? (value / (float)count) : 0;
	}

	dict.clear();

	return means;
}

string Value::description() {
	switch (m_core->m_type) {
	case vt::none:
		return "None";
	case vt::kbool:
		return m_core->m_value.m_bool ? "True" : "False";
	case vt::kint:
		return to_string(m_core->m_value.m_int);
	case vt::int64:
		return to_string(m_core->m_value.m_int64);
	case vt::kfloat:
		return to_string(m_core->m_value.m_float);
	case vt::string:
		return "'" + m_encode_esc(m_core->m_string) + "'";
	case vt::list:
		return description(*(List*)m_core->m_value.m_pData);
	case vt::dict:
		return description(*(Dict*)m_core->m_value.m_pData);
	case vt::shape:
		return description(*(Shape*)m_core->m_value.m_pData);
	case vt::farray:
	{
		Array<float> farr = *(Array<float>*)m_core->m_value.m_pData;
		return "farr" + farr.shape().desc();
		/*
		Idx idx;
		idx.set_size(farr.dim());
		string contents = farr.m_to_string(idx, 0, false);
		return contents;
		*/
	}
	case vt::narray:
	{
		Array<int> narr = *(Array<int>*)m_core->m_value.m_pData;
		Idx idx;
		idx.set_size(narr.dim());
		string contents = narr.m_to_string(idx, 0, false);
		return contents;
	}
	case vt::barray:
		throw KaiException(KERR_ASSERT);
		return "hello";
	}
	throw KaiException(KERR_ASSERT);
	return "xxx";
}

void Value::print_dict_keys(Dict dict, string desc) {
	string keys = "Keys of Dict(" + desc + "):";

	for (Dict::iterator it = dict.begin(); it != dict.end(); it++) {
		keys += " " + it->first;
	}

	logger.Print("%s", keys.c_str());
}

string Value::description(Dict dict) {
	string desc;
	string delimeter = "{";
	for (Dict::iterator it = dict.begin(); it != dict.end(); it++) {
		desc += delimeter;
		delimeter = ",";

		string key = it->first;
		string value = it->second.description();

		desc += "'" + key + "':" + value;
	}

	desc += "}";

	return desc;
}

string Value::description(List list) {
	string desc;
	string delimeter = "[";
	for (List::iterator it = list.begin(); it != list.end(); it++) {
		desc += delimeter;
		delimeter = ",";
		desc += it->description();
	}

	desc += "]";

	return desc;
}

string Value::description(Shape shape) {
	string desc;
	string delimeter = "(";
	for (int n = 0; n < shape.size(); n++) {
		desc += delimeter;
		delimeter = ",";
		desc += to_string(shape[n]);
	}

	desc += ")";

	return desc;
}

void Value::serial_save(FILE* fid, Value value) {
	m_fwrite(&value.m_core->m_type, sizeof(enum vt), 1, fid);

	switch (value.m_core->m_type) {
	case vt::none:
		break;
	case vt::kbool:
		m_fwrite(&value.m_core->m_value.m_bool, sizeof(bool), 1, fid);
		break;
	case vt::kint:
		m_fwrite(&value.m_core->m_value.m_int, sizeof(int), 1, fid);
		break;
	case vt::int64:
		m_fwrite(&value.m_core->m_value.m_int64, sizeof(int64), 1, fid);
		break;
	case vt::kfloat:
		m_fwrite(&value.m_core->m_value.m_float, sizeof(float), 1, fid);
		break;
	case vt::string:
		serial_save(fid, value.m_core->m_string);
		break;
	case vt::list:
		serial_save(fid, (*(List*)value.m_core->m_value.m_pData));
		break;
	case vt::dict:
		serial_save(fid, (*(Dict*)value.m_core->m_value.m_pData));
		break;
	case vt::shape:
		serial_save(fid, (*(Shape*)value.m_core->m_value.m_pData));
		break;
	case vt::farray:
		serial_save(fid, (*(Array<float>*)value.m_core->m_value.m_pData));
		break;
	case vt::narray:
		serial_save(fid, (*(Array<int>*)value.m_core->m_value.m_pData));
		break;
	case vt::barray:
		serial_save(fid, (*(Array<bool>*)value.m_core->m_value.m_pData));
		break;
	}
}

void Value::serial_save(FILE* fid, string dat) {
	int str_leng = (int) dat.length();
	m_fwrite(&str_leng, sizeof(int), 1, fid);
	m_fwrite(dat.c_str(), sizeof(char), str_leng, fid);
}

void Value::serial_save(FILE* fid, Dict dict) {
	int64 length =  dict.size();

	m_fwrite(&length, sizeof(int64), 1, fid);
	for (Dict::iterator it = dict.begin(); it != dict.end(); it++) {
		serial_save(fid, (string) it->first);
		serial_save(fid, it->second);
	}
}

void Value::serial_save(FILE* fid, List list) {
	int64 length =  list.size();

	m_fwrite(&length, sizeof(int64), 1, fid);
	for (List::iterator it = list.begin(); it != list.end(); it++) {
		serial_save(fid, *it);
	}
}

void Value::serial_save(FILE* fid, Array<float> array) {
	CudaConn cuda("serial_save", NULL);

	int64 data_size = array.m_core->data_size();

	serial_save(fid, array.dimension());

	float* data = cuda.get_host_data(array);

	m_fwrite(&data_size, sizeof(int64), 1, fid);
	m_fwrite(data, data_size, 1, fid);
}

void Value::serial_save(FILE* fid, Array<int> array) {
	int64 data_size = array.m_core->data_size();

	serial_save(fid, array.dimension());

	m_fwrite(&data_size, sizeof(int64), 1, fid);
	m_fwrite(array.m_core->data(), data_size, 1, fid);
}

void Value::serial_save(FILE* fid, Array<bool> array) {
	int64 data_size = array.m_core->data_size();

	serial_save(fid, array.dimension());

	m_fwrite(&data_size, sizeof(int64), 1, fid);
	m_fwrite(array.m_core->data(), data_size, 1, fid);
}

void Value::serial_save(FILE* fid, Shape shape) {
	int64 size = shape.size();
	m_fwrite(&size, sizeof(int), 1, fid);

	for (int n = 0; n < size; n++) {
		int64 nth_size = shape[n];
		m_fwrite(&nth_size, sizeof(int64), 1, fid);
	}
}

void Value::serial_save(FILE* fid, Dim dim) {
#ifdef TEMP_DIM_SAVE_MISS
	m_fwrite(&dim.m_core->m_nDim, sizeof(int64), 1, fid);
#else
	m_fwrite(&dim.m_core->m_nDim, sizeof(int), 1, fid);
#endif
	m_fwrite(&dim.m_core->m_element, sizeof(int64), dim.m_core->m_nDim, fid);
}

void Value::serial_save(FILE* fid, map<int, int>& dat) {
	int64 length = dat.size();

	m_fwrite(&length, sizeof(int64), 1, fid);
	for (map<int, int>::iterator it = dat.begin(); it != dat.end(); it++) {
		m_fwrite(&it->first, sizeof(int), 1, fid);
		m_fwrite(&it->second, sizeof(int), 1, fid);
	}
}

void Value::serial_save(FILE* fid, map<int64, int64>& dat) {
	int64 length = dat.size();

	m_fwrite(&length, sizeof(int64), 1, fid);
	for (map<int64, int64>::iterator it = dat.begin(); it != dat.end(); it++) {
		m_fwrite(&it->first, sizeof(int64), 1, fid);
		m_fwrite(&it->second, sizeof(int64), 1, fid);
	}
}

void Value::serial_save(FILE* fid, int* pnums, int64 num) {
	//m_fwrite(&num, sizeof(int64), 1, fid);
	m_fwrite(pnums, sizeof(int), num, fid);
}

void Value::serial_save(FILE* fid, int64* pnums, int64 num) {
	//m_fwrite(&num, sizeof(int64), 1, fid);
	m_fwrite(pnums, sizeof(int64), num, fid);
}

void Value::serial_load(FILE* fid, Value& value) {
	enum vt type;
	m_fread(&type, sizeof(enum vt), 1, fid);

	int ival;
	int64 i64val;
	bool bval;
	float fval;
	string sval;
	List list;
	Dict dict;
	Shape shape;
	Array<float> farray;
	Array<int> narray;
	Array<bool> barray;

	switch (type) {
	case vt::none:
		break;
	case vt::kbool:
		m_fread(&bval, sizeof(bool), 1, fid);
		value = bval;
		break;
	case vt::kint:
		m_fread(&ival, sizeof(int), 1, fid);
		value = ival;
		break;
	case vt::int64:
		m_fread(&i64val, sizeof(int64), 1, fid);
		value = i64val;
		break;
	case vt::kfloat:
		m_fread(&fval, sizeof(float), 1, fid);
		value = fval;
		break;
	case vt::string:
		serial_load(fid, sval);
		value = sval;
		break;
	case vt::list:
		serial_load(fid, list);
		value = list;
		break;
	case vt::dict:
		serial_load(fid, dict);
		value = dict;
		break;
	case vt::shape:
		serial_load(fid, shape);
		value = shape;
		break;
	case vt::farray:
		serial_load(fid, farray);
		value = farray;
		break;
	case vt::narray:
		serial_load(fid, narray);
		value = narray;
		break;
	case vt::barray:
		serial_load(fid, barray);
		value = barray;
		break;
	}
}

void Value::serial_load(FILE* fid, string& dat) {
	int str_leng;
	char buf[1024];
	m_fread(&str_leng, sizeof(int), 1, fid);
	if (str_leng < 1024) {
		m_fread(buf, sizeof(char), str_leng, fid);
		buf[str_leng] = 0;
		dat = buf;
	}
	else {
		char* pBuffer = (char*) malloc(str_leng+1);
		m_fread(pBuffer, sizeof(char), str_leng, fid);
		pBuffer[str_leng] = 0;
		dat = pBuffer;
		free(pBuffer);
	}
}


void Value::serial_load(FILE* fid, Dict& dict) {
	int64 length;
	string key;
	Value value;

	m_fread(&length, sizeof(int64), 1, fid);
	for (int n = 0; n < length; n++) {
		serial_load(fid, key);
		serial_load(fid, value);
		dict[key] = value;
	}
}

void Value::serial_load(FILE* fid, List& list) {
	int64 length;
	Value value;

	m_fread(&length, sizeof(int64), 1, fid);
	for (int n = 0; n < length; n++) {
		serial_load(fid, value);
		list.push_back(value);
	}
}

void Value::serial_load(FILE* fid, Array<float>& array) {
	Dim dim;
	int64 data_size;

	serial_load(fid, dim);
	array = Array<float>(dim);
	m_fread(&data_size, sizeof(int64), 1, fid);
	assert(data_size == (int64)(array.total_size() * sizeof(float)));
	m_fread(array.data_ptr(), data_size, 1, fid);
}

void Value::serial_load(FILE* fid, Array<int>& array) {
	Dim dim;
	int64 data_size;

	serial_load(fid, dim);
	array = Array<int>(dim);
	m_fread(&data_size, sizeof(int64), 1, fid);
	assert(data_size == (int64)(array.total_size() * sizeof(int)));
	m_fread(array.data_ptr(), data_size, 1, fid);
}

void Value::serial_load(FILE* fid, Array<bool>& array) {
	Dim dim;
	int64 data_size;

	serial_load(fid, dim);
	array = Array<bool>(dim);
	m_fread(&data_size, sizeof(int64), 1, fid);
	assert(data_size == (int64)(array.total_size() * sizeof(bool)));
	m_fread(array.data_ptr(), data_size, 1, fid);
}

void Value::serial_load(FILE* fid, Shape& shape) {
	int size;
	int64 element[KAI_MAX_DIM];

	m_fread(&size, sizeof(int), 1, fid);

	m_fread(element, sizeof(int64), size, fid);

	shape = Shape(element, size);
}

void Value::serial_load(FILE* fid, Dim& dim) {
#ifdef TEMP_DIM_SAVE_MISS
	int64 temp;
	m_fread(&temp, sizeof(int64), 1, fid);
	memcpy(&dim.m_core->m_nDim, &temp, sizeof(int));
#else
	m_fread(&dim.m_core->m_nDim, sizeof(int), 1, fid);
#endif
	m_fread(&dim.m_core->m_element, sizeof(int64), dim.m_core->m_nDim, fid);
}

void Value::serial_load(FILE* fid, map<int, int>& dat) {
	int64 length;
	int key, value;

	m_fread(&length, sizeof(int64), 1, fid);

	for (int n = 0; n < length; n++) {
		m_fread(&key, sizeof(int), 1, fid);
		m_fread(&value, sizeof(int), 1, fid);
		dat[key] = value;
	}
}

void Value::serial_load(FILE* fid, map<int64, int64>& dat) {
	int64 length;
	int64 key, value;

	m_fread(&length, sizeof(int64), 1, fid);

	for (int n = 0; n < length; n++) {
		m_fread(&key, sizeof(int64), 1, fid);
		m_fread(&value, sizeof(int64), 1, fid);
		dat[key] = value;
	}
}

void Value::serial_load(FILE* fid, int* pnums, int64 num) {
	m_fread(pnums, sizeof(int), num, fid);
}

void Value::serial_load(FILE* fid, int64* pnums, int64 num) {
	m_fread(pnums, sizeof(int64), num, fid);
}

void Value::serial_load_params(FILE* fid, Value& value) {
	enum vt type;
	m_fread(&type, sizeof(enum vt), 1, fid);

	assert(type == value.type());

	List list;
	Dict dict;
	Shape shape;
	Array<float> farray;
	Array<int> narray;
	Array<bool> barray;

	switch (type) {
	case vt::none:
		break;
	case vt::kbool:
		m_fread(&value.m_core->m_value.m_bool, sizeof(bool), 1, fid);
		break;
	case vt::kint:
		m_fread(&value.m_core->m_value.m_int, sizeof(int), 1, fid);
		break;
	case vt::int64:
		m_fread(&value.m_core->m_value.m_int64, sizeof(int64), 1, fid);
		break;
	case vt::kfloat:
		m_fread(&value.m_core->m_value.m_float, sizeof(float), 1, fid);
		break;
	case vt::string:
		serial_load_params(fid, value.m_core->m_string);
		break;
	case vt::list:
		list = *(List*) value.m_core->m_value.m_pData;
		serial_load_params(fid, list);
		break;
	case vt::dict:
		dict = *(Dict*)value.m_core->m_value.m_pData;
		serial_load_params(fid, dict);
		break;
	case vt::shape:
		shape = *(Shape*)value.m_core->m_value.m_pData;
		serial_load_params(fid, shape);
		break;
	case vt::farray:
		farray = *(Array<float>*)value.m_core->m_value.m_pData;
		serial_load_params(fid, farray);
		break;
	case vt::narray:
		narray = *(Array<int>*)value.m_core->m_value.m_pData;
		serial_load_params(fid, narray);
		break;
	case vt::barray:
		barray = *(Array<bool>*)value.m_core->m_value.m_pData;
		serial_load_params(fid, barray);
		break;
	}
}

void Value::serial_load_params(FILE* fid, string& dat) {
	int str_leng;
	char buf[1024];
	m_fread(&str_leng, sizeof(int), 1, fid);
	if (str_leng < 1024) {
		m_fread(buf, sizeof(char), str_leng, fid);
		buf[str_leng] = 0;
		dat = buf;
	}
	else {
		char* pBuffer = (char*)malloc(str_leng + 1);
		m_fread(pBuffer, sizeof(char), str_leng, fid);
		pBuffer[str_leng] = 0;
		dat = pBuffer;
		free(pBuffer);
	}
}


void Value::serial_load_params(FILE* fid, Dict& dict) {
	int64 length;

	m_fread(&length, sizeof(int64), 1, fid);

	assert(length == (int)dict.size());

	for (int n = 0; n < length; n++) {
		string key;
		serial_load_params(fid, key);
		Value& value = dict[key];
		serial_load_params(fid, value);
	}
}

void Value::serial_load_params(FILE* fid, List& list) {
	int64 length;

	m_fread(&length, sizeof(int64), 1, fid);

	assert(length == (int)list.size());

	for (int n = 0; n < length; n++) {
		serial_load_params(fid, list[n]);
	}
}

void Value::serial_load_params(FILE* fid, Array<float>& array) {
	Dim dim;
	int64 data_size;

	serial_load_params(fid, dim);
	assert(dim == array.shape());
	m_fread(&data_size, sizeof(int64), 1, fid);
	assert(data_size == (int64)(array.total_size() * sizeof(float)));
	if (array.is_cuda()) {
		void* pbuf = malloc(data_size);
		m_fread(pbuf, data_size, 1, fid);
		CudaConn::Copy_host_to_cuda(array.data_ptr(), pbuf, data_size);
		free(pbuf);
	}
	else {
		m_fread(array.data_ptr(), data_size, 1, fid);
	}
}

void Value::serial_load_params(FILE* fid, Array<int>& array) {
	Dim dim;
	int64 data_size;

	serial_load_params(fid, dim);
	assert(dim == array.shape());
	m_fread(&data_size, sizeof(int64), 1, fid);
	assert(data_size == (int64)(array.total_size() * sizeof(int)));
	if (array.is_cuda()) {
		void* pbuf = malloc(data_size);
		m_fread(pbuf, data_size, 1, fid);
		CudaConn::Copy_host_to_cuda(array.data_ptr(), pbuf, data_size);
		free(pbuf);
	}
	else {
		m_fread(array.data_ptr(), data_size, 1, fid);
	}
}

void Value::serial_load_params(FILE* fid, Array<bool>& array) {
	Dim dim;
	int64 data_size;

	serial_load_params(fid, dim);
	assert(dim == array.shape());
	m_fread(&data_size, sizeof(int64), 1, fid);
	assert(data_size == (int64)(array.total_size() * sizeof(bool)));
	if (array.is_cuda()) {
		void* pbuf = malloc(data_size);
		m_fread(pbuf, data_size, 1, fid);
		CudaConn::Copy_host_to_cuda(array.data_ptr(), pbuf, data_size);
		free(pbuf);
	}
	else {
		m_fread(array.data_ptr(), data_size, 1, fid);
	}
}

void Value::serial_load_params(FILE* fid, Shape& shape) {
	throw KaiException(KERR_ASSERT);
	int size;
	int64 element[KAI_MAX_DIM];

	m_fread(&size, sizeof(int), 1, fid);

	m_fread(element, sizeof(int64), size, fid);

	shape = Shape(element, size);
}

void Value::serial_load_params(FILE* fid, Dim& dim) {
	int dim_size;

#ifdef TEMP_DIM_SAVE_MISS
	int64 temp;
	m_fread(&temp, sizeof(int64), 1, fid);
	memcpy(&dim_size, &temp, sizeof(int));
#else
	m_fread(&dim_size, sizeof(int), 1, fid);
#endif
	dim.m_core->m_nDim = dim_size;
	m_fread(dim.m_core->m_element, sizeof(int64), dim_size, fid);
}

void Value::serial_load_params(FILE* fid, map<int, int>& dat) {
	throw KaiException(KERR_ASSERT);
	int64 length;
	int key, value;

	m_fread(&length, sizeof(int64), 1, fid);

	for (int n = 0; n < length; n++) {
		m_fread(&key, sizeof(int), 1, fid);
		m_fread(&value, sizeof(int), 1, fid);
		dat[key] = value;
	}
}

void Value::serial_load_params(FILE* fid, int* pnums, int num) {
	throw KaiException(KERR_ASSERT);
	m_fread(pnums, sizeof(int), num, fid);
}

Dict Value::wrap_dict(string key, Value value) {
	Dict dict;
	dict[key] = value;
	return dict;
}

int LookAheader::look() {
	m_skip_space();
	return m_buffer[m_begin];
}

int LookAheader::get() {
	m_skip_space();
	return m_buffer[m_begin++];
}

void LookAheader::report_err(string msg) {
	size_t left_from = (m_begin > 10) ? m_begin - 10 : 0;
	size_t left_size = (m_begin > 10) ? 10 : m_begin;

	size_t len = m_buffer.size();

	string left = m_buffer.substr(left_from, left_size);

	size_t right_from = m_begin + 1;
	size_t right_size = ((size_t)m_begin < len - 10) ? 10 : len - m_begin - 1;

	string right = m_buffer.substr(right_from, right_size);

	int ch = m_buffer[m_begin];

	logger.Print("json parsing error: %s\n%s...%c...%s", msg.c_str(), left.c_str(), ch, right.c_str());
}

void LookAheader::check(int ch) {
	int read = get();
	if (read != ch) {
		char buffer[128];
#ifdef KAI2021_WINDOWS
		throw KaiException(KERR_ASSERT);
#else
		sprintf(buffer, "unexpedted char (need '%c' vs. read '%c')", ch, read);
#endif
		report_err(buffer);
		throw KaiException(KERR_ASSERT);
	}
}

bool LookAheader::at_end() {
	m_skip_space();
	return m_at_end();
}

bool LookAheader::pass(int ch) {
	if (look() != ch) return false;
	m_begin++;
	return true;
}

string LookAheader::substr(int ch) {
	int pos = m_begin;
	int read;

	while ((read=m_buffer[pos++]) != ch) {
		if (read == '\\') pos++;
		if (pos >= m_end) {
			pos = pos - m_begin;
			m_read_buffer();
		}
	}

	string result = m_buffer.substr(m_begin, pos-m_begin-1);

	m_begin = pos;

	size_t esc_pos = result.find('\\');

	while (esc_pos != string::npos) {
		result = result.substr(0, esc_pos) + result.substr(esc_pos+1);
		esc_pos = result.find('\\', esc_pos+1);
	}

	return result;
}

bool LookAheader::next(string str) {
	int length = (int) str.length();
	m_skip_space();
	while (m_end - m_begin < length) {
		m_read_buffer();
	}
	if (m_buffer.substr(m_begin, length) == str) {
		m_begin += length;
		return true;
	}
	return false;
}

void LookAheader::m_skip_space() {
	while (true) {
		if (m_begin >= m_end) m_read_buffer();
		else {
			if (!isspace(m_buffer[m_begin])) break;
			m_begin++;
		}
	}
}

StringLookAheader::StringLookAheader(string exp) : LookAheader() {
	m_buffer = exp;
	m_begin = 0;
	m_end = (int)m_buffer.length();
}

StringLookAheader::~StringLookAheader() {
	while (m_begin < m_end && isspace(m_buffer[m_begin])) m_begin++;
	if (m_begin < m_end && m_buffer[m_begin] == 0) m_begin++;
	assert(m_begin == m_end);
}

void StringLookAheader::m_read_buffer() {
	throw KaiException(KERR_ASSERT);
}

bool StringLookAheader::m_at_end() {
	return m_begin >= m_end;
};

FileLookAheader::FileLookAheader(string path) {
	m_fid = Util::fopen(path.c_str(), "rt");
	m_begin = 0;
	m_end = 0;
}

FileLookAheader::~FileLookAheader() {
	fclose(m_fid);
}

void FileLookAheader::m_read_buffer() {
	char buffer[10241];
	int nread = (int)fread(buffer, sizeof(char), 10240, m_fid);
	if (nread <= 0) throw KaiException(KERR_ASSERT);
	buffer[nread] = 0;

	m_acc_read += nread;

	if (m_begin < m_end) {
		m_buffer = m_buffer.substr(m_begin, m_end - m_begin) + (string)buffer;
	}
	else {
		m_buffer = (string)buffer;
	}

	m_begin = 0;
	m_end = (int)m_buffer.length();
}

bool FileLookAheader::m_at_end() {
	if (m_begin < m_end) return false;
	return feof(m_fid);
}
