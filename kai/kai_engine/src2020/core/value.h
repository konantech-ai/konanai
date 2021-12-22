/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"
#include "array.h"
#include "log.h"

enum class vt { none, kbool, kint, int64, kfloat, string, list, dict, farray, narray, n64array, barray, shape };

class ValueCore {
public:
	ValueCore();
	virtual ~ValueCore();

protected:
	friend class Value;
	int m_nRefCount;
	enum vt m_type;
	union Value_value {
		bool m_bool;
		int m_int;
		int64 m_int64;
		float m_float;
		void* m_pData;
	} m_value;
	string m_string;

	static mutex_wrap ms_mu_ref;

	void destroy();
};

class LookAheader;

class Value {
public:
	Value() {
		m_core = new ValueCore();
	}
	Value(const Value& src);
	Value& operator = (const Value& src);

	Value(List list);
	Value(Dict dict);
	Value(Array<float> array);
	Value(Array<int> array);
	Value(Array<int64> array);
	Value(Array<bool> array);
	Value(Shape shape);
	Value(string str);
	Value(int val);
	Value(int64 val);
	Value(float val);
	Value(bool val);
	Value(const char* str);

	virtual ~Value() {
		m_core->destroy();
	}

	enum vt type() const { return m_core->m_type; }

	bool is_none() { return m_core->m_type == vt::none; }

	//List to_list();
	operator List();
	operator Dict();
	operator Array<float>();
	operator Array<int>();
	operator Array<int64>();
	operator Array<bool>();
	operator Shape();
	operator string();
	operator float();
	operator int();
	operator int64();
	operator bool();

	string description();
	static string description(Dict dict);
	static string description(List list);
	static string description(Shape shape);

	static Value decode_fmt_list(const char* exp, ...);
	static Value parse_json_file(string file_path);

	static Value parse_json(LookAheader& aheader);

	static List parse_list(const char* rest);
	static Dict parse_dict(const char* rest);
	static void update_dict(Dict& dest, const char* exp);
	static Dict merge_dict(Dict org, const char* exp);

	static Value copy(Value src);

	static Value seek_option(Dict options, string key, Value def) {
		return (options.find(key) != options.end()) ? options[key] : def;
	}

	static void dict_accumulate(Dict& dest, Dict src);
	static Dict dict_mean_reset(Dict& dict, int count = -1);

	static void serial_save(FILE* fid, Value value);
	static void serial_save(FILE* fid, string dat);
	static void serial_save(FILE* fid, Dict dict);
	static void serial_save(FILE* fid, List list);
	static void serial_save(FILE* fid, Array<float> array);
	static void serial_save(FILE* fid, Array<int> array);
	static void serial_save(FILE* fid, Array<int64> array);
	static void serial_save(FILE* fid, Array<bool> array);
	static void serial_save(FILE* fid, Dim dim);
	static void serial_save(FILE* fid, Shape shape);
	
	static void serial_save(FILE* fid, map<int, int>& dat);
	static void serial_save(FILE* fid, map<int64, int64>& dat);
	static void serial_save(FILE* fid, int* pnums, int64 num);
	static void serial_save(FILE* fid, int64* pnums, int64 num);

	static void serial_load(FILE* fid, Value& value);
	static void serial_load(FILE* fid, string& dat);
	static void serial_load(FILE* fid, Dict& dict);
	static void serial_load(FILE* fid, List& list);
	static void serial_load(FILE* fid, Array<float>& array);
	static void serial_load(FILE* fid, Array<int>& array);
	static void serial_load(FILE* fid, Array<int64>& array);
	static void serial_load(FILE* fid, Array<bool>& array);
	static void serial_load(FILE* fid, Dim& dim);
	static void serial_load(FILE* fid, Shape& shape);

	static void serial_load(FILE* fid, map<int, int>& dat);
	static void serial_load(FILE* fid, map<int64, int64>& dat);
	static void serial_load(FILE* fid, int* pnums, int64 num);
	static void serial_load(FILE* fid, int64* pnums, int64 num);

	static void serial_load_params(FILE* fid, Value& value);
	static void serial_load_params(FILE* fid, string& dat);
	static void serial_load_params(FILE* fid, Dict& dict);
	static void serial_load_params(FILE* fid, List& list);
	static void serial_load_params(FILE* fid, Array<float>& array);
	static void serial_load_params(FILE* fid, Array<int>& array);
	static void serial_load_params(FILE* fid, Array<int64>& array);
	static void serial_load_params(FILE* fid, Array<bool>& array);
	static void serial_load_params(FILE* fid, Dim& dim);
	static void serial_load_params(FILE* fid, Shape& shape);

	static void serial_load_params(FILE* fid, map<int, int>& dat);
	static void serial_load_params(FILE* fid, int* pnums, int num);

#ifdef KAI2021_WINDOWS
	static enum vt data_type();
#else
	template <class T>
	static enum vt data_type();
#endif

	static Dict wrap_dict(string key, Value value);

	static void print_dict_keys(Dict dict, string desc);

protected:
	static Dict copy_dict(Dict src);
	static List copy_list(List src);

	static List decode_list(LookAheader& aheader);
	static Dict decode_dict(LookAheader& aheader);
	static Value decode_number(LookAheader& aheader);
	static Value decode_string(LookAheader& aheader);
	static Value decode_bool(LookAheader& aheader);

	//static char ms_lookahead(const char*& rest);

	static void m_fwrite(const void* ptr, int64 size, int64 cnt, FILE* fid) {
#ifdef KAI2021_WINDOWS
		if (fwrite(ptr, size, cnt, fid) != cnt) throw KaiException(KERR_ASSERT);
#else
		if ((int64)fwrite(ptr, size, cnt, fid) != cnt) throw KaiException(KERR_ASSERT);
#endif
	}

	static void m_fread(void* ptr, int64 size, int64 cnt, FILE* fid) {
#ifdef KAI2021_WINDOWS
		int64 res = fread(ptr, size, cnt, fid);
#else
		int64 res = (int64)fread(ptr, size, cnt, fid);
#endif
		if (res != cnt) {
			logger.Print("res = %lld, cnt = %lld", res, cnt);
			throw KaiException(KERR_ASSERT);
		}
	}

	string m_encode_esc(string str);

protected:
	friend class ValueCore;
	friend class DictCore;
	ValueCore* m_core;
};

extern const Value None;

class LookAheader {
public:
	int look();
	int get();
	void check(int ch);	// 반드시 특정 문자 입력 받아야 함
	bool at_end();
	bool pass(int ch); // 지정 문자면 통과하면서 ,true, 아니면 false
	string substr(int ch);
	bool next(string str);
	void report_err(string msg);

protected:
	string m_buffer;
	int m_begin, m_end;

	void m_skip_space();

	virtual void m_read_buffer() = 0;
	virtual bool m_at_end() = 0;
};

class FileLookAheader : public LookAheader {
public:
	FileLookAheader(string path);
	virtual ~FileLookAheader();

protected:
	virtual void m_read_buffer();
	virtual bool m_at_end();

	FILE* m_fid;
	int m_pos;
	int m_acc_read;
};

class StringLookAheader : public LookAheader {
public:
	StringLookAheader(string exp);
	virtual ~StringLookAheader();

protected:
	virtual void m_read_buffer();
	virtual bool m_at_end();
};