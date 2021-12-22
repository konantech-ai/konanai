/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../common.h"

class LookAheader {
public:
	int look();
	int get();
	void check(int ch);	// 반드시 특정 문자 입력 받아야 함
	bool at_end();
	bool pass(int ch); // 지정 문자면 통과하면서 ,true, 아니면 false
	KString substr(int ch);
	bool next(KString str);
	void report_err(KString msg);

protected:
	KString m_buffer;
	int m_begin, m_end;

	void m_skip_space();

	virtual void m_read_buffer() = 0;
	virtual bool m_at_end() = 0;
};

class FileLookAheader : public LookAheader {
public:
	FileLookAheader(KString path);
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
	StringLookAheader(KString exp);
	virtual ~StringLookAheader();

protected:
	virtual void m_read_buffer();
	virtual bool m_at_end();
};

class JsonParser {
public:
	JsonParser();
	virtual ~JsonParser();

	KaiValue parse_file(KString sFilePath);
	KaiValue parse_json(LookAheader& aheader);

	KaiList decode_list(LookAheader& aheader);
	KaiDict decode_dict(LookAheader& aheader);
	KaiValue decode_number(LookAheader& aheader);
	KaiValue decode_string(LookAheader& aheader);
	KaiValue decode_bool(LookAheader& aheader);
};

