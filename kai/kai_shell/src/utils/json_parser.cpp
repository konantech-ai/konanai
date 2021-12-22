/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "json_parser.h"
#include "utils.h"

JsonParser::JsonParser() {
}

JsonParser::~JsonParser() {
}

KaiValue JsonParser::parse_file(KString sFilePath) {
	FileLookAheader aheader(sFilePath);
	return parse_json(aheader);
}

KaiValue JsonParser::parse_json(LookAheader& aheader) {
	int ch = aheader.look();

	if (ch == '[') return decode_list(aheader);
	else if (ch == '{') return decode_dict(aheader);
	else if ((ch >= '0' && ch <= '9') || ch == '-') return decode_number(aheader);
	else if (ch == '\'' || ch == '"') return decode_string(aheader);
	else return decode_bool(aheader);
}

KaiList JsonParser::decode_list(LookAheader& aheader) {
	KaiList list;

	if (aheader.at_end()) return list;

	aheader.check('[');
	int ch = aheader.look();

	if (ch != ']') {
		while (true) {
			KaiValue value = parse_json(aheader);
			list.push_back(value);
			if (aheader.look() == ']') break;
			aheader.check(',');
			ch = aheader.look();
		}
	}

	aheader.check(']');
	return list;
}

KaiDict JsonParser::decode_dict(LookAheader& aheader) {
	KaiDict dict;

	if (aheader.at_end()) return dict;

	aheader.check('{');
	int ch = aheader.look();

	if (ch != '}') {
		while (true) {
			KString key = decode_string(aheader);
			aheader.check(':');
			KaiValue value = parse_json(aheader);
			dict[key] = value;
			if (aheader.look() == '}') break;
			aheader.check(',');
			ch = aheader.look();;
		}
	}

	aheader.check('}');
	return dict;
}

KaiValue JsonParser::decode_number(LookAheader& aheader) {
	int value = 0, sign = 1;

	if (aheader.pass('-')) sign = -1;

	while (aheader.look() >= '0' && aheader.look() <= '9') {
		value = value * 10 + (aheader.get() - '0');
	}

	if (aheader.pass('.')) {
		float fvalue = (float)value, unit = (float)0.1;
		while (aheader.look() >= '0' && aheader.look() <= '9') {
			fvalue = fvalue + (float)(aheader.get() - '0') * unit;
			unit *= (float)0.1;
		}
		return (float)sign * fvalue;
	}
	else if (aheader.look() == 'e' || aheader.look() == 'E') {
		THROW(KERR_FAIL_ON_JSON_PARSING);
	}

	return sign * value;
}

KaiValue JsonParser::decode_string(LookAheader& aheader) {
	int quote = aheader.get();
	if (quote != '\'' && quote != '"') {
		aheader.report_err("missing quote for string");
		THROW(KERR_FAIL_ON_JSON_PARSING);
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

KaiValue JsonParser::decode_bool(LookAheader& aheader) {
	if (aheader.next("True")) return true;
	else if (aheader.next("False")) return false;
	THROW(KERR_FAIL_ON_JSON_PARSING);
	return false;
}

int LookAheader::look() {
	m_skip_space();
	return m_buffer[m_begin];
}

int LookAheader::get() {
	m_skip_space();
	return m_buffer[m_begin++];
}

void LookAheader::report_err(KString msg) {
	size_t left_from = (m_begin > 10) ? m_begin - 10 : 0;
	size_t left_size = (m_begin > 10) ? 10 : m_begin;

	size_t len = m_buffer.size();

	KString left = m_buffer.substr(left_from, left_size);

	size_t right_from = m_begin + 1;
	size_t right_size = ((size_t)m_begin < len - 10) ? 10 : len - m_begin - 1;

	KString right = m_buffer.substr(right_from, right_size);

	int ch = m_buffer[m_begin];

	THROW(KERR_FAIL_ON_JSON_PARSING); // , msg, right);

	//logger.Print("json parsing error: %s\n%s...%c...%s", msg.c_str(), left.c_str(), ch, right.c_str());
}

void LookAheader::check(int ch) {
	int read = get();
	if (read != ch) {
		char buffer[128];
		report_err(buffer);
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

KString LookAheader::substr(int ch) {
	int pos = m_begin;
	int read;

	while ((read = m_buffer[pos++]) != ch) {
		if (read == '\\') pos++;
		if (pos >= m_end) {
			pos = pos - m_begin;
			m_read_buffer();
		}
	}

	KString result = m_buffer.substr(m_begin, pos - m_begin - 1);

	m_begin = pos;

	size_t esc_pos = result.find('\\');

	while (esc_pos != KString::npos) {
		result = result.substr(0, esc_pos) + result.substr(esc_pos + 1);
		esc_pos = result.find('\\', esc_pos + 1);
	}

	return result;
}

bool LookAheader::next(KString str) {
	int length = (int)str.length();
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

StringLookAheader::StringLookAheader(KString exp) : LookAheader() {
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
	THROW(KERR_FAIL_ON_JSON_PARSING);
}

bool StringLookAheader::m_at_end() {
	return m_begin >= m_end;
};

FileLookAheader::FileLookAheader(KString path) {
	m_fid = Utils::fopen(path.c_str(), "rt");
	m_begin = 0;
	m_end = 0;
}

FileLookAheader::~FileLookAheader() {
	fclose(m_fid);
}

void FileLookAheader::m_read_buffer() {
	char buffer[10241];
	int nread = (int)fread(buffer, sizeof(char), 10240, m_fid);
	if (nread <= 0) {
		THROW(KERR_FAIL_ON_JSON_PARSING);
	}
	buffer[nread] = 0;

	m_acc_read += nread;

	if (m_begin < m_end) {
		m_buffer = m_buffer.substr(m_begin, m_end - m_begin) + (KString)buffer;
	}
	else {
		m_buffer = (KString)buffer;
	}

	m_begin = 0;
	m_end = (int)m_buffer.length();
}

bool FileLookAheader::m_at_end() {
	if (m_begin < m_end) return false;
	return feof(m_fid);
}
