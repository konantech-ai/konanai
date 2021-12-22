/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../include/kai_api.h"

class KLogger {
public:
	KLogger();
	virtual ~KLogger();

	void setCallback(void* pInst, KCbPrint* pcbFunc);

	void open(const char* comm, const char* type, const char* mission = NULL);

	void PrintWait(const char* format, ...);
	void Print(const char* format, ...);
	void PrintLogWait(const char* format, ...);
	void PrintLog(const char* format, ...);
	void Bookeep(const char* fmt, ...);

	string get_date() { return m_date; }
	string get_datetime() { return m_datetime; }

	void change_to_cont_file(string date, string filename);

protected:
	FILE* m_fid;
	string m_date;
	string m_datetime;
	mutex* m_mu_log;

	void* m_pCbInst;
	KCbPrint* m_pcbFunc;
};

extern KLogger logger;