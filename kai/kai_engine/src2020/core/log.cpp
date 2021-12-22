/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdarg>

#include "../core/common.h"
#include "log.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

LogManager dlog;

LogManager::LogManager() {
	m_fid = NULL;
	m_mu_log = new std::mutex();
	m_pCbInst = NULL;
	m_pcbFunc = NULL;
}

LogManager::~LogManager() {
	if (m_fid) fclose(m_fid);
	delete m_mu_log;
}

void LogManager::open(const char* comm, const char* type, const char* mission) {
	m_mu_log->lock();

	assert(m_fid == NULL);

	const char* pos = strrchr(comm, '/');
	if (pos != NULL) comm = pos + 1;

	string dir = KArgs::log_root;
	Util::mkdir(dir);

	time_t now_t = time(NULL);
#ifdef KAI2021_WINDOWS
	struct tm now_tm;
	localtime_s(&now_tm, &now_t);
	struct tm* now = &now_tm;
#else
	struct tm* now = localtime(&now_t);
#endif

	char filepath[1024];
	snprintf(filepath, 1024, "%04d%02d%02d", now->tm_year + 1900, now->tm_mon + 1, now->tm_mday);
	m_date = filepath;
	snprintf(filepath , 1024, "%s-%02d%02d%02d", m_date.c_str(), now->tm_hour, now->tm_min, now->tm_sec);
	m_datetime = filepath;

	snprintf(filepath, 1024, "%s%s", dir.c_str(), m_date.c_str());

	Util::mkdir(filepath);

	size_t len = strlen(filepath);
	snprintf(filepath + len, 1024 - len, "/%s_%s", comm, type);

	if (mission != NULL) {
		len = strlen(filepath);
		snprintf(filepath + len, 1024 - len, "_%s", mission);
	}

	len = strlen(filepath);
	snprintf(filepath + len, 1024 - len, "%s.log", m_datetime.c_str());

	m_fid = Util::fopen(filepath, "wt");

	if (m_fid == NULL) {
		throw KaiException(KERR_FAIL_TO_OPEN_LOGFILE, filepath);
	}

	m_mu_log->unlock();
}

void LogManager::change_to_cont_file(string date, string filename) {
	m_mu_log->lock();

	assert(m_fid != NULL);

	string dir = KArgs::log_root;
	string filepath = dir + date + "/" + filename;

#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
	FILE* fid = NULL;
#else
	FILE* fid = fopen(filepath.c_str(), "at");
#endif

	if (fid == NULL) {
		logger.Print("Fail to open existing log file: %s", filepath.c_str());
		throw KaiException(KERR_ASSERT);
	}

	fprintf(m_fid, "*** Jump to the continuous log file: %s...\n", filepath.c_str());
	fclose(m_fid);

	m_fid = fid;
	m_date = date;
	fprintf(m_fid, "*** Process resumed ***\n");

	m_mu_log->unlock();
}

void LogManager::setCallback(void* pInst, KCbPrint* pcbFunc) {
	m_pCbInst = pInst;
	m_pcbFunc = pcbFunc;
}

void LogManager::PrintWait(const char* fmt, ...) {
	m_mu_log->lock();

	assert(m_fid != NULL);

	va_list args;

	if (m_pcbFunc) {
		char buffer[4096];
		va_start(args, fmt);
		int nWritten = vsnprintf(buffer, 4095, fmt, args);
		buffer[nWritten] = 0;
		m_pcbFunc(m_pCbInst, buffer, false);
		va_end(args);
	}

	va_start(args, fmt);
	vfprintf(m_fid, fmt, args);
	fflush(m_fid);
	va_end(args);

	m_mu_log->unlock();
}

void LogManager::Print(const char* fmt, ...) {
	m_mu_log->lock();

	assert(m_fid != NULL);

	va_list args;

	if (m_pcbFunc) {
		char buffer[4096];
		va_start(args, fmt);
		int nWritten = vsnprintf(buffer, 4095, fmt, args);
		buffer[nWritten] = 0;
		m_pcbFunc(m_pCbInst, buffer, true);
		va_end(args);
	}

	va_start(args, fmt);
	vfprintf(m_fid, fmt, args);
	fflush(m_fid);
	va_end(args);

	m_mu_log->unlock();
}

void LogManager::Bookeep(const char* fmt, ...) {
	m_mu_log->lock();

	assert(m_fid != NULL);

	va_list args;
	va_start(args, fmt);
	vfprintf(m_fid, fmt, args);
	va_end(args);

	fflush(m_fid);

	m_mu_log->unlock();
}
