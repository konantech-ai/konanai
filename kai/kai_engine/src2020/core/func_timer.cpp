/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "func_timer.h"
#include "value.h"
#include "log.h"

Dict FuncTimer::ms_records;
mutex_wrap FuncTimer::ms_mu_ftimer(true);

FuncTimer::FuncTimer(string name) {
	m_name = name;
	m_start_time = high_resolution_clock::now();
}

FuncTimer::~FuncTimer() {
	high_resolution_clock::time_point  end_time = high_resolution_clock::now();

	duration<double, std::ratio<1, 1000000>> time_span = duration_cast<duration<double, std::ratio<1, 1000000>>>(end_time - m_start_time);

	int64 mtime = (int64) time_span.count();

	Dict records;

	ms_mu_ftimer.lock();

	if (ms_records.find(m_name) != ms_records.end()) {
		records = ms_records[m_name];
	}
	else {
		records["num"] = 0;
		records["sum"] = (int64) 0;
		records["sqsum"] = (int64)0;
	}

	records["num"] = (int) records["num"] + 1;
	records["sum"] = (int64)records["sum"] + mtime;
	records["sqsum"] = (int64)records["sqsum"] + mtime * mtime;

	ms_records[m_name] = records;
	ms_mu_ftimer.unlock();

}

void FuncTimer::init() {
	ms_mu_ftimer.lock();
	ms_records.clear();
	ms_mu_ftimer.unlock();
}

void FuncTimer::dump() {
	if (!KArgs::show_functime) return;
	logger.Print("*** Function execution times ***");
	logger.Print("%-30s %15s %15s %15s %15s", "name", "num", "sum", "avg", "std");
	int total_num = 0;
	int64 total_sum = 0;
	int64 total_sqsum = 0;
	Dict avgs;
	Dict ranks;
	ms_mu_ftimer.lock();
	for (Dict::iterator it = ms_records.begin(); it != ms_records.end(); it++) {
		string name = it->first;
		Dict records = it->second;
		int num = records["num"];
		int64 sum = records["sum"];
		int64 sqsum = records["sqsum"];
		float avg = (float) sum / (float)num;
		avgs[name] = avg;
		ranks[name] = 0;
		total_num += num;
		total_sum += sum;
		total_sqsum += sqsum;
	}
	for (Dict::iterator it1 = avgs.begin(); it1 != avgs.end(); it1++) {
		for (Dict::iterator it2 = it1; it2 != avgs.end(); it2++) {
			if (it1 == it2) continue;
			float avg1 = it1->second;
			float avg2 = it2->second;

			if (avg1 < avg2) ranks[it1->first] = (int)ranks[it1->first] + 1;
			else ranks[it2->first] = (int)ranks[it2->first] + 1;
		}
	}
	for (int n = 0; n < (int)ranks.size(); n++) {
		for (Dict::iterator it = ranks.begin(); it != ranks.end(); it++) {
			if ((int) it->second == n) {
				string name = it->first;
				Dict records = ms_records[name];
				int num = records["num"];
				int64 sum = records["sum"];
				int64 sqsum = records["sqsum"];
				float avg = avgs[name];
				float std = (float) ::sqrt((float)sqsum / (float)num - avg*avg);
				logger.Print("%-30s %15d %15lld %15.3f %15.3f", name.c_str(), num, sum, avg, std);
				//break;
			}
		}
	}
	ms_mu_ftimer.unlock();
	float avg = (float)total_sum / (float)total_num;
	float std = (float) ::sqrt((float)total_sqsum / (float)total_num - avg * avg);
	logger.Print("%-30s %15d %15lld %15.3f %15.3f", "TOTAL", total_num, total_sum, avg, std);
}
