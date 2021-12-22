/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

using namespace std::chrono;

class FuncTimer {
public:
	FuncTimer(string name);
	virtual ~FuncTimer();

	static void init();
	static void dump();
protected:
	string m_name;
	high_resolution_clock::time_point m_start_time;

	static Dict ms_records;
	static mutex_wrap ms_mu_ftimer;
};
