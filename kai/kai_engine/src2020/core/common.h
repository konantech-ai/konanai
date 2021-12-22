/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <set>
#include <mutex>
#include <queue>
#include <thread>

#ifdef KAI2021_WINDOWS

#include <io.h>
#include <iostream>
#include <iomanip>
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <random>

#include "../../src/session/kcommon.h"

using namespace std;

/*
#ifdef KAI2021_WINDOWS
#else
#endif
*/

typedef long long int64;

template <class T> class Array;

//enum class dt { dt_bool1=1, dt_int32=14, dt_float32=24 };
//enum class dt { none = 0, bool1 = 1, int8 = 11, int16 = 12, int32 = 14, int64 = 18, float32 = 24, float64 = 28 };

enum class sortdir { sd_asc, sd_desc };

#ifdef KAI2021_WINDOWS
#pragma warning(disable: 4819)

inline void mysprintf(char* p, const char* fmt, ...) {
	throw KaiException(KERR_ASSERT);
}
inline void mynanosleep(struct timespec* req, struct timespec* aux) {
	throw KaiException(KERR_ASSERT);
}
#else
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

enum class arr_type { arr_float, arr_int, arr_int64, arr_uchar, arr_bool, arr_short, arr_none };

inline arr_type to_enum_arr_type(string name) {
	if (name == "f") return arr_type::arr_float;
	else throw KaiException(KERR_ASSERT);
	return arr_type::arr_none;
}

enum class loss_mode { regression, binary, classify, classify_idx, classify_1st, custom, autoencode, }; // , loss_mode::autoencode, dm_etc2, dm_etc3, dm_etc4, dm_etc5 };
enum class data_channel { train, test, validate, visualize, autoencode, }; // , dc_etc1, dc_etc2, dc_etc3, dc_etc4, dc_etc5 };
enum class param_init { zeros, ones, gauss, uniform };

class mutex_wrap {
public:
	mutex_wrap(bool use = true) {
		m_use = use;
	}

	virtual ~mutex_wrap() {
	}

	void lock() {
		if (m_use) m_mutex.lock();
	}

	void unlock() {
		if (m_use) m_mutex.unlock();
	}

protected:
	bool m_use;
	std::mutex m_mutex;
};

class Layer;
class Value;

typedef vector<Value> List;
typedef map<string, Value> Dict;
typedef vector<Layer*> Layers;

class Dim;
class Shape;
class Idx;
