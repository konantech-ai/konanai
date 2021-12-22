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


using namespace std;

#define THROW(x) throw KaiException(x)

#include "../include/kai_types.h"
#include "../utils/kexception.h"
#include "../session/kargs.h"
#include "../utils/klogger.h"

#define KAI_MAX_DIM 8

#define MAX(x,y) ((x>y)?(x):(y))
#define MIN(x,y) ((x<y)?(x):(y))

inline KInt ax_mod(KInt a, KInt b) { return b ? (a % b + b) % b : 0; }

class kmutex_wrap {
public:
	kmutex_wrap(bool use = true) { m_use = use; }
	virtual ~kmutex_wrap() {}
	void lock() { if (m_use) m_mutex.lock(); }
	void unlock() { if (m_use) m_mutex.unlock(); }
protected:
	bool m_use;
	std::mutex m_mutex;
};

enum class exp_op {
	/*dict, term, hidden,*/ feed, arg, subexp, constant, string, add, sub, mult, div, equal, _and, _or, gt, lt, ge, le, exp, log, mean, sum, sqrt, square,
	argmax, max, max_col, equal_col, sigmoid, softmax, subvector, vstack, filter, iou,
	sigmoid_cross_entropy_with_logits, softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_idx,
};

#define NARRAY(x) KaiArray<KInt>((KaiArrayCore<KInt>*)(KHObject)x);
#define FARRAY(x) KaiArray<KFloat>((KaiArrayCore<KFloat>*)(KHObject)x);
