/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#ifndef THROW
	void report_shell_error(int errCode); // in kai_shell.cpp
	#define THROW(x) {  report_shell_error(x); }
#endif

#include "../../kai_engine/src/include/kai_api.h"
#include "../../kai_engine/src/include/kai_types.h"
#include "../../kai_engine/src/include/kai_errors.h"

#define KERR_SHELL_DATASE_FEEDER_ALREADY_CONNECTED	90001
#define KERR_CALLBACK_NOT_DEFINED_FOR_FEEDING_DATA	90002
#define KERR_REQUEST_FLOAT_DATA_FOR_NON_FLOAT_FIELD	90003
#define KERR_REQUEST_ARRAY_DATA_WITH_BAD_NUM_IDXS	90004
#define KERR_REQUEST_NON_FLOAT_DATA_FOR_FLOAT_FIELD	90005

#define KERR_CHK(call_exp) { KRetCode ret = call_exp; if (ret != KRetOK) THROW(ret) }

#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#ifdef KAI2021_WINDOWS
#include <direct.h>
#else
#include <cstring>
#endif