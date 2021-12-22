/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once
#ifdef KAI2021_WINDOWS //hs.cho
#ifdef KAI_CUDA_EXPORTS
#define KAI_CUDA_API __declspec(dllexport)
#else
#define KAI_CUDA_API __declspec(dllimport)
#endif

extern "C" KAI_CUDA_API int get_phone_num();

typedef int(__stdcall* ANSWERCB)(int, int);
typedef int* (__stdcall* ALLOCCB)(int);
typedef void (__stdcall* FREECB)(int*);

extern ANSWERCB cb_answer;
extern ALLOCCB cb_alloc;
extern FREECB cb_free;


extern FREECB cb_free;
extern "C" KAI_CUDA_API int SetCallback(ANSWERCB fp, ALLOCCB fa, FREECB ff, int n, int m);

#include "../core/common.h"

class MysqlConn {
public:
	MysqlConn();
	virtual ~MysqlConn();

	void test();
};
#endif