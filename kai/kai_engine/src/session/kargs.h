/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include <string>

#include "kcommon.h"

using namespace std;

class KArgs {
public:
	static string kai_root;
	static string data_root;
	static string work_root;
	static string cache_root;
	static string param_root;
	static string image_root;
	static string result_root;
	static string log_root;
	static string data_index_root;
	static bool show_functime;
	static bool show_image;
	static bool trace_print;
};