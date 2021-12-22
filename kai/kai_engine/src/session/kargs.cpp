/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kargs.h"

#include <stdlib.h>  
#include <stdio.h>

//#ifdef KAI2021_WINDOWS

//hs.cho
//#include "../nightly/path_generator.h"


//hs.cho
string solution_root = string("./");// PathGenerator::GetSolutionPathA(4096);

string KArgs::kai_root = solution_root + "../";

//string KArgs::kai_root = "K:/kai2021/";

string KArgs::data_root = kai_root + "data/";
string KArgs::work_root = kai_root + "work/";

string KArgs::cache_root = work_root + "cache/";
string KArgs::param_root = work_root + "param/";
string KArgs::result_root = work_root + "result/";
string KArgs::image_root = work_root + "image/";
string KArgs::log_root = work_root + "log/";
string KArgs::data_index_root = work_root + "data_index/";
//#else
//string KArgs::data_root = "/home/dhyoon/work/data/";
//string KArgs::cache_root = "/home/dhyoon/work/kodell/cache/";
//string KArgs::param_root = "/home/dhyoon/work/kodell/param/";
//string KArgs::image_root = "/home/dhyoon/work/kodell/image/";
//string KArgs::result_root = "/home/dhyoon/work/kodell/result/";
//string KArgs::log_root = "/home/dhyoon/work/kodell/log/";
//string KArgs::data_index_root = "/home/dhyoon/work/kodell/data_index/";
//#endif

bool KArgs::show_functime = false;
bool KArgs::show_image = false;
bool KArgs::trace_print = true;