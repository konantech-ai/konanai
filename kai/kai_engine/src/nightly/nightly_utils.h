/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include <iostream>
#include <chrono>
#include <string>

#include "../include/kai_value.hpp"

// TRACE for debugging
#define ACTIVATE_TRACE    0
#define TRACE_KAI_FOLDER_CLASS_RECURSIVE_DATASET_READFILE     0
#define TRACE_KAI_FOLDER_CLASS_RECURSIVE_DATASET_FETCHDATA    0
#define TRACE_KAI_EXEC_CONTEXT_M_TRAIN                        0
#define TRACE_KAI_EXEC_CONTEXT_M_SHUFFLE_DATA                 0
#define TRACE_KAI_EXEC_CONTEXT_M_TRAIN_MINIBATCH              1
#define TRACE_KAI_EXEC_CONTEXT_M_FORWARD_NEURALNET            0
#define TRACE_KAI_EXEC_CONTEXT_M_EXEC_REPORT                  0
#define TRACE_KAI_EXEC_CONTEXT_M_VISUALIZE_OUTPUT             1
#define TRACE_KAI_EXPRESSION_EVALUATE_WITH_GRAD               0
#define TRACE_KAI_OPTIMIZER_BACKPROP_AFFINE                   0

// Activaate test codes for debugging
#define ACTIVATE_TEST    0
#define TEST_DISABLE_SHUFFLE                                  1

// hs.cho
// use NORANDOM   
//#define TEST_NORMAL_DISTRIBUTION_WITH_FIXED_VALUES            1

void KAI_API print_kvalue(Ken_value_type enum_type, KaiValue kval, std::string str_prefix, unsigned int indent_size, std::string str_end_string);
void KAI_API print_klist(KaiList klist, std::string str_prefix, unsigned int indent_size, std::string str_obj_name);
void KAI_API print_kdict(KaiDict kdict, std::string str_prefix, unsigned int indent_size, std::string str_obj_name);
void KAI_API print_kshape(KaiShape kshape, std::string str_prefix, unsigned int indent_size, std::string str_obj_name);

void KAI_API print_karray_kint(void* pkarray, std::string str_prefix, unsigned int indent_size, std::string str_obj_name, unsigned int print_step = 0);
void KAI_API print_karray_kfloat(void* pkarray, std::string str_prefix, unsigned int indent_size, std::string str_obj_name, unsigned int print_step = 0);

void KAI_API cv_load_and_resize_image(KFloat* pDstImageData, KString sImageFilename, KInt rows, KInt cols, KInt channels);
