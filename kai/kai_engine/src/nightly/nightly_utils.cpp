/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../include/kai_errors.h"
#include "../math/karray.h"
#include "nightly_utils.h"

// For using OpenCV
#include <opencv2/opencv.hpp>

#define NIGHTLY_UTILS_INDENT_UNIT_SIZE	4

void print_kvalue(Ken_value_type enum_type, KaiValue kval, std::string str_prefix, unsigned int indent_size, std::string str_end_string) {
	std::string str_indent = std::string(indent_size, ' ');
	switch (enum_type) {
	case Ken_value_type::none:
		printf("%s%sNone%s", str_prefix.c_str(), str_indent.c_str(), str_end_string.c_str());
		break;
	case Ken_value_type::kint:
		printf("%s%s%lld%s", str_prefix.c_str(), str_indent.c_str(), (KInt)kval, str_end_string.c_str());
		break;
	case Ken_value_type::kfloat:
		printf("%s%s%.15lf%s", str_prefix.c_str(), str_indent.c_str(), (KFloat)kval, str_end_string.c_str());
		break;
	case Ken_value_type::string:
		printf("%s%s%s%s", str_prefix.c_str(), str_indent.c_str(), ((KString)kval).c_str(), str_end_string.c_str());
		break;
	case Ken_value_type::list:
		print_klist(kval, str_prefix, indent_size, "sublist");
		break;
	case Ken_value_type::dict:
		print_kdict(kval, str_prefix, indent_size, "subdict");
		break;
	case Ken_value_type::shape:
		print_kshape(kval, str_prefix, indent_size, "");
		break;
	case Ken_value_type::object:
		printf("%s%s0x%p%s", str_prefix.c_str(), str_indent.c_str(), (void*)kval, str_end_string.c_str());
		break;
	default:
		printf("%s%sunsupported type(%d)%s", str_prefix.c_str(), str_indent.c_str(), (int)enum_type, str_end_string.c_str());
		break;
	}
}

void print_klist(KaiList klist, std::string str_prefix, unsigned int indent_size, std::string str_obj_name) {
	std::string str_indent = std::string(indent_size, ' ');
	for (int list_idx = 0; list_idx < klist.size(); ++list_idx) {
		KaiValue temp_kval = klist[list_idx];
		printf("%s%s%s[%d] = ", str_prefix.c_str(), str_indent.c_str(), str_obj_name.c_str(), list_idx);
		if (temp_kval.type() == Ken_value_type::list) {
			printf("(KaiList)\n");
			print_kvalue(temp_kval.type(), temp_kval, str_prefix, indent_size + NIGHTLY_UTILS_INDENT_UNIT_SIZE, "\n");
		}
		else if (temp_kval.type() == Ken_value_type::dict) {
			printf("(KaiDict)\n");
			print_kvalue(temp_kval.type(), temp_kval, str_prefix, indent_size + NIGHTLY_UTILS_INDENT_UNIT_SIZE, "\n");
		}
		else if (temp_kval.type() == Ken_value_type::shape) {
			printf("(KaiShape)\n");
			print_kvalue(temp_kval.type(), temp_kval, str_prefix, indent_size + NIGHTLY_UTILS_INDENT_UNIT_SIZE, "\n");
		}
		else {
			print_kvalue(temp_kval.type(), temp_kval, "", 0, "\n");
		}
	}
}

void print_kdict(KaiDict kdict, std::string str_prefix, unsigned int indent_size, std::string str_obj_name) {
	std::string str_indent = std::string(indent_size, ' ');
	for (KaiDictIter it = kdict.begin(); it != kdict.end(); ++it) {
		printf("%s%s%s[\"%s\"] = ", str_prefix.c_str(), str_indent.c_str(), str_obj_name.c_str(), ((KString)(it->first)).c_str());

		// Temporary settings
		if (it->first == KString("data")) {
			printf("(omitted)\n");
			continue;
		}

		if (it->second.type() == Ken_value_type::list) {
			printf("(KaiList)\n");
			print_kvalue(it->second.type(), it->second, str_prefix, indent_size + NIGHTLY_UTILS_INDENT_UNIT_SIZE, "\n");
		}
		else if (it->second.type() == Ken_value_type::dict) {
			printf("(KaiDict)\n");
			print_kvalue(it->second.type(), it->second, str_prefix, indent_size + NIGHTLY_UTILS_INDENT_UNIT_SIZE, "\n");
		}
		else if (it->second.type() == Ken_value_type::shape) {
			printf("(KaiShape)\n");
			print_kvalue(it->second.type(), it->second, str_prefix, indent_size + NIGHTLY_UTILS_INDENT_UNIT_SIZE, "\n");
		}
		else {
			print_kvalue(it->second.type(), it->second, "", 0, "\n");
		}
	}
}

void print_kshape(KaiShape kshape, std::string str_prefix, unsigned int indent_size, std::string str_obj_name) {
	std::string str_indent = std::string(indent_size, ' ');
	//for (int list_idx=0; list_idx<kshape.size(); ++list_idx) {
	//	KInt temp_kint = kshape[list_idx];
	//	printf("%s%s%s[%d] = ", str_prefix.c_str(), str_indent.c_str(), str_obj_name.c_str(), list_idx);
	//	print_kvalue(KaiValueType::kint, temp_kint, "", 0, "\n");
	//}
	if (str_obj_name != "") {
		str_obj_name += " = ";
	}
	printf("%s%s%s{", str_prefix.c_str(), str_indent.c_str(), str_obj_name.c_str());
	for (int list_idx = 0; list_idx < kshape.size(); ++list_idx) {
		printf(" %lld", kshape[list_idx]);
	}
	printf(" }\n");
}

template <class T>
void print_karray(KaiArray<T> karray, std::string str_prefix, unsigned int indent_size, std::string str_obj_name, unsigned int print_step = 0) {
	KInt total_size   = karray.total_size();
	KInt dim          = karray.dim();
	KString type_name = karray.element_type_name();

	std::string str_indent = std::string(indent_size, ' ');
	
	printf("%s%s%s<%s>", str_prefix.c_str(), str_indent.c_str(), str_obj_name.c_str(), type_name.c_str());

	for (KInt dim_idx=0; dim_idx<dim; ++dim_idx)
		printf("[%lld]", karray.axis_size(dim_idx));
	printf(" = ");

	char zFmtStr[64] = {0, };
	if (type_name == KString("KInt"))
		sprintf(zFmtStr, "%%lld ");
	else if (type_name == KString("KFloat"))
		//sprintf(zFmtStr, "%%9.6f ");	// %9.6f format is KaiArray<T>::dump() style
		sprintf(zFmtStr, "%%.6lf ");
	else {
		printf("{ Unknown type is not supported }\n");
		return;
	}

	// Get a pointer of data
	KaiArray<T> host_arr;
	T* ptr = NULL;
	if (karray.is_cuda()) {
		host_arr = karray.to_host();
		ptr = host_arr.data_ptr();
		printf("<device> ");
	}
	else
		ptr = karray.data_ptr();

	if (print_step >= total_size)
		print_step = 0;

	if (print_step == 0) {
		printf("{ ");
		for (int array_idx=0; array_idx<total_size; ++array_idx)
			printf(zFmtStr, *ptr++);
		printf("}\n");
	}
	else {
		std::string array_indent = std::string(str_prefix.length() + indent_size + 4, ' ');

		printf("{\n");
		for (int array_idx=0; array_idx<total_size; ++array_idx) {
			if (array_idx % print_step == 0)
				printf("%s", array_indent.c_str());

			printf(zFmtStr, *ptr++);

			if ((array_idx+1) % print_step == 0 || (array_idx+1) == total_size)
				printf("\n");
		}
		printf("%s}\n", std::string(str_prefix.length() + indent_size, ' ').c_str());
	}
}

void print_karray_kint(void* pkarray, std::string str_prefix, unsigned int indent_size, std::string str_obj_name, unsigned int print_step) {
	print_karray<KInt>(*(KaiArray<KInt>*)pkarray, str_prefix, indent_size, str_obj_name, print_step);
}

void print_karray_kfloat(void* pkarray, std::string str_prefix, unsigned int indent_size, std::string str_obj_name, unsigned int print_step) {
	print_karray<KFloat>(*(KaiArray<KFloat>*)pkarray, str_prefix, indent_size, str_obj_name, print_step);
}

void cv_load_and_resize_image(KFloat* pDstImageData, KString sImageFilename, KInt rows, KInt cols, KInt channels) {
	cv::Mat img;

	// Load an image
	if (channels == 3)
		img = cv::imread(sImageFilename, cv::IMREAD_COLOR);
	else if (channels == 1)
		img = cv::imread(sImageFilename, cv::IMREAD_GRAYSCALE);
	else {
		printf("error: %s(%u): channels(%lld) is invalid.\n", __FUNCTION__, __LINE__, channels);
		THROW(KERR_FILE_OPEN_FAILURE);
	}

	// Resize
	cv::resize(img, img, cv::Size((int)rows, (int)cols), 0, 0, cv::INTER_AREA);

	// Convert to 
	cv::Mat img_float;
	img.convertTo(img_float, CV_32FC3);

	// Calculate the pixel count
	int nPixels = rows * cols * channels;

	// Copy the image data to target memory block
	KFloat* p = (KFloat*)(img_float.data);

	for (int i=0; i<nPixels; ++i)
		*pDstImageData++ = *p++;
}
