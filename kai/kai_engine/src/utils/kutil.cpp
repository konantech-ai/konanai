/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kutil.h"
#include "../math/kmath.h"

//hs.cho
// 추후 boost::filesystem  사용
#ifdef KAI2021_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include "../nightly/findfirst.h"
#define _mkdir(filepath)  mkdir(filepath, 0777)
//inline int localtime_s(struct tm *tmp, const time_t *timer){ struct tm* tmp2=localtime(timer); memcpy(tmp,tmp2,sizeof(*tmp2));return 0;}
inline struct tm* localtime_s(struct tm* tmp, const time_t* timer) { localtime_r(timer, tmp); return 0; }
#define strcpy_s(a,b,c) !strncpy(a,c,b)
#define strtok_s strtok_r
#endif

// Check the C++ compiler, STL version, and OS type.
#if (defined(_MSC_VER) && (_MSVC_LANG >= 201703L || _HAS_CXX17)) || (defined(__GNUC__) && (__cplusplus >= 201703L))
	// ISO C++17 Standard (/std:c++17 or -std=c++17)
#include <filesystem>
namespace fs = std::filesystem;
#else
//hs.cho
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem> // C++14
namespace fs = std::experimental::filesystem;
#endif



#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

KUtil kutil;

//hs.cho
//#ifdef KAI2021_WINDOWS
void KUtil::mkdir(KString path) {
#ifdef KAI2021_WINDOWS
	std::replace(path.begin(), path.end(), '/', '\\');
#endif
	::_mkdir(path.c_str());
	//printf("\"%s\" has been created.\n", path.c_str());
}

KaiList KUtil::list_dir(KString path) {
	KaiList list;

//hs.cho
//#ifdef KAI2021_WINDOWS
	path = path + "/*";
#ifdef KAI2021_WINDOWS
	std::replace(path.begin(), path.end(), '/', '\\');
#endif
	_finddata_t fd;
	intptr_t hObject;
	KInt result = 1;

	hObject = _findfirst(path.c_str(), &fd);

	if (hObject == -1) return list;

	while (result != -1) {
		if (fd.name[0] != '.') list.push_back(fd.name);
		result = _findnext(hObject, &fd);

	}
#ifdef NORANDOM
	std::sort(list.begin(), list.end(), [](const KaiValue& left, const KaiValue& right) {
		return (strcmp(left.desc().c_str(), right.desc().c_str()) < 0) ? true : false;
		});
#endif	
	_findclose(hObject);

	return list;
//#else
//	DIR* dir;
//	struct dirent* ent;
//	if ((dir = opendir(path.c_str())) != NULL) {
//		while ((ent = readdir(dir)) != NULL) {
//			if (ent->d_name[0] == '.') continue;
//			list.push_back(ent->d_name);
//			//logger.Print("%s", ent->d_name);
//		}
//		closedir(dir);
//	}
//	else {
//		throw KaiException(KERR_ASSERT);
//	}
//
//	return list;
//#endif
}

////hs.cho
////#ifdef KAI2021_WINDOWS
//#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
//#include <experimental/filesystem> // C++14
//namespace fs = std::experimental::filesystem;

void KUtil::remove_all(KString path) {
	fs::remove_all(path.c_str());
}
//#else
//bool KUtil::remove_all(KString path) {
//	throw KaiException(KERR_UNIMPLEMENTED_YET);
//	return false;
//}
//#endif

#ifdef KAI2021_WINDOWS
FILE* KUtil::fopen(KString filepath, KString mode) {
	FILE* fid = NULL;

	std::replace(filepath.begin(), filepath.end(), '/', '\\');

	if (fopen_s(&fid, filepath.c_str(), mode.c_str()) != 0) {
		throw KaiException(KERR_FILE_OPEN_FAILURE, filepath, mode);
	}
	return fid;
}
#else
FILE* KUtil::fopen(KString filepath, KString mode) {
	return ::fopen(filepath.c_str(), mode.c_str());
}
#endif

KString KUtil::get_timestamp(time_t tm) {
	struct tm timeinfo;
	char buffer[80];

	if (localtime_s(&timeinfo , &tm)) throw KaiException(KERR_FAILURE_ON_GET_LOCALTIME);


	strftime(buffer, 80, "%D %T", &timeinfo);
	return KString(buffer);
}

void KUtil::load_jpeg_image_pixels(KFloat* pBuf, KString filepath, KaiShape data_shape) {
//hs.cho
#ifdef KAI2021_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#endif
	cv::Mat img = cv::imread(filepath, 1);
	cv::resize(img, img, cv::Size((int)data_shape[0], (int)data_shape[1]), 0, 0, cv::INTER_AREA); //hs.cho cubic interpolation result can vary in some versions of OPENCV

	int chn = (int)img.channels();

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			cv::Vec3b intensity = img.at<cv::Vec3b>(j, i);
			for (int k = 0; k < chn; k++) {
				KFloat dump = (KFloat)intensity.val[k];;
				*pBuf++ = (KFloat)intensity.val[k];
			}
		}
	}
}

vector<vector<string>> KUtil::load_csv(KString filepath, KaiList* pHead) {
//hs.cho
#ifdef KAI2021_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#endif

	ifstream infile(filepath);

	if (infile.fail()) throw KaiException(KERR_FILE_OPEN_FAILURE, filepath);

	vector<vector<string>> rows;

	string line;
	char buffer[1024];
	char* context;

	getline(infile, line);
	if (pHead) {
		if (strcpy_s(buffer, 1024, line.c_str())) throw KaiException(KERR_ASSERT);
		char* token = strtok_s(buffer, ",", &context);
		while (token) {
			(*pHead).push_back(token);
			token = strtok_s(NULL, ",", &context);
		}
	}

	while (std::getline(infile, line)) {
		if (line[0] == ',') {
			line = "0" + line;
		}
		if (line[line.length() - 1] == ',') {
			line = line + "0";;
		}

		std::size_t pos = line.find(",,");
		while (pos != std::string::npos) {
			line = line.substr(0, pos + 1) + "0" + line.substr(pos + 1);
			pos = line.find(",,");
		}

		if (strcpy_s(buffer, 1024, line.c_str())) throw KaiException(KERR_ASSERT);
		char* token = strtok_s(buffer, ",", &context);
		vector<string> row;
		while (token) {
			row.push_back(token);
			token = strtok_s(NULL, ",", &context);
		}
		rows.push_back(row);
	}

	infile.close();

	return rows;
}

KaiValue KUtil::seek_dict(KaiDict dict, KString sKey, KaiValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) return kDefaultValue;
	else return dict[sKey];
}

KaiValue KUtil::seek_set_dict(KaiDict& dict, KString sKey, KaiValue kDefaultValue) {
	if (dict.find(sKey) == dict.end()) {
		dict[sKey] = kDefaultValue;
		return kDefaultValue;
	}
	else {
		return dict[sKey];
	}
}

void KUtil::save_value(FILE* fid, KaiValue value) {
	save_int(fid, (KInt)value.type());

	switch (value.type()) {
	case Ken_value_type::none:
		break;
	case Ken_value_type::kint:
		save_int(fid, (KInt)value);
		break;
	case Ken_value_type::kfloat:
		save_float(fid, (KFloat)value);
		break;
	case Ken_value_type::string:
		save_str(fid, (KString)value);
		break;
	case Ken_value_type::list:
		save_list(fid, (KaiList)value);
		break;
	case Ken_value_type::dict:
		save_dict(fid, (KaiDict)value);
		break;
	case Ken_value_type::object:
	{
		KHObject hObject = value;
		Ken_object_type obj_type = hObject->get_type();
		save_int(fid, (KInt)obj_type);
		if (obj_type == Ken_object_type::narray) {
			KaiArray<KInt> arr = NARRAY(value);
			save_array(fid, arr);
		}
		else if (obj_type == Ken_object_type::farray) {
			KaiArray<KFloat> arr = FARRAY(value);
			save_array(fid, arr);
		}
		else {
			throw KaiException(KERR_UNSUPPORTED_HANLE_TYPE_IN_SAVE);
		}
	}
	}
}

KaiValue KUtil::read_value(FILE* fid) {
	KaiValue value;
	enum class Ken_value_type type = (enum class Ken_value_type) read_int(fid);
	enum class Ken_object_type obj_type;

	switch (type) {
	case Ken_value_type::none:
		break;
	case Ken_value_type::kint:
		value = read_int(fid);
		break;
	case Ken_value_type::kfloat:
		value = read_float(fid);
		break;
	case Ken_value_type::string:
		value = read_str(fid);
		break;
	case Ken_value_type::list:
		value = read_list(fid);
		break;
	case Ken_value_type::dict:
		value = read_dict(fid);
		break;
	case Ken_value_type::object:
		obj_type = (enum class Ken_object_type) read_int(fid);
		switch (obj_type) {
		case Ken_object_type::narray:
		{
			KaiArray<KInt> arr = read_narray(fid);
			value = arr.get_core();
			/*
			KaiArray<KInt> array = read_n64array(fid);
			KaiArray<KInt>* pArray = new KaiArray<KInt>();
			*pArray = array;
			value = KArray(array_type::at_int, (KHArray)pArray);
			*/
		}
		break;
		case Ken_object_type::farray:
		{
			KaiArray<KFloat> arr = read_farray(fid);
			value = arr.get_core();
		}
		break;
		}
	}

	return value;
}

void KUtil::save_int(FILE* fid, KInt dat) {
	if (fwrite(&dat, sizeof(KInt), 1, fid) != 1) throw KaiException(KERR_FILE_SAVE_INT_FAILURE);
}

void KUtil::save_float(FILE* fid, KFloat dat) {
	if (fwrite(&dat, sizeof(KFloat), 1, fid) != 1) throw KaiException(KERR_FILE_SAVE_FLOAT_FAILURE);
}

void KUtil::save_str(FILE* fid, KString dat) {
	KInt length = (KInt)dat.length();
	save_int(fid, length);
	if (fwrite(dat.c_str(), sizeof(char), length, fid) != length) throw KaiException(KERR_FILE_SAVE_STRING_FAILURE);
}

void KUtil::save_list(FILE* fid, KaiList list) {
	save_int(fid, list.size());
	for (auto it = list.begin(); it != list.end(); it++) {
		save_value(fid, *it);
	}
}

void KUtil::save_dict(FILE* fid, KaiDict dict) {
	save_int(fid, dict.size());
	for (auto it = dict.begin(); it != dict.end(); it++) {
		save_str(fid, it->first);
		save_value(fid, it->second);
	}
}

void KUtil::save_shape(FILE* fid, KaiShape shape) {
	KInt dim = shape.size();
	save_int(fid, dim);
	for (KInt n = 0; n < dim; n++) {
		save_int(fid, shape[n]);
	}
}

KInt KUtil::read_int(FILE* fid) {
	KInt dat;
	if (fread(&dat, sizeof(KInt), 1, fid) != 1) throw KaiException(KERR_FILE_READ_INT_FAILURE);
	return dat;
}

KFloat KUtil::read_float(FILE* fid) {
	KFloat dat;
	if (fread(&dat, sizeof(KFloat), 1, fid) != 1) throw KaiException(KERR_FILE_READ_FLOAT_FAILURE);
	return dat;
}

KString KUtil::read_str(FILE* fid) {
	KInt length = read_int(fid);
	char* piece = new char[length + 1];
	if (fread(piece, sizeof(char), length, fid) != length) throw KaiException(KERR_FILE_READ_STRING_FAILURE);
	piece[length] = 0;
	KString str = piece;
	delete piece;
	return str;
}

KaiList KUtil::read_list(FILE* fid) {
	KaiList list;
	KInt size = read_int(fid);
	for (KInt n = 0; n < size; n++) {
		KaiValue value = read_value(fid);
		list.push_back(value);
	}
	return list;
}

KaiDict KUtil::read_dict(FILE* fid) {
	KaiDict dict;
	KInt size = read_int(fid);
	for (KInt n = 0; n < size; n++) {
		KString key = read_str(fid);
		KaiValue value = read_value(fid);
		dict[key] = value;
	}
	return dict;
}

void KUtil::read_shape(FILE* fid, KaiShape& shape) {
	KInt dim = read_int(fid);
	KInt ax_size[KAI_MAX_DIM];

	if (fread(ax_size, sizeof(KInt), dim, fid) != dim) throw KaiException(KERR_ASSERT);

	shape = KaiShape(dim, ax_size);
}

template <class T> void KUtil::save_array(FILE* fid, KaiArray<T> arr) {
	save_shape(fid, arr.shape());
	arr = arr.to_host();
	KInt size = arr.total_size();
	T* p = arr.data_ptr();
	if (fwrite(p, sizeof(T), size, fid) != size) throw KaiException(KERR_SAVE_ARRAY_FAILURE);
}

template void KUtil::save_array(FILE* fid, KaiArray<KInt> arr);
template void KUtil::save_array(FILE* fid, KaiArray<KFloat> arr);

KaiArray<KInt> KUtil::read_narray(FILE* fid) {
	KaiShape shape;
	read_shape(fid, shape);
	KaiArray<KInt> arr(shape);
	KInt size = arr.total_size();
	KInt* p = arr.data_ptr();
	if (fread(p, sizeof(KInt), size, fid) != size) throw KaiException(KERR_READ_ARRAY_FAILURE);
	return arr;
}

KaiArray<KFloat> KUtil::read_farray(FILE* fid) {
	KaiShape shape;
	read_shape(fid, shape);
	KaiArray<KFloat> arr(shape);
	KInt size = arr.total_size();
	KFloat* p = arr.data_ptr();
	if (fread(p, sizeof(KFloat), size, fid) != size) throw KaiException(KERR_READ_ARRAY_FAILURE);
	return arr;
}
