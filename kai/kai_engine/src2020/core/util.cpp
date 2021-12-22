/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "common.h"
#include "util.h"
#include "array.h"
#include "value.h"
#include "dim.h"
#include "idx.h"
#include "host_math.h"
#include "log.h"
#include "../cuda/cuda_note.h"

#include <fstream>

#include <string.h>
#include <stdio.h>
#include <cstdarg>

//hs.cho
#ifdef KAI2021_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include "../../src/nightly/findfirst.h"
#define _mkdir(filepath)  mkdir(filepath, 0777)
//inline int localtime_s(struct tm *tmp, const time_t *timer){ struct tm* tmp2=localtime(timer); memcpy(tmp,tmp2,sizeof(*tmp2));return 0;}
inline struct tm* localtime_s(struct tm* tmp, const time_t* timer) { localtime_r(timer, tmp); return 0; }
#define strcpy_s(a,b,c) !strncpy(a,c,b)
#define strtok_s strtok_r

#endif

vector<vector<string>> Util::load_csv(string filepath, vector<string>* pHead) {
#ifdef KAI2021_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#endif

	ifstream infile(filepath);

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
		if (strcpy_s(buffer, 1024, line.c_str())) throw KaiException(KERR_ASSERT);
		char* token = strtok_s(buffer, ",", &context);
		vector<string> row;
		while (token) {
			row.push_back(token);
			token = strtok_s(NULL, ",", &context);
		}
		rows.push_back(row);
	}

	return rows;
}
//hs.cho
//#ifdef KAI2021_WINDOWS
void Util::mkdir(string path) {
	string win_path = path;
#ifdef KAI_2021_WINDOWS
	std::replace(win_path.begin(), win_path.end(), '/', '\\');
#endif
	::_mkdir(win_path.c_str());
}
//#else
//void Util::mkdir(string path) {
//	::mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
//}
//#endif

//hs.cho
vector<string> Util::list_dir(string path) {
	vector<string> list;

//#ifdef KAI2021_WINDOWS
	path = path + "/*";
#ifdef KAI2021_WINDOWS
	std::replace(path.begin(), path.end(), '/', '\\');
#endif
	_finddata_t fd;
	intptr_t hObject;
	int64 result = 1;

	hObject = _findfirst(path.c_str(), &fd);

	if (hObject == -1) return list;

	while (result != -1) {
		if (fd.name[0] != '.') list.push_back(fd.name);
		result = _findnext(hObject, &fd);

	}

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


//hs.cho
//#ifdef KAI2021_WINDOWS
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem> // C++14
namespace fs = std::experimental::filesystem;

bool Util::remove_all(string path) {
	fs::remove_all(path.c_str());
	return true;
	/*
	vector<string> file_list, folder_list;

	string search_path = path + "/*";

	std::replace(search_path.begin(), search_path.end(), '/', '\\');

	_finddata_t fd;
	intptr_t hObject;
	int64 result = 1;
	int64 cnt_lookup = 0;

	hObject = _findfirst(search_path.c_str(), &fd);

	if (hObject == -1) return true;

	while (result != -1) {
		if (fd.name == "." || fd.name == ".."); // skip current folder and parent folder
		else {
			cnt_lookup++;
			if (fd.attrib & (_A_RDONLY | _A_HIDDEN | _A_SYSTEM | _A_ARCH)); // cannot remove
			else if (fd.attrib | _A_SUBDIR) {
				if (remove_dir(path + "/" + fd.name)) folder_list.push_back(fd.name);
			}
			else file_list.push_back(fd.name);
		}
		result = _findnext(hObject, &fd);
	}

	_findclose(hObject);

	for (auto it = file_list.begin(); it != file_list.end(); it++) {
		string file_path = path + "/" + *it;
		::remove(file_path.c_str());
	}

	for (auto it = folder_list.begin(); it != folder_list.end(); it++) {
		string folder_path = path + "/" + *it;
		::remove(folder_path.c_str());	 // 디렉토리 삭제에도 적용되면 file_list와 folder_list 합병, 안 되면 디렉토리 삭제 방법 찾아볼 것
	}

	return cnt_lookup == file_list.size() + folder_list.size();
	*/
}
//#else
//bool Util::remove_all(string path) {
//	throw KaiException(KERR_UNIMPLEMENTED_YET);
//	return false;
//}
//#endif

Array<float> Util::load_image_pixels(string filepath, Shape resolution, Shape input_shape) {
	throw KaiException(KERR_ASSERT);
	return Array<float>();
}
//hs.cho
#ifdef KAI2021_WINDOWS
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

void Util::load_jpeg_image_pixels(float* pBuf, string filepath, Shape data_shape) {
	std::replace(filepath.begin(), filepath.end(), '/', '\\');

	cv::Mat img = cv::imread(filepath, 1);
	cv::resize(img, img, cv::Size((int)data_shape[0], (int)data_shape[1]), 0, 0, cv::INTER_CUBIC);

	int chn = (int) img.channels();

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			cv::Vec3b intensity = img.at<cv::Vec3b>(j, i);
			for (int k = 0; k < chn; k++) {
				*pBuf++ = (float)intensity.val[k];
			}
		}
	}
}

void Util::draw_images_horz(bool show, string folder, Array<float> arr, Shape shape, int ratio, int gap) {
	if (shape.size() == 2) shape = shape.append(1);

	assert(shape.size() == 3);

	assert(arr.total_size() % shape.total_size() == 0);

	int mb_size = (int) arr.axis_size(0);
	int img_count = (int)(arr.total_size() / shape.total_size());

	int img_rows = 1;
	int img_cols = mb_size;

	if (mb_size != img_count) {
		assert(img_count % mb_size == 0);

		img_rows = mb_size;
		img_cols = img_count / mb_size;
	}

	shape = shape.add_front(img_rows, img_cols);

	arr = arr.reshape(shape).to_host();

	int piece_height = (int) shape[-3];
	int piece_width = (int) shape[-2];
	int chn = (int) shape[-1];

	int image_height = piece_height * ratio * img_rows + gap * (img_rows - 1);
	int image_width  = piece_width * ratio * img_cols + gap * (img_cols - 1);

	cv::Mat img(image_height, image_width, (chn == 3) ? CV_8UC3 : CV_8UC1);

	cv::Vec3b color_pixel;

	for (int nr = 0; nr < image_height; nr++) {
		for (int nc = 0; nc < image_width; nc++) {
			int pr = nr % (piece_height * ratio + gap) / ratio;
			int pc = nc % (piece_width * ratio + gap) / ratio;

			if (pr < piece_height && pc < piece_width) {
				int ar = nr / (piece_height * ratio + gap);
				int ac = nc / (piece_width * ratio + gap);

				if (chn == 3) {
					color_pixel[0] = (unsigned char)(arr[Idx(ar, ac, pr, pc, 0)] * 128 + 128);;
					color_pixel[1] = (unsigned char)(arr[Idx(ar, ac, pr, pc, 1)] * 128 + 128);;
					color_pixel[2] = (unsigned char)(arr[Idx(ar, ac, pr, pc, 2)] * 128 + 128);;
					img.at<cv::Vec3b>(nr, nc) = color_pixel;
				}
				else {
					img.at<uchar>(nr, nc) = (unsigned char)(arr[Idx(ar, ac, pr, pc, 0)] * 128 + 128);
				}
			}
			else {
				if (chn == 3) {
					color_pixel[0] = 0;
					color_pixel[1] = 0;
					color_pixel[2] = 0;
					img.at<cv::Vec3b>(nr, nc) = color_pixel;
				}
				else {
					img.at<uchar>(nr, nc) = 0;
				}
			}
		}
	}

	/*
	int ybase = 0;
	for (int nr = 0; nr < img_rows; nr++) {
		int xbase = 0;
		for (int nc = 0; nc < img_cols; nc++) {
			for (int i = 0; i < piece_height; i++) {
				for (int j = 0; j < piece_width; j++) {
					if (chn == 3) {
						color_pixel[0] = (unsigned char)(arr[Idx(nr, nc, i, j, 0)] * 128 + 128);
						color_pixel[1] = (unsigned char)(arr[Idx(nr, nc, i, j, 1)] * 128 + 128);
						color_pixel[2] = (unsigned char)(arr[Idx(nr, nc, i, j, 2)] * 128 + 128);
					}
					else {
						mono_pixel = (unsigned char)(arr[Idx(nr, nc, i, j, 0)] * 128 + 128);
					}

					int row = j * ratio + ybase;
					int col = i * ratio + xbase;
					for (int r = 0; r < ratio; r++) {
						for (int c = 0; c < ratio; c++) {
							if (chn == 3) {
								img.at<cv::Vec3b>(row + r, col + c) = color_pixel;
							}
							else {
								img.at<uchar>(row + r, col + c) = mono_pixel;
							}
						}
					}
				}
			}
			xbase += piece_width * ratio + gap;
		}
		ybase += piece_height * ratio + gap;
	}
	*/

	if (folder != "") {
		string savepath = KArgs::image_root;
		Util::mkdir(savepath);
		savepath += folder;
		Util::mkdir(savepath);
		savepath += "/" + get_timestamp() + ".jpg";
		std::replace(savepath.begin(), savepath.end(), '/', '\\');
		cv::imwrite(savepath, img);
	}

	if (show) {
		cv::imshow("name", img);
		cv::waitKey(10000);
		cv::destroyWindow("name");
	}
}
#else
#endif

void Util::load_kodell_dump_file(string filepath, Array<float>& xs, Array<float>& ys, vector<string>* pTargetNames, int cat_cnt) {
	FILE* fid = Util::fopen(filepath.c_str(), "rb");
	xs = ms_load_array_from_kodell_dump_file(fid);
	ys = ms_load_array_from_kodell_dump_file(fid);
	int read_cat_cnt = read_int(fid);
	if (read_cat_cnt != cat_cnt) throw KaiException(KERR_ASSERT);
	for (int n = 0; n < cat_cnt; n++) {
		ms_load_names_from_kodell_dump_file(pTargetNames++, fid);
	}
	fclose(fid);
}

Array<float> Util::ms_load_array_from_kodell_dump_file(FILE* fid) {
	int64 shape_len = read_i64(fid);
	int64 size[KAI_MAX_DIM];
	if (fread(size, sizeof(int64), shape_len, fid) != shape_len) throw KaiException(KERR_ASSERT);
	Shape shape(size, shape_len);
	Array<float> arr(shape);
	int64 data_size = shape.total_size();
	if (fread(arr.data_ptr(), sizeof(float), data_size, fid) != data_size) throw KaiException(KERR_ASSERT);
	return arr;
}

void Util::ms_load_names_from_kodell_dump_file(vector<string>* pTargetNames, FILE* fid) {
	int64 name_cnt = read_i64(fid);
	for (int64 n = 0; n < name_cnt; n++) {
		char name_buf[256];
		int name_len = read_int(fid);
		assert(name_len < 256);
		if ((int)fread(name_buf, sizeof(char), name_len, fid) != name_len) throw KaiException(KERR_ASSERT);
		name_buf[name_len] = 0;
		pTargetNames->push_back(name_buf);
	}
}

void Util::save_kodell_dump_file(string filepath, Array<float>& xs, Array<float>& ys, vector<string>* pTargetNames, int cat_cnt) {
	FILE* fid = Util::fopen(filepath.c_str(), "wb");
	ms_save_array_to_kodell_dump_file(fid, xs);
	ms_save_array_to_kodell_dump_file(fid, ys);
	save_int(fid, cat_cnt);
	for (int n = 0; n < cat_cnt; n++) {
		ms_save_names_to_kodell_dump_file(fid, pTargetNames++);
	}
	fclose(fid);
}

void Util::ms_save_array_to_kodell_dump_file(FILE* fid, Array<float> arr) {
	Shape shape = arr.shape();
	int64 shape_len = shape.size();
	int64 size[KAI_MAX_DIM];
	int64 data_size = shape.total_size();

	for (int64 n = 0; n < shape_len; n++) size[n] = (int) shape[n];

	save_int64(fid, shape_len);
	if (fwrite(size, sizeof(int64), shape_len, fid) != shape_len) throw KaiException(KERR_ASSERT);
	if (fwrite(arr.data_ptr(), sizeof(float), data_size, fid) != data_size) throw KaiException(KERR_ASSERT);
}

void Util::ms_save_names_to_kodell_dump_file(FILE* fid, vector<string>* pTargetNames) {
	int64 name_cnt = pTargetNames->size();
	save_int64(fid, name_cnt);
	for (int64 n = 0; n < name_cnt; n++) {
		string name = pTargetNames->at(n);
		int name_len = (int) name.length();
		assert(name_len < 256);
		save_int(fid, name_len);
		if (fwrite(name.c_str(), sizeof(char), name_len, fid) != name_len) throw KaiException(KERR_ASSERT);
	}
}

void Util::print(string title, Shape shape) {
	logger.Print("[%s] %s", title.c_str(), shape.desc().c_str());
}

void Util::read_vv_int(FILE* fid, vector<vector<int>>& sents) {
	int cnt = read_int(fid);
	for (int n = 0; n < cnt; n++) {
		vector<int> sent;
		int leng = read_int(fid);
		for (int m = 0; m < leng; m++) {
			sent.push_back(read_int(fid));
		}
		sents.push_back(sent);
	}
}

void Util::read_vv_i64(FILE* fid, vector<vector<int64>>& sents) {
	int64 cnt = read_i64(fid);
	for (int64 n = 0; n < cnt; n++) {
		vector<int64> sent;
		int64 leng = read_i64(fid);
		for (int64 m = 0; m < leng; m++) {
			sent.push_back(read_i64(fid));
		}
		sents.push_back(sent);
	}
}

void Util::read_v_int(FILE* fid, vector<int>& words) {
	int leng = read_int(fid);
	for (int m = 0; m < leng; m++) {
		words.push_back(read_int(fid));
	}
}

void Util::read_v_i64(FILE* fid, vector<int64>& words) {
	int64 leng = read_i64(fid);
	for (int64 m = 0; m < leng; m++) {
		words.push_back(read_i64(fid));
	}
}

void Util::read_map_si(FILE* fid, map<string, int>& dic) {
	int cnt = read_int(fid);
	for (int n = 0; n < cnt; n++) {
		string key = read_str(fid);
		int value = read_int(fid);
		dic[key] = value;
	}
}

void Util::read_map_si64(FILE* fid, map<string, int64>& dic) {
	int64 cnt = read_i64(fid);
	for (int64 n = 0; n < cnt; n++) {
		string key = read_str(fid);
		int64 value = read_i64(fid);
		dic[key] = value;
	}
}

void Util::read_map_is(FILE* fid, map<int, string>& dic) {
	int cnt = read_int(fid);
	for (int n = 0; n < cnt; n++) {
		int key = read_int(fid);
		string value = read_str(fid);
		dic[key] = value;
	}
}

void Util::read_map_i64s(FILE* fid, map<int64, string>& dic) {
	int64 cnt = read_i64(fid);
	for (int64 n = 0; n < cnt; n++) {
		int64 key = read_i64(fid);
		string value = read_str(fid);
		dic[key] = value;
	}
}

void Util::read_map_ii(FILE* fid, map<int, int>& dic) {
	int cnt = read_int(fid);
	for (int n = 0; n < cnt; n++) {
		int key = read_int(fid);
		int value = read_int(fid);
		dic[key] = value;
	}
}

void Util::read_map_i64i64(FILE* fid, map<int64, int64>& dic) {
	int64 cnt = read_i64(fid);
	for (int64 n = 0; n < cnt; n++) {
		int64 key = read_i64(fid);
		int64 value = read_i64(fid);
		dic[key] = value;
	}
}

void Util::save_bool(FILE* fid, bool dat) {
	if ((int)fwrite(&dat, sizeof(bool), 1, fid) != 1) throw KaiException(KERR_ASSERT);
}

void Util::save_int(FILE* fid, int dat) {
	if ((int)fwrite(&dat, sizeof(int), 1, fid) != 1) throw KaiException(KERR_ASSERT);
}

void Util::save_int64(FILE* fid, int64 dat) {
	if ((int)fwrite(&dat, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
}

void Util::save_float(FILE* fid, float dat) {
	if ((int)fwrite(&dat, sizeof(float), 1, fid) != 1) throw KaiException(KERR_ASSERT);
}

void Util::save_str(FILE* fid, string dat) {
	int length = (int) dat.length();
	save_int(fid, length);
	if ((int) fwrite(dat.c_str(), sizeof(char), length, fid) != length) throw KaiException(KERR_ASSERT);
}

void Util::save_list(FILE* fid, List dat) {
	save_int64(fid, (int64)dat.size());
	for (List::iterator it = dat.begin(); it != dat.end(); it++) {
		Value value = *it;
		save_value(fid, value);
	}
}

void Util::save_dict(FILE* fid, Dict dat) {
	save_int64(fid, (int64)dat.size());
	for (Dict::iterator it = dat.begin(); it != dat.end(); it++) {
		save_str(fid, it->first);
		save_value(fid, it->second);
	}
}

void Util::save_shape(FILE* fid, Shape shape) {
	int64 dim = shape.size();
	save_int64(fid, dim);
	for (int64 n = 0; n < dim; n++) {
		save_int64(fid, shape[n]);
	}
}

void Util::save_barray(FILE* fid, Array<bool> dat) {
	save_shape(fid, dat.shape());
	dat = dat.to_host();
	int64 size = dat.total_size();
	bool* p = dat.data_ptr();
	if (fwrite(p, sizeof(bool), size, fid) != size) throw KaiException(KERR_ASSERT);
}

void Util::save_farray(FILE* fid, Array<float> dat) {
	save_shape(fid, dat.shape());
	dat = dat.to_host();
	int64 size = dat.total_size();
	float* p = dat.data_ptr();
	if (fwrite(p, sizeof(float), size, fid) != size) throw KaiException(KERR_ASSERT);
}

void Util::save_narray(FILE* fid, Array<int> dat) {
	save_shape(fid, dat.shape());
	dat = dat.to_host();
	int64 size = dat.total_size();
	int* p = dat.data_ptr();
	if (fwrite(p, sizeof(int), size, fid) != size) throw KaiException(KERR_ASSERT);
}

void Util::save_n64array(FILE* fid, Array<int64> dat) {
	save_shape(fid, dat.shape());
	dat = dat.to_host();
	int64 size = dat.total_size();
	int64* p = dat.data_ptr();
	if (fwrite(p, sizeof(int64), size, fid) != size) throw KaiException(KERR_ASSERT);
}

bool Util::read_bool(FILE* fid) {
	bool dat;
	if ((int)fread(&dat, sizeof(bool), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	return dat;
}

int Util::read_int(FILE* fid) {
	int dat;
	if ((int)fread(&dat, sizeof(int), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	return dat;
}

int64 Util::read_i64(FILE* fid) {
	int64 dat;
	if ((int)fread(&dat, sizeof(int64), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	return dat;
}

string Util::read_str(FILE* fid) {
	int length = read_int(fid);
	char* piece = new char[length + 1];
	if ((int)fread(piece, sizeof(char), length, fid) != length) throw KaiException(KERR_ASSERT);
	piece[length] = 0;
	string str = piece;
	delete piece;
	return str;
}

float Util::read_float(FILE* fid) {
	float dat;
	if ((int)fread(&dat, sizeof(float), 1, fid) != 1) throw KaiException(KERR_ASSERT);
	return dat;
}

List Util::read_list(FILE* fid) {
	List list;
	int64 size = read_i64(fid);
	for (int64 n = 0; n < size; n++) {
		Value value = read_value(fid);
		list.push_back(value);
	}
	return list;
}

Dict Util::read_dict(FILE* fid) {
	Dict dict;
	int64 size = read_i64(fid);
	for (int64 n = 0; n < size; n++) {
		string key = read_str(fid);
		Value value = read_value(fid);
		dict[key] = value;
	}
	return dict;
}

Shape Util::read_shape(FILE* fid) {
	Shape shape;
	int64 size = read_i64(fid);
	for (int64 n = 0; n < size; n++) {
		shape.append(read_i64(fid));
	}
	return shape;
}

Array<float> Util::read_farray(FILE* fid) {
	Shape shape = read_shape(fid);
	Array<float> arr(shape);
	int64 size = arr.total_size();
	float* p = arr.data_ptr();
	if (fread(p, sizeof(float), size, fid) != size) throw KaiException(KERR_ASSERT);
	return arr;
}

Array<bool> Util::read_barray(FILE* fid) {
	Shape shape = read_shape(fid);
	Array<bool> arr(shape);
	int64 size = arr.total_size();
	bool* p = arr.data_ptr();
	if (fread(p, sizeof(bool), size, fid) != size) throw KaiException(KERR_ASSERT);
	return arr;
}

Array<int> Util::read_narray(FILE* fid) {
	Shape shape = read_shape(fid);
	Array<int> arr(shape);
	int64 size = arr.total_size();
	int* p = arr.data_ptr();
	if (fread(p, sizeof(int), size, fid) != size) throw KaiException(KERR_ASSERT);
	return arr;
}

Array<int64> Util::read_n64array(FILE* fid) {
	Shape shape = read_shape(fid);
	Array<int64> arr(shape);
	int64 size = arr.total_size();
	int64* p = arr.data_ptr();
	if (fread(p, sizeof(int64), size, fid) != size) throw KaiException(KERR_ASSERT);
	return arr;
}

void Util::save_vv_int(FILE* fid, vector<vector<int>>& sents) {
	save_int(fid, (int)sents.size());
	for (vector<vector<int>>::iterator it = sents.begin(); it != sents.end(); it++) {
		vector<int>& sent = *it;
		save_int(fid, (int)sent.size());
		for (vector<int>::iterator it2 = sent.begin(); it2 != sent.end(); it2++) {
			save_int(fid, *it2);
		}
	}
}

void Util::save_vv_i64(FILE* fid, vector<vector<int64>>& sents) {
	save_int64(fid, (int64)sents.size());
	for (vector<vector<int64>>::iterator it = sents.begin(); it != sents.end(); it++) {
		vector<int64>& sent = *it;
		save_int64(fid, (int64)sent.size());
		for (vector<int64>::iterator it2 = sent.begin(); it2 != sent.end(); it2++) {
			save_int64(fid, *it2);
		}
	}
}

void Util::save_v_int(FILE* fid, vector<int>& words) {
	save_int(fid, (int)words.size());
	for (vector<int>::iterator it2 = words.begin(); it2 != words.end(); it2++) {
		save_int(fid, *it2);
	}
}

void Util::save_v_i64(FILE* fid, vector<int64>& words) {
	save_int64(fid, (int64)words.size());
	for (vector<int64>::iterator it2 = words.begin(); it2 != words.end(); it2++) {
		save_int64(fid, *it2);
	}
}

void Util::read_shape(FILE* fid, Shape& shape) {
	int ax_size[KAI_MAX_DIM];
	int dim = read_int(fid);

	if ((int)fread(ax_size, sizeof(int), dim, fid) != dim) throw KaiException(KERR_ASSERT);

	shape = Shape(ax_size, dim);
}

void Util::read_farray(FILE* fid, Array<float>& param) {
	Shape shape;

	read_shape(fid, shape);

	param = Array<float>(shape);

	int size = (int)param.total_size();
	float* p = param.data_ptr();

	if ((int)fread(p, sizeof(float), size, fid) != size) throw KaiException(KERR_ASSERT);
}

void Util::read_narray(FILE* fid, Array<int>& param) {
	Shape shape;

	read_shape(fid, shape);

	param = Array<int>(shape);

	int size = (int)param.total_size();
	int* p = param.data_ptr();

	if ((int)fread(p, sizeof(int), size, fid) != size) throw KaiException(KERR_ASSERT);
}

void Util::save_map_si(FILE* fid, map<string, int>& dic) {
	save_int(fid, (int)dic.size());
	for (map<string, int>::iterator it = dic.begin(); it != dic.end(); it++) {
		save_str(fid, it->first);
		save_int(fid, it->second);
	}
}

void Util::save_map_is(FILE* fid, map<int, string>& dic) {
	save_int(fid, (int)dic.size());
	for (map<int, string>::iterator it = dic.begin(); it != dic.end(); it++) {
		save_int(fid, it->first);
		save_str(fid, it->second);
	}
}

void Util::save_map_ii(FILE* fid, map<int, int>& dic) {
	save_int(fid, (int)dic.size());
	for (map<int, int>::iterator it = dic.begin(); it != dic.end(); it++) {
		save_int(fid, it->first);
		save_int(fid, it->second);
	}
}

void Util::save_map_si64(FILE* fid, map<string, int64>& dic) {
	save_int64(fid, (int64)dic.size());
	for (map<string, int64>::iterator it = dic.begin(); it != dic.end(); it++) {
		save_str(fid, it->first);
		save_int64(fid, it->second);
	}
}

void Util::save_map_i64s(FILE* fid, map<int64, string>& dic) {
	save_int64(fid, (int64)dic.size());
	for (map<int64, string>::iterator it = dic.begin(); it != dic.end(); it++) {
		save_int64(fid, it->first);
		save_str(fid, it->second);
	}
}

void Util::save_map_i64i64(FILE* fid, map<int64, int64>& dic) {
	save_int64(fid, (int64)dic.size());
	for (map<int64, int64>::iterator it = dic.begin(); it != dic.end(); it++) {
		save_int64(fid, it->first);
		save_int64(fid, it->second);
	}
}

void Util::read_dict(FILE* fid, Dict& dict) {
	int64 size = read_i64(fid);
	for (int64 n = 0; n < size; n++) {
		string key = read_str(fid);
		Value value = read_value(fid);
		dict[key] = value;
	}
}

void Util::save_value(FILE* fid, Value value) {
	save_int(fid, (int)value.type());
	switch (value.type()) {
	case vt::none:
		break;
	case vt::kbool:
		save_bool(fid, (bool)value);
		break;
	case vt::kint:
		save_int(fid, (int)value);
		break;
	case vt::int64:
		save_int64(fid, (int64)value);
		break;
	case vt::kfloat:
		save_float(fid, (float)value);
		break;
	case vt::string:
		save_str(fid, (string)value);
		break;
	case vt::list:
	{
		List temp = value;
		save_list(fid, temp);
		break;
	}
	case vt::dict:
		save_dict(fid, (Dict)value);
		break;
	case vt::farray:
	{
		Array<float> temp = value;
		save_farray(fid, temp);
		break;
	}
	case vt::narray:
	{
		Array<int> temp = value;
		save_narray(fid, temp);
		break;
	}
	case vt::n64array:
	{
		Array<int64> temp = value;
		save_n64array(fid, temp);
		break;
	}
	case vt::barray:
	{
		Array<bool> temp = value;
		save_barray(fid, temp);
		break;
	}
	case vt::shape:
	{
		Shape temp = value;
		save_shape(fid, temp);
		break;
	}
	}
}

Value Util::read_value(FILE* fid) {
	Value value;
	enum class vt type = (enum class vt) read_int(fid);

	switch (type) {
	case vt::none:
		break;
	case vt::kbool:
		value = read_bool(fid);
		break;
	case vt::kint:
		value = read_int(fid);
		break;
	case vt::int64:
		value = read_i64(fid);
		break;
	case vt::kfloat:
		value = read_float(fid);
		break;
	case vt::string:
		value = read_str(fid);
		break;
	case vt::list:
		value = read_list(fid);
		break;
	case vt::dict:
		value = read_dict(fid);
		break;
	case vt::farray:
		value = read_farray(fid);
		break;
	case vt::narray:
		value = read_narray(fid);
		break;
	case vt::n64array:
		value = read_n64array(fid);
		break;
	case vt::barray:
		value = read_barray(fid);
		break;
	case vt::shape:
		value = read_shape(fid);
		break;
	}

	return value;
}

string Util::str_format(const char* fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	string result = Util::str_format(fmt, ap);
	va_end(ap);
	return result;
}

string Util::str_format(const char* fmt, va_list ap) {
	va_list ap2;
	va_copy(ap2, ap);
	int count = vsnprintf(NULL, 0, fmt, ap) + 1;
	assert(count > 0);

	std::string result = std::string(count, '\0');
	if (vsnprintf(&result[0], count, fmt, ap2) < 0) { return "error"; }

	return result;
}

string Util::join(string glue, vector<string> in_words) {
	int size =  (int) in_words.size();

	if (size == 0) return "";
	
	string joined = in_words[0];

	for (int n = 1; n < size; n++) {
		joined += glue + in_words[n];
	}

	return joined;
}

string Util::externd_conf(string format, Dict info) {
	string result;

#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
#else
	int64 pos = 0;

	while (true) {
		int64 pos1 = format.find('%', pos);
		if (pos1 == (int64)string::npos) {
			result += format.substr(pos);
			break;
		}
		int64 pos2 = format.find('%', pos1+1);
		assert (pos2 != (int64)string::npos);

		result += format.substr(pos, pos1 - pos);
		string name = format.substr(pos1 + 1, pos2 - pos1 - 1);
		assert(info.find(name) != info.end());
		result += info[name].description();
		
		pos = pos2 + 1;
	}
#endif
	return result;
}

string Util::get_timestamp() {
	time_t now_t = time(NULL);
	struct tm now;
	if (localtime_s(&now, &now_t) != 0) throw KaiException(KERR_ASSERT);
	char timestamp[1024];
	snprintf(timestamp, 1024, "%04d%02d%02d-%02d%02d%02d", now.tm_year + 1900, now.tm_mon + 1, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);
	return timestamp;
}

#ifdef KAI2021_WINDOWS
FILE* Util::fopen(const char* filepath, const char* mode) {
	FILE* fid = NULL;

	string win_path = filepath;
	std::replace(win_path.begin(), win_path.end(), '/', '\\');

	if (fopen_s(&fid, win_path.c_str(), mode) != 0) {
		throw exception("file open failure");
	}
	return fid;
}
#else
FILE* Util::fopen(const char* filepath, const char* mode) {
	return ::fopen(filepath, mode);
}
#endif

#ifdef KAI2021_WINDOWS
#include <Windows.h>

/* Windows sleep in 100ns units */
bool Util::nanosleep(int64 ns) {
	/* Declarations */
	HANDLE timer;	/* Timer hObject */
	LARGE_INTEGER li;	/* Time defintion */
	/* Create timer */
	if (!(timer = CreateWaitableTimer(NULL, TRUE, NULL)))
		return FALSE;
	/* Set timer properties */
	li.QuadPart = -ns / 100;
	if (!SetWaitableTimer(timer, &li, 0, NULL, NULL, FALSE)) {
		CloseHandle(timer);
		return FALSE;
	}
	/* Start & wait for timer */
	WaitForSingleObject(timer, INFINITE);
	/* Clean resources */
	CloseHandle(timer);
	/* Slept without problems */
	return TRUE;
}
#else
bool Util::nanosleep(int64 ns) {
	struct timespec req = { 0 };
	req.tv_sec = ns / 1000000000LL;
	req.tv_nsec = ns % 1000000000LL;
	return ::nanosleep(&req, (struct timespec*)NULL);
}
#endif

void Util::read_wav_file(string filepath, WaveInfo* pInfo)
{
	// Read the wave file
	FILE* fhandle = fopen(filepath.c_str(), "rb");

	fread(pInfo->ChunkID, 1, 4, fhandle);
	fread(&pInfo->ChunkSize, 4, 1, fhandle);
	fread(pInfo->Format, 1, 4, fhandle);
	fread(pInfo->Subchunk1ID, 1, 4, fhandle);
	fread(&pInfo->Subchunk1Size, 4, 1, fhandle);
	fread(&pInfo->AudioFormat, 2, 1, fhandle);
	fread(&pInfo->NumChannels, 2, 1, fhandle);
	fread(&pInfo->SampleRate, 4, 1, fhandle);
	fread(&pInfo->ByteRate, 4, 1, fhandle);
	fread(&pInfo->BlockAlign, 2, 1, fhandle);
	fread(&pInfo->BitsPerSample, 2, 1, fhandle);
	fread(&pInfo->Subchunk2ID, 1, 4, fhandle);
	fread(&pInfo->Subchunk2Size, 4, 1, fhandle);

	if (pInfo->Subchunk2Size != pInfo->ChunkSize - 36) {
		//if (pInfo->Subchunk2Size != 3 && pInfo->Subchunk2Size != 4) throw KaiException(KERR_ASSERT);
		pInfo->Subchunk2Size = pInfo->ChunkSize - 36;
	}

	pInfo->pData = new unsigned char[pInfo->Subchunk2Size]; // Create an element for every sample
	int64 nRead = fread(pInfo->pData, 1, pInfo->Subchunk2Size, fhandle); // Reading raw audio data
	if (nRead != pInfo->Subchunk2Size) throw KaiException(KERR_ASSERT);
	//if (!feof(fhandle)) throw KaiException(KERR_ASSERT);

	fclose(fhandle);
}

void Util::write_wav_file(string filepath, WaveInfo* pInfo)
{
	// Write the same file
	FILE* fhandle = fopen(filepath.c_str(), "wb");
	fwrite(pInfo->ChunkID, 1, 4, fhandle);
	fwrite(&pInfo->ChunkSize, 4, 1, fhandle);
	fwrite(pInfo->Format, 1, 4, fhandle);
	fwrite(pInfo->Subchunk1ID, 1, 4, fhandle);
	fwrite(&pInfo->Subchunk1Size, 4, 1, fhandle);
	fwrite(&pInfo->AudioFormat, 2, 1, fhandle);
	fwrite(&pInfo->NumChannels, 2, 1, fhandle);
	fwrite(&pInfo->SampleRate, 4, 1, fhandle);
	fwrite(&pInfo->ByteRate, 4, 1, fhandle);
	fwrite(&pInfo->BlockAlign, 2, 1, fhandle);
	fwrite(&pInfo->BitsPerSample, 2, 1, fhandle);
	fwrite(&pInfo->Subchunk2ID, 1, 4, fhandle);
	fwrite(&pInfo->Subchunk2Size, 4, 1, fhandle);
	fwrite(pInfo->pData, 1, pInfo->Subchunk2Size, fhandle);

	fclose(fhandle);
}
