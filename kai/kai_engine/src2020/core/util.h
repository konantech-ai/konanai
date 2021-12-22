/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "value.h"

#ifdef KAI2021_WINDOWS
#else
#include <dirent.h>
#endif

struct WaveInfo {
public:
	char ChunkID[4], Format[4], Subchunk1ID[4], Subchunk2ID[4];
	int ChunkSize, Subchunk1Size, SampleRate, ByteRate, Subchunk2Size;
	short AudioFormat, NumChannels, BlockAlign, BitsPerSample;
	
	unsigned char* pData;

public:
	WaveInfo() { pData = NULL; }
	virtual ~WaveInfo() { delete[] pData; }
};

class Util {
public:
	static vector<vector<string>> load_csv(string filepath, vector<string>* pHead = NULL);
	
	static void mkdir(string path);
	static vector<string> list_dir(string path);
	static bool remove_all(string path);

	static Array<float> load_image_pixels(string filepath, Shape resolution, Shape input_shape);

	static void load_kodell_dump_file(string filepath, Array<float>& xs, Array<float>& ys, vector<string>* pTargetNames, int cat_cnt);

	static void save_kodell_dump_file(string filepath, Array<float>& xs, Array<float>& ys, vector<string>* pTargetNames, int cat_cnt);

	static string str_format(const char* fmt, ...);
	static string str_format(const char* fmt, va_list ap);

	static string join(string glue, vector<string> in_words);

	static void print(string title, Shape shape);

	static void read_map_si(FILE* fid, map<string, int>& dic);
	static void read_map_is(FILE* fid, map<int, string>& dic);
	static void read_map_ii(FILE* fid, map<int, int>& dic);

	static void read_vv_int(FILE* fid, vector<vector<int>>& sents);
	static void read_v_int(FILE* fid, vector<int>& words);

	static void read_map_si64(FILE* fid, map<string, int64>& dic);
	static void read_map_i64s(FILE* fid, map<int64, string>& dic);
	static void read_map_i64i64(FILE* fid, map<int64, int64>& dic);

	static void read_vv_i64(FILE* fid, vector<vector<int64>>& sents);
	static void read_v_i64(FILE* fid, vector<int64>& words);

	static void save_bool(FILE* fid, bool dat);
	static void save_int(FILE* fid, int dat);
	static void save_int64(FILE* fid, int64 dat);
	static void save_str(FILE* fid, string dat);
	static void save_float(FILE* fid, float dat);
	static void save_list(FILE* fid, List dat);
	static void save_dict(FILE* fid, Dict dict);
	static void save_shape(FILE* fid, Shape shape);
	static void save_farray(FILE* fid, Array<float> param);
	static void save_barray(FILE* fid, Array<bool> param);
	static void save_narray(FILE* fid, Array<int> param);
	static void save_n64array(FILE* fid, Array<int64> param);

	static bool read_bool(FILE* fid);
	static int read_int(FILE* fid);
	static int64 read_i64(FILE* fid);
	static string read_str(FILE* fid);
	static float read_float(FILE* fid);
	static List read_list(FILE* fid);
	static Dict read_dict(FILE* fid);
	static Shape read_shape(FILE* fid);
	static Array<float> read_farray(FILE* fid);
	static Array<bool> read_barray(FILE* fid);
	static Array<int> read_narray(FILE* fid);
	static Array<int64> read_n64array(FILE* fid);

	static void save_map_si(FILE* fid, map<string, int>& dic);
	static void save_map_is(FILE* fid, map<int, string>& dic);
	static void save_map_ii(FILE* fid, map<int, int>& dic);

	static void save_vv_int(FILE* fid, vector<vector<int>>& sents);
	static void save_v_int(FILE* fid, vector<int>& words);

	static void save_map_si64(FILE* fid, map<string, int64>& dic);
	static void save_map_i64s(FILE* fid, map<int64, string>& dic);
	static void save_map_i64i64(FILE* fid, map<int64, int64>& dic);

	static void save_vv_i64(FILE* fid, vector<vector<int64>>& sents);
	static void save_v_i64(FILE* fid, vector<int64>& words);

	static void read_shape(FILE* fid, Shape& shape);
	static void read_farray(FILE* fid, Array<float>& param);
	static void read_narray(FILE* fid, Array<int>& param);

	static void read_dict(FILE* fid, Dict& dict);

	static void save_value(FILE* fid, Value value);
	static Value read_value(FILE* fid);

	static string externd_conf(string format, Dict info);

	static string get_timestamp();

	static FILE* fopen(const char* filepath, const char* mode);

	static bool nanosleep(int64 ns);
//hs.cho
#ifdef KAI2021_WINDOWS
	static void load_jpeg_image_pixels(float* pBuf, string filepath, Shape data_shape);
	static void draw_images_horz(bool show, string folder, Array<float> arr, Shape image_shape, int ratio=1, int gap=5);
#endif
	static void read_wav_file(string filepath, WaveInfo* pInfo);
	static void write_wav_file(string filepath, WaveInfo* pInfo);

protected:
	static Array<float> ms_load_array_from_kodell_dump_file(FILE* fid);
	static void ms_load_names_from_kodell_dump_file(vector<string>* pTargetNames, FILE* fid);

	static void ms_save_array_to_kodell_dump_file(FILE* fid, Array<float> arr);
	static void ms_save_names_to_kodell_dump_file(FILE* fid, vector<string>* pTargetNames);
};