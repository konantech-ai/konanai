/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "../math/karray.h"

class KUtil {
public:
	void mkdir(KString dir);
	void remove_all(KString sFilePath);

	KaiList list_dir(KString path);

	FILE* fopen(KString filepath, KString mode);

	KString get_timestamp(time_t tm);

	void load_jpeg_image_pixels(KFloat* pBuf, KString filepath, KaiShape data_shape);
	vector<vector<string>> load_csv(KString filepath, KaiList* pHead = NULL);

	KaiValue seek_dict(KaiDict dict, KString sKey, KaiValue kDefaultValue);
	KaiValue seek_set_dict(KaiDict& dict, KString sKey, KaiValue kDefaultValue);

	void save_value(FILE* fid, KaiValue value);
	KaiValue read_value(FILE* fid);

	void save_int(FILE* fid, KInt dat);
	void save_str(FILE* fid, KString dat);
	void save_float(FILE* fid, KFloat dat);

	void save_list(FILE* fid, KaiList value);
	void save_dict(FILE* fid, KaiDict value);

	void save_shape(FILE* fid, KaiShape shape);

	KInt read_int(FILE* fid);
	KString read_str(FILE* fid);
	KFloat read_float(FILE* fid);

	KaiList read_list(FILE* fid);
	KaiDict read_dict(FILE* fid);

	void read_shape(FILE* fid, KaiShape& shape);

	KaiArray<KInt> read_narray(FILE* fid);
	KaiArray<KFloat> read_farray(FILE* fid);

	template <class T> void save_array(FILE* fid, KaiArray<T> arr);

};

extern KUtil kutil;