/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../common.h"

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

class Utils {
public:
	static KString join(KStrList list, KString sDelimeter);
	static void mkdir(KString path);
	static bool file_exist(KString path);
	static FILE* fopen(KString filepath, KString mode, bool bthrow=true);
	static void read_wav_file(KString filepath, WaveInfo* pInfo);
	static void load_jpeg_image_pixels(KFloat* pBuf, KString filepath, KaiShape data_shape, bool crop);
};