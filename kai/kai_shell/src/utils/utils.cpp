/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "utils.h"

#include <algorithm>

#include "../data_feeders/urban_feeder.h"
#ifdef KAI2021_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#define _mkdir(filepath)  mkdir(filepath, 0777)
#endif
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

KString Utils::join(KStrList list, KString sDelimeter) {
	if (list.size() == 0) return "";

	KString sJoined = list[0];

	for (size_t n = 1; n < list.size(); n++) {
		sJoined += sDelimeter;
		//hs.cho
		//sJoined += list[n];
		std::string temp = list[n];
		sJoined += temp;
	}

	return sJoined;
}

void Utils::mkdir(KString path) {
#ifdef KAI2021_WINDOWS
	std::replace(path.begin(), path.end(), '/', '\\');
#endif
	int rs = ::_mkdir(path.c_str());
	printf("\"%s\" has been created.\n", path.c_str());
}

bool Utils::file_exist(KString filepath) {
	FILE* fid = fopen(filepath, "rb", false);
	if (fid == NULL) return false;
	fclose(fid);
	return true;
}

#ifdef KAI2021_WINDOWS
FILE* Utils::fopen(KString filepath, KString mode, bool bThrow) {
	FILE* fid = NULL;

	std::replace(filepath.begin(), filepath.end(), '/', '\\');

	if (fopen_s(&fid, filepath.c_str(), mode.c_str()) != 0) {
		if (bThrow) {
			THROW(KERR_FILE_OPEN_FAILURE);
		}
		else return NULL;
	}
	return fid;
}
#else
FILE* Utils::fopen(KString filepath, KString mode,bool bThrow) {
	return ::fopen(filepath.c_str(), mode.c_str());
}
#endif

void Utils::read_wav_file(KString filepath, WaveInfo* pInfo)
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
	KInt nRead = fread(pInfo->pData, 1, pInfo->Subchunk2Size, fhandle); // Reading raw audio data
	if (nRead != pInfo->Subchunk2Size) throw KERR_ASSERT;
	//if (!feof(fhandle)) throw KaiException(KERR_ASSERT);

	fclose(fhandle);
}

void Utils::load_jpeg_image_pixels(KFloat* pBuf, KString filepath, KaiShape data_shape, bool crop) {
	std::replace(filepath.begin(), filepath.end(), '/', '\\');

	cv::Mat img = cv::imread(filepath, 1);

	int height = (int)data_shape[0];
	int width = (int)data_shape[1];

	int h_base = 0, w_base = 0;

	if (crop) {
		assert(width == height);

		int img_height = img.rows;
		int img_width = img.cols;

		if (img_height * width > img_width * height) {
			float ratio = (float)img_width / img_height;
			w_base = (width - height * ratio) / 2;
			width = height * ratio;
		}
		else if (img_height * width < img_width * height) {
			float ratio = (float)img_height / img_width;
			h_base = (height - width * ratio) / 2;
			height = width * ratio;
		}
	}

	cv::resize(img, img, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);

	int chn = (int)img.channels();

	for (int nh = 0; nh < img.rows; nh++) {
		float* pRow = pBuf + (nh + h_base) * width * 3;
		for (int nw = 0; nw < img.cols; nw++) {
			float* pCol = pRow + (nw + w_base) * 3;
			cv::Vec3b intensity = img.at<cv::Vec3b>(nh, nw);
			for (int k = 0; k < chn; k++) {
				pCol[k] = (KFloat)intensity.val[k];
			}
		}
	}
}

