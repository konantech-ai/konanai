/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "urbansound.h"

#include "../core/random.h"
#include "../cuda/cuda_util.cuh"
/*
#include "../core/array.h"
#include "../core/util.h"
#include "../core/log.h"
*/

UrbanSoundDataset::UrbanSoundDataset(string data_name, string cache_name, int64 step_cnt, int64 step_win, int64 freq_win, int64 freq_cnt) : Dataset("urbansound", "classify") {
	logger.Print("dataset loading...");
	
	string data_path = KArgs::data_root + data_name;
	string cache_path = KArgs::cache_root + "urbansound";

	Util::mkdir(KArgs::cache_root);
	Util::mkdir(cache_path);

	cache_path += "/" + cache_name;

#ifdef KAI2021_WINDOWS
	try {
		load_cache(cache_path);
	}
	catch (exception err) {
		create_cache(data_path, step_cnt, step_win, freq_win, freq_cnt);
		save_cache(cache_path);
	}
#else
	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 1);
#endif

	Shape xs_shape = m_default_xs.shape();

	m_data_cnt = xs_shape[0];
	m_timesteps = xs_shape[1];
	m_timefeats = xs_shape[2];
	
	m_target_cnt = m_target_names.size();

	//m_default_xlen = kmath->zeros_int(Shape(m_data_cnt));
	//m_default_xlen += (int64)m_timesteps;

	input_shape = Shape(m_timefeats);
	output_shape = Shape(m_target_cnt);

	m_shuffle_index(m_data_cnt);

	logger.Print("dataset prepared...");
}

UrbanSoundDataset::~UrbanSoundDataset() {
}

#ifdef KAI2021_WINDOWS
void UrbanSoundDataset::load_cache(string cache_path) {
	logger.Print("load_cache start time: %lld", time(NULL));
	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 1);
	logger.Print("load_cache end time: %lld", time(NULL));
}

void UrbanSoundDataset::save_cache(string cache_path) {
	logger.Print("save_cache start time: %lld", time(NULL));
	Util::save_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 1);
	logger.Print("save_cache end time: %lld", time(NULL));
}

/*
* step_cnt: 추출되는 시계열 데이터의 길이, 시간대  갯수로 표현
* step_win: 시계열 데이터의 시간대 단위 길이, ms 단위의 시간으로 표현
* freq_cnt: 추출된 데이터의 시간대별 데이터 크기, 분석된 주파수 스펙트럼의 주파수 갯수로 표현
* freq_win: 주파수 분석에 사용될 wav 데이터의 길이, ms 단위의 시간으로 표현
*/
void UrbanSoundDataset::create_cache(string data_path, int64 step_cnt, int64 step_win, int64 freq_win, int64 freq_cnt) {
	logger.Print("create_cache start time: %lld", time(NULL));

	vector<string> head;
	vector<vector<string>> rows = Util::load_csv(data_path + "/train.csv", &head);

	int64 file_count = rows.size();
	
	int64 sample_rate = 44100;

	int64 fft_width = freq_win;
	int64 step_width = sample_rate * step_win / 1000;
	int64 fetch_width = step_width * (step_cnt - 1) + fft_width;

	Array<float> wave_buffer(Shape(file_count, fetch_width));
	Array<int64> cat_idxs = Array<int64>(Shape(file_count));
	
	float* wbufp = wave_buffer.data_ptr();

	Dict bad_info;

	for (int64 n = 0; n < file_count; n++) {
		vector<string> row = rows[n];

		auto at_pos = find(m_target_names.begin(), m_target_names.end(), row[1]);
		if (at_pos != m_target_names.end()) {
			cat_idxs[Idx(n)] = (int64) (at_pos - m_target_names.begin());
		}
		else {
			cat_idxs[Idx(n)] = (int64) m_target_names.size();
			m_target_names.push_back(row[1]);
		}
		
		string wav_path = data_path + "/Train/" + row[0] + ".wav";

		try {
			m_load_wave_data(wav_path, wbufp, fetch_width, sample_rate);
		}
		catch (...) {
			throw KaiException(KERR_ASSERT);
		}

		wbufp += fetch_width;
	}

	logger.Print("fft analyze start time: %lld", time(NULL));

	m_default_xs = CudaUtil::WaveFFT(wave_buffer, step_width, step_cnt, fft_width, freq_cnt);
	m_default_ys = kmath->onehot(cat_idxs, m_target_names.size());

	logger.Print("fft analyze end time: %lld", time(NULL));
}

void UrbanSoundDataset::m_load_wave_data(string wav_path, float* pWaveBuffer, int64 fetch_width, int64 sample_rate) {
	WaveInfo wav_info;
	Util::read_wav_file(wav_path, &wav_info);
	if (strncmp(wav_info.Format, "WAVE", 4) != 0) throw exception("bad format");
	if (strncmp(wav_info.ChunkID, "RIFF", 4) != 0) throw exception("bad chunk ID");

	int64 start_offset = 0;
	int64 chunk_size = wav_info.Subchunk2Size;

	int64 sample_width = chunk_size * 8 / (wav_info.BitsPerSample * wav_info.NumChannels);

	int64 need_width = fetch_width / 44100 * wav_info.SampleRate;

	if (sample_width > need_width) {
		start_offset = Random::dice(sample_width - need_width + 1);
	}

	_WavedataPicker picker(&wav_info, start_offset, sample_rate);

	for (int64 n = 0; n < fetch_width; n++) {
		*pWaveBuffer++ = picker.fetch(n);
	}
}

_WavedataPicker::_WavedataPicker(WaveInfo* pWaveInfo, int64 start_offset, int64 nFetchRate) {
	m_pWaveInfo = pWaveInfo;
	m_start_offset = start_offset;
	m_fetch_rate = nFetchRate;
}

_WavedataPicker::~_WavedataPicker() {
}

float _WavedataPicker::fetch(int64 nth) {
	float result = 0;

	int64 nth_prod = m_pWaveInfo->SampleRate * nth;
	
	int64 nth_idx = nth_prod / m_fetch_rate;
	int64 nth_mod = nth_prod % m_fetch_rate;

	if (nth_mod == 0) return m_get_sample(nth_idx);

	float data1 = m_get_sample(nth_idx);
	float data2 = m_get_sample(nth_idx+1);

	return data1 + (data2 - data1) * nth_mod / m_fetch_rate;
}

float _WavedataPicker::m_get_sample(int64 nth) {
	int64 offset = (nth + m_start_offset) * m_pWaveInfo->BitsPerSample * m_pWaveInfo->NumChannels / 8;

	unsigned char* pData = m_pWaveInfo->pData + offset;

	if (offset + m_pWaveInfo->BitsPerSample / 8 > m_pWaveInfo->Subchunk2Size) return 0;

	try {
		if (m_pWaveInfo->BitsPerSample == 16) return (float)*(short*)pData / 65536.0f;
		else if (m_pWaveInfo->BitsPerSample == 8) return ((float)*pData - 128.0f) / 128.0f;
		else if (m_pWaveInfo->BitsPerSample == 24) {
			int value = pData[0] | (pData[1] << 8) | (pData[2] << 16);
			value = (value & 0x7fffff) - (value & 0x800000);
			return (float)value / 8388608.0f;
		}
		else if (m_pWaveInfo->BitsPerSample == 32) return (float)*(int*)pData / (8388608.0f * 256);
		else if (m_pWaveInfo->BitsPerSample == 4) {
			if (m_pWaveInfo->NumChannels % 2 == 0 || (nth + m_start_offset) % 2 == 0)
				return ((float)(*pData >> 4) - 8.0f) / 8.0f;
			else return ((float)(*pData & 0x0f) - 8.0f) / 8.0f;
		}
		else throw KaiException(KERR_ASSERT);
	}
	catch (...) {
		int tmp = 0;
		int a = 0;
	}

	return 0;
}

#else
#endif

/*
void UrbanSoundDataset::gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
	memset(py, 0, ysize * sizeof(float));
	int64 npos = (int64) m_default_ys[Idx(data_idx)];
	py[npos] = 1.0;
}
*/

/*
void UrbanSoundDataset::generate_data(int64* data_idxs, int64 size, Value& xs, Value& ys) {
	Array<float> xarr(input_shape.add_front(m_timesteps).add_front(size));
	Array<float> yarr = kmath->zeros(output_shape.add_front(size));

	for (int64 n = 0; n < size; n++) {
		int64 npos = (int64)m_ys[Idx(data_idxs[n])];;
		xarr[Axis(n, _all_, _all_)] = m_xs[Axis(data_idxs[n], _all_, _all_)];
		yarr[Axis(n, npos)] = 1.0;
	}

	xs = xarr;
	ys = yarr;
}
*/

void UrbanSoundDataset::visualize(Value cxs, Value cest, Value cans) {
	m_show_select_results(cest, cans, m_target_names); 
}
