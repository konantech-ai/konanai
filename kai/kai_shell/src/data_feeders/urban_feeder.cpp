/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#define THROW(x) {  fprintf(stdout, "Kai_cuda error-%d\n", x); assert(0); } 

#include "../../../kai_engine/src/include/kai_errors.h"

#include "urban_feeder.h"
#ifdef KAI2021_WINDOWS
#include <direct.h>
#else
#include <cstring>
#endif
int UrbanDataFeeder::ms_checkCode = 60753427;

UrbanDataFeeder::UrbanDataFeeder(KHSession hSession) : DataFeeder() {
	m_hSession = hSession;
	m_version = 1;
	m_pFFT = 0;

	m_file_count = 0;
	m_step_cnt = 0;
	m_freq_cnt = 0;
}

UrbanDataFeeder::~UrbanDataFeeder() {
	delete[] m_pFFT;
}

void UrbanDataFeeder::loadData(KString data_path, KString cache_path) {
	Utils::mkdir(cache_path);

	if (!m_load_cache(cache_path)) {
		m_load_data(data_path);
		m_save_cache(cache_path);
	}
}

void UrbanDataFeeder::m_load_data(KString data_path) {
	printf("loading urban sound information file...\n");

	FILE* fList = Utils::fopen(data_path + "/train.csv", "rt");

	char buffer[64];

	fgets(buffer, 64, fList);  // first line is header

	KaiDict product_ids;
	KaiList file_ids;

	while (!feof(fList)) {
		char* pRead = fgets(buffer, 64, fList);
		if (pRead == NULL) break;
		KString sLine = pRead;
		size_t pos1 = sLine.find(',');
		size_t pos2 = sLine.find('\n');
		KString id = sLine.substr(0, pos1);
		KString product = sLine.substr(pos1 + 1, pos2 - pos1 - 1);
		product.erase(std::remove(product.begin(), product.end(), '\r'),product.end());
		KInt index;
		if (product_ids.find(product) == product_ids.end()) {
			index = product_ids.size();
			product_ids[product] = index;
			m_category_names.push_back(product);
		}
		else {
			index = product_ids[product];
		}

		file_ids.push_back(id);
		m_dat_categorys.push_back(index);
	}

	fclose(fList);

	KInt file_count = file_ids.size();

	printf("reading urban sound wave data for %lld files...\n", file_count);

	KInt sample_rate = 44100;

	KInt step_cnt = 200;
	KInt step_win = 10;
	KInt freq_win = 16384;
	KInt freq_cnt = 128;

	KInt fft_width = freq_win;
	KInt step_width = sample_rate * step_win / 1000;
	KInt fetch_width = step_width * (step_cnt - 1) + fft_width;

	KFloat* pWave = new KFloat[file_count * fetch_width];
	KFloat* p_buffer = pWave;

	for (KInt n = 0; n < file_count; n++) {
		KString wav_path = data_path + "/Train/" + (KString)file_ids[n] + ".wav";

		try {
			m_load_wave_data(wav_path, p_buffer, fetch_width, sample_rate);
		}
		catch (...) {
			assert(0);
		}

		p_buffer += fetch_width;
	}

	if (m_pFFT) {
		delete[] m_pFFT;
		m_pFFT = 0;
	}

	m_file_count = file_count;
	m_step_cnt = step_cnt;
	m_freq_cnt = freq_cnt;

	m_data_size = m_file_count * m_step_cnt * m_freq_cnt;

	printf("Fast Furier Transfer analyzing %lld files...\n", file_count);

	m_pFFT = new KFloat[m_data_size];

	KERR_CHK(KAI_Util_fft(m_hSession, pWave, m_pFFT, file_count, fetch_width, step_width, step_cnt, fft_width, freq_cnt));

	delete[] pWave;
}

KaiList UrbanDataFeeder::getTargetNames() {
	return m_category_names;
}

void UrbanDataFeeder::m_load_wave_data(KString wav_path, KFloat* pWaveBuffer, KInt fetch_width, KInt sample_rate) {
	WaveInfo wav_info;
	Utils::read_wav_file(wav_path, &wav_info);
	if (strncmp(wav_info.Format, "WAVE", 4) != 0) throw "bad format";
	if (strncmp(wav_info.ChunkID, "RIFF", 4) != 0) throw "bad chunk ID";

	KInt start_offset = 0;
	KInt chunk_size = wav_info.Subchunk2Size;

	KInt sample_width = chunk_size * 8 / (wav_info.BitsPerSample * wav_info.NumChannels);

	KInt need_width = fetch_width / 44100 * wav_info.SampleRate;

	if (sample_width > need_width) {
		//start_offset = Random::dice(sample_width - need_width + 1);	// random
		start_offset = (sample_width - need_width) / 2;					// at center
	}

	_WavedataPicker picker(&wav_info, start_offset, sample_rate);

	for (KInt n = 0; n < fetch_width; n++) {
		*pWaveBuffer++ = picker.fetch(n);
	}
}

bool UrbanDataFeeder::m_load_cache(KString cache_path) {
	FILE* fid = Utils::fopen(cache_path + "/urban_cache.dat", "rb", false);
	
	if (fid == NULL) return false;

	printf("loading urban sound cache data...\n");

	int checkCode;
	int version;
	int nCatCount;
	int nDatCount;

	if (fread(&checkCode, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }
	if (fread(&version, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }
	if (fread(&nCatCount, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }

	if (checkCode != ms_checkCode) { fclose(fid); return false; }
	if (version != m_version) { fclose(fid); return false; }

	m_category_names.clear();

	char buffer[128];
	int name_len;

	for (int n = 0; n < nCatCount; n++) {
		if (fread(&name_len, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }
		if (fread(buffer, sizeof(char), name_len, fid) != name_len) { fclose(fid); return false; }
		buffer[name_len] = 0;
		m_category_names.push_back(buffer);
	}

	if (fread(&checkCode, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }
	if (fread(&nDatCount, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }

	if (checkCode != ms_checkCode) { fclose(fid); return false; }

	int cat;

	for (int n = 0; n < nDatCount; n++) {
		if (fread(&cat, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }
		m_dat_categorys.push_back(cat);
	}

	if (fread(&checkCode, sizeof(int), 1, fid) != 1) { fclose(fid); return false; }
	if (fread(&m_file_count, sizeof(KInt), 1, fid) != 1) { fclose(fid); return false; }
	if (fread(&m_step_cnt, sizeof(KInt), 1, fid) != 1) { fclose(fid); return false; }
	if (fread(&m_freq_cnt, sizeof(KInt), 1, fid) != 1) { fclose(fid); return false; }
	if (fread(&m_data_size, sizeof(KInt), 1, fid) != 1) { fclose(fid); return false; }

	if (checkCode != ms_checkCode) { fclose(fid); return false; }
	if (m_data_size != m_file_count * m_step_cnt * m_freq_cnt) { fclose(fid); return false; }

	delete[] m_pFFT;
	m_pFFT = new KFloat[m_data_size];
	
	if (fread(m_pFFT, sizeof(KFloat), m_data_size, fid) != m_data_size) { fclose(fid); return false; }

	fclose(fid);

	return true;
}

void UrbanDataFeeder::m_save_cache(KString cache_path) {
	printf("saving urban sound cache data...\n");

	FILE* fid = Utils::fopen(cache_path + "/urban_cache.dat", "wb");

	if (fid == NULL) THROW(KERR_FAILURE_ON_FILE_SAVE);

	int nCatCount = (int)m_category_names.size();
	int nDatCount = (int)m_dat_categorys.size();

	if (fwrite(&ms_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_version, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&nCatCount, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);

	for (int n = 0; n < nCatCount; n++) {
		KString sCategory = m_category_names[n];
		int nCatNameLen = (int)strlen(sCategory.c_str());
		if (fwrite(&nCatNameLen, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
		if (fwrite(sCategory.c_str(), sizeof(char), nCatNameLen, fid) != nCatNameLen) THROW(KERR_FAILURE_ON_FILE_SAVE);
	}

	if (fwrite(&ms_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&nDatCount, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);

	for (int n = 0; n < nDatCount; n++) {
		int cat = (int)(KInt)m_dat_categorys[n];
		if (fwrite(&cat, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	}

	if (fwrite(&ms_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_file_count, sizeof(KInt), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_step_cnt, sizeof(KInt), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_freq_cnt, sizeof(KInt), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_data_size, sizeof(KInt), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(m_pFFT, sizeof(KFloat), m_data_size, fid) != m_data_size) THROW(KERR_FAILURE_ON_FILE_SAVE);

	fclose(fid);
}

KBool UrbanDataFeeder::m_getDataCount(void* pAux, KInt* pnDataCount) {
	*pnDataCount = m_file_count;
	return true;
}

KBool UrbanDataFeeder::m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod) {
	*pSuffleMethod = Ken_data_suffle_method::random;
	return true;
}

KBool UrbanDataFeeder::m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) {
	pFieldInfo->m_bIsFloat = true;
	pFieldInfo->m_shape = bInput ? KaiShape{ m_freq_cnt } : KaiShape{ m_category_names.size() };
	pFieldInfo->m_bIsSeq = bInput ? true : false;
	pFieldInfo->m_nTimesteps = bInput ? m_step_cnt : 0;

	return true;
}

KBool UrbanDataFeeder::m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	THROW(KERR_INTERNAL_LOGIC_ERROR);
	return false;
}

KBool UrbanDataFeeder::m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
	KInt dat_size = shape.total_size();
	if (bInput) {
		KFloat* pData = m_pFFT;
		for (auto& idx : nDatIndexs) {
			memcpy(pfBuffer, pData + idx * dat_size, dat_size * sizeof(KFloat));
			pfBuffer += dat_size;
		}
	}
	else {
		for (auto& idx : nDatIndexs) {
			memset(pfBuffer, 0, dat_size * sizeof(KFloat));
			pfBuffer[(KInt)m_dat_categorys[idx]] = 1.0f;
			pfBuffer += dat_size;
		}
	}

	return true;
}

_WavedataPicker::_WavedataPicker(WaveInfo* pWaveInfo, KInt start_offset, KInt nFetchRate) {
	m_pWaveInfo = pWaveInfo;
	m_start_offset = start_offset;
	m_fetch_rate = nFetchRate;
}

_WavedataPicker::~_WavedataPicker() {
}

float _WavedataPicker::fetch(KInt nth) {
	float result = 0;

	KInt nth_prod = m_pWaveInfo->SampleRate * nth;

	KInt nth_idx = nth_prod / m_fetch_rate;
	KInt nth_mod = nth_prod % m_fetch_rate;

	if (nth_mod == 0) return m_get_sample(nth_idx);

	float data1 = m_get_sample(nth_idx);
	float data2 = m_get_sample(nth_idx + 1);

	return data1 + (data2 - data1) * nth_mod / m_fetch_rate;
}

float _WavedataPicker::m_get_sample(KInt nth) {
	KInt offset = (nth + m_start_offset) * m_pWaveInfo->BitsPerSample * m_pWaveInfo->NumChannels / 8;

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
		else throw KERR_ASSERT;
	}
	catch (...) {
		int tmp = 0;
		int a = 0;
	}

	return 0;
}
