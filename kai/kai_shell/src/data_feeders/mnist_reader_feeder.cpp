/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "mnist_reader_feeder.h"

int MnistReaderFeeder::ms_checkCode = 29883720;

MnistReaderFeeder::MnistReaderFeeder() : DataFeeder() {
	m_version = 1;

	m_data_count = 0;
	m_pImages = 0;
	m_pLabels = 0;
}

MnistReaderFeeder::~MnistReaderFeeder() {
	delete[] m_pImages;
	delete[] m_pLabels;
}

void MnistReaderFeeder::loadData(KString data_path, KString cache_path) {
	Utils::mkdir(cache_path);

	if (!m_load_cache(cache_path)) {
		m_load_data(data_path);
		m_save_cache(cache_path);
	}

	m_fill_digit_words();
}

void MnistReaderFeeder::m_load_data(KString data_path) {
	printf("loading mnist reader information file...\n");

	KString image_path = data_path + "/train-images-idx3-ubyte";
	KString label_path = data_path + "/train-labels-idx1-ubyte";

	m_load_image_file(image_path);
	m_load_label_file(label_path);
}

int MnistReaderFeeder::m_fread_int_msb(FILE* fid) {
	int num_msb, num_lsb = 0;
	if (fread(&num_msb, sizeof(int), 1, fid) != 1) {
		THROW(KERR_ASSERT);
	}
	num_lsb |= (num_msb & 0x000000FF) << 24;
	num_lsb |= (num_msb & 0x0000FF00) << 8;
	num_lsb |= (num_msb & 0x00FF0000) >> 8;
	num_lsb |= (num_msb & 0xFF000000) >> 24;

	return num_lsb;
}

void MnistReaderFeeder::m_load_image_file(KString filepath) {
	FILE* fid = Utils::fopen(filepath.c_str(), "rb");
	assert(fid != NULL);
	int magic_num = m_fread_int_msb(fid);
	assert(magic_num == 2051);

	int component_num = m_fread_int_msb(fid);
	int row_num = m_fread_int_msb(fid);
	int col_num = m_fread_int_msb(fid);

	assert(row_num == 28);
	assert(col_num == 28);

	int data_size = component_num * row_num * col_num;

	unsigned char* pBuffer = new unsigned char[data_size];

	if (fread(pBuffer, sizeof(unsigned char), data_size, fid) != data_size) THROW(KERR_ASSERT);

	delete[] m_pImages;

	m_data_count = component_num;
	m_pImages = new KFloat[data_size];

	for (int n = 0; n < data_size; n++) m_pImages[n] = (pBuffer[n] - 127.5f) / 127.5f;

	delete[] pBuffer;
	fclose(fid);
}

void MnistReaderFeeder::m_load_label_file(KString filepath) {
	FILE* fid = Utils::fopen(filepath.c_str(), "rb");
	assert(fid != NULL);
	int magic_num = m_fread_int_msb(fid);
	assert(magic_num == 2049);

	int component_num = m_fread_int_msb(fid);
	assert(component_num == m_data_count);

	delete[] m_pLabels;

	m_pLabels = new unsigned char[component_num];
	if (fread(m_pLabels, sizeof(unsigned char), component_num, fid) != component_num) THROW(KERR_ASSERT);

	fclose(fid);
}

bool MnistReaderFeeder::m_load_cache(KString cache_path) {
	FILE* fid = Utils::fopen(cache_path + "/mnist_reader.dat", "rb", false);

	if (fid == NULL) return false;

	printf("loading mnist reader cache data...\n");

	int checkCode;
	int version;
	int data_count;

	if (fread(&checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fread(&version, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fread(&data_count, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);

	if (checkCode != ms_checkCode) { fclose(fid); return false; }
	if (version != m_version) { fclose(fid); return false; }

	m_data_count = data_count;

	delete[] m_pImages;
	delete[] m_pLabels;

	int data_size = m_data_count * 28 * 28;

	m_pImages = new KFloat[data_size];
	m_pLabels = new unsigned char[m_data_count];

	if (fread(m_pImages, sizeof(KFloat), data_size, fid) != data_size) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fread(m_pLabels, sizeof(unsigned char), m_data_count, fid) != m_data_count) THROW(KERR_FAILURE_ON_FILE_SAVE);

	fclose(fid);

	return true;
}

void MnistReaderFeeder::m_save_cache(KString cache_path) {
	printf("saving mnist reader cache data...\n");

	FILE* fid = Utils::fopen(cache_path + "/mnist_reader.dat", "wb");

	if (fid == NULL) THROW(KERR_FAILURE_ON_FILE_SAVE);

	if (fwrite(&ms_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_version, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_data_count, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);

	int data_size = m_data_count * 28 * 28;

	if (fwrite(m_pImages, sizeof(KFloat), data_size, fid) != data_size) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(m_pLabels, sizeof(unsigned char), m_data_count, fid) != m_data_count) THROW(KERR_FAILURE_ON_FILE_SAVE);

	fclose(fid);
}

void MnistReaderFeeder::m_fill_digit_words() {
	static const char* words[] = { "zero", "one", "two", "three", "four","five", "six", "seven", "eight", "nine" };

	int check = sizeof(m_digit_words);

	memset(m_digit_words, 0, sizeof(m_digit_words));

	for (int n = 0; n < 10; n++) {
		int word_len = (int) strlen(words[n]);
		for (int k = 0; k < 6; k++) {
			int alphabet = (k < word_len) ? words[n][k] - 'a' + 1 : 0;
			m_digit_words[n][k][alphabet] = 1.0f;
		}
	}
}

KBool MnistReaderFeeder::m_getDataCount(void* pAux, KInt* pnDataCount) {
	*pnDataCount = m_data_count;
	return true;
}

KBool MnistReaderFeeder::m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) {
	pFieldInfo->m_bIsFloat = true;
	pFieldInfo->m_shape = bInput ? KaiShape{ 28 * 28 } : KaiShape{ 27 };
	pFieldInfo->m_bIsSeq = bInput ? false : true;
	pFieldInfo->m_nTimesteps = bInput ? 0 : 6;

	return true;
}

KBool MnistReaderFeeder::m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	THROW(KERR_INTERNAL_LOGIC_ERROR);
	return false;
}

KBool MnistReaderFeeder::m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
	KInt dat_size = shape.total_size();
	if (bInput) {
		for (KInt n = 0; n < (KInt)nDatIndexs.size(); n++) {
			KInt idx = nDatIndexs[n];
			memcpy(pfBuffer + n * dat_size, m_pImages + idx * dat_size, dat_size * sizeof(KFloat));
		}
	}
	else {
		for (KInt n = 0; n < (KInt)nDatIndexs.size(); n++) {
			KInt idx = nDatIndexs[n];
			KInt nChar = m_pLabels[idx];
			assert(nChar >= 0 && nChar < 10);
			memcpy(pfBuffer + n * dat_size, m_digit_words[n], dat_size * sizeof(KFloat));
		}
	}

	return true;
}
