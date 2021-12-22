/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "bert_feeder.h"

// Skip warning messages when using strtok()
#pragma warning(disable:4996)

int BertFeeder::ms_checkCode = 53118583;

BertFeeder::BertFeeder() : DataFeeder() {
	m_version = 3;
	m_voc_count = 0;
	m_max_position = 0;

	m_nMarkCount = 0;

	m_pTokens = 0;
	m_pMaskIndex = 0;
	m_pMaskedWords = 0;
	m_pNextSent = 0;

	m_EMT = 0;
	m_CLS = 1;
	m_SEP = 2;
	m_MSK = 3;
}

BertFeeder::~BertFeeder() {
	delete[] m_pTokens;
	delete[] m_pMaskIndex;
	delete[] m_pMaskedWords;
	delete[] m_pNextSent;
}

void BertFeeder::loadData(KString data_path, KString cache_path) {
	Utils::mkdir(cache_path);

	if (m_sub_model == "ptb_large" || m_sub_model == "ptb_small") m_sDataName = "ptb";
	else if (m_sub_model == "eng_mini") m_sDataName = "eng";

	KString cache_file = cache_path + "/" + m_sDataName + ".cache";;

	if (!m_load_cache(cache_file)) {
		m_load_data(data_path);
		m_save_cache(cache_file);
	}

	/*
	for (auto& it : m_idx_to_word) {
		printf("Word[%lld] %s\n", it.first, it.second.c_str());
	}
	*/
}

void BertFeeder::m_load_data(KString data_path) {
	printf("loading bert information file...\n");

	m_word_to_idx["[EMT]"] = 0;
	m_word_to_idx["[CLS]"] = 1;
	m_word_to_idx["[SEP]"] = 2;
	m_word_to_idx["[MSK]"] = 3;

	m_idx_to_word[0] = "[EMT]";
	m_idx_to_word[1] = "[CLS]";
	m_idx_to_word[2] = "[SEP]";
	m_idx_to_word[3] = "[MSK]";

	m_voc_count = 4;

	m_tr_start = 0;
	m_va_start = m_load_text_file(data_path, m_sDataName + ".train.txt");
	m_te_start = m_load_text_file(data_path, m_sDataName + ".valid.txt");
	m_dat_count = m_load_text_file(data_path, m_sDataName + ".test.txt");
}

KInt BertFeeder::m_load_text_file(KString data_path, KString filename) {
	KString filepath = data_path + "/" + filename;
	FILE* fid = Utils::fopen(filepath.c_str(), "rt");

	KString sLine;

	char buffer[4096];

	while (fgets(buffer, 4096, fid) != NULL) {
		std::vector<KInt> sentence;

		char* pWord = strtok(buffer, " \n");

		while (pWord != NULL) {
			KString word = pWord;

			KInt idx;

			if (m_word_to_idx.find(word) == m_word_to_idx.end()) {
				idx = m_voc_count++;
				m_word_to_idx[word] = idx;
				m_idx_to_word[idx] = word;
			}
			else {
				idx = m_word_to_idx[word];
			}

			sentence.push_back(idx);

			pWord = strtok(NULL, " \n");
		}

		m_sentences.push_back(sentence);
	}

	fclose(fid);

	if (m_sentences.size() % 2 != 0) m_sentences.pop_back();

	return (KInt)m_sentences.size();
}

bool BertFeeder::m_load_cache(KString cache_file) {
	FILE* fid = Utils::fopen(cache_file.c_str(), "rb", false);

	if (fid == NULL) return false;

	KInt version;
	KInt checkCode;

	fread(&version, sizeof(KInt), 1, fid);
	fread(&checkCode, sizeof(KInt), 1, fid);

	if (version != m_version || checkCode != ms_checkCode) {
		fclose(fid);
		return false;
	}

	fread(&m_dat_count, sizeof(KInt), 1, fid);
	fread(&m_voc_count, sizeof(KInt), 1, fid);

	fread(&m_tr_start, sizeof(KInt), 1, fid);
	fread(&m_va_start, sizeof(KInt), 1, fid);
	fread(&m_te_start, sizeof(KInt), 1, fid);

	for (KInt n = 0; n < m_voc_count; n++) {
		char buffer[256];

		KInt word_len;
		KInt word_idx;

		fread(&word_len, sizeof(KInt), 1, fid);
		fread(buffer, sizeof(char), word_len, fid);

		std::string word(buffer, word_len);

		fread(&word_idx, sizeof(KInt), 1, fid);

		if (word_idx != n) return false;

		m_idx_to_word[n] = word;
		m_word_to_idx[word] = word_idx;
	}

	fread(&checkCode, sizeof(KInt), 1, fid);
	if (checkCode != ms_checkCode) {
		fclose(fid);
		m_reset();
		return false;
	}

	for (KInt n = 0; n < m_dat_count; n++) {
		std::vector<KInt > sentence;

		KInt sent_len;

		fread(&sent_len, sizeof(KInt), 1, fid);

		for (KInt m = 0; m < sent_len; m++) {
			KInt wid;
			fread(&wid, sizeof(KInt), 1, fid);
			sentence.push_back(wid);
		}

		m_sentences.push_back(sentence);
	}

	fread(&checkCode, sizeof(KInt), 1, fid);
	if (checkCode != ms_checkCode) {
		fclose(fid);
		m_reset();
		return false;
	}

	fclose(fid);

	return true;
}

void BertFeeder::m_save_cache(KString cache_file) {
	printf("saving bert cache data...\n");

	FILE* fid = Utils::fopen(cache_file.c_str(), "wb");

	fwrite(&m_version, sizeof(KInt), 1, fid);
	fwrite(&ms_checkCode, sizeof(KInt), 1, fid);

	fwrite(&m_dat_count, sizeof(KInt), 1, fid);
	fwrite(&m_voc_count, sizeof(KInt), 1, fid);

	fwrite(&m_tr_start, sizeof(KInt), 1, fid);
	fwrite(&m_va_start, sizeof(KInt), 1, fid);
	fwrite(&m_te_start, sizeof(KInt), 1, fid);

	assert(m_word_to_idx.size() == m_voc_count);
	assert(m_idx_to_word.size() == m_voc_count);

	for (KInt n = 0; n < m_voc_count; n++) {
		std::string word = m_idx_to_word[n];

		KInt word_len = (KInt)strlen(word.c_str());
		KInt word_idx = m_word_to_idx[word];

		fwrite(&word_len, sizeof(KInt), 1, fid);
		fwrite(word.c_str(), sizeof(char), word_len, fid);

		fwrite(&word_idx, sizeof(KInt), 1, fid);
	}

	assert(m_sentences.size() == m_dat_count);

	fwrite(&ms_checkCode, sizeof(KInt), 1, fid);

	for (KInt n = 0; n < m_dat_count; n++) {
		std::vector<KInt > sentence = m_sentences[n];

		KInt sent_len = (KInt)sentence.size();

		fwrite(&sent_len, sizeof(KInt), 1, fid);

		for (KInt m = 0; m < sent_len; m++) {
			KInt wid = sentence[m];
			fwrite(&wid, sizeof(KInt), 1, fid);
		}
	}

	fwrite(&ms_checkCode, sizeof(KInt), 1, fid);

	fclose(fid);
}

void BertFeeder::m_reset() {
	m_dat_count = 0;
	m_voc_count = 0;

	m_tr_start = 0;
	m_va_start = 0;
	m_te_start = 0;

	m_word_to_idx.clear();
	m_idx_to_word.clear();

	m_sentences.clear();
}

KBool BertFeeder::m_getDataCount(void* pAux, KInt* pnDataCount) {
	*pnDataCount = m_dat_count / 2;
	return true;
}

KBool BertFeeder::m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod) {
	*pSuffleMethod = Ken_data_suffle_method::sequential;
	return true;
}

KBool BertFeeder::m_getSecDataCount(void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount) {
	if (section == Ken_data_section::train) *pnDataCount = m_va_start / 2;
	else if (section == Ken_data_section::validate) *pnDataCount = (m_te_start - m_va_start) / 2;
	else if (section == Ken_data_section::test) *pnDataCount = (m_dat_count - m_te_start) / 2;
	return true;
}

KBool BertFeeder::m_getSecBatchSize(void* pAux, Ken_data_section section, KInt* pnBatchSize) {
	*pnBatchSize = (m_sub_model != "eng_mini") ? 10 : 2;
	return true;
}

KBool BertFeeder::m_getExtraFields(void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields) {
	*pbUseDefInput = false;
	*pbUseDefOutput = false;

	psInputFields->push_back("tokens");
	psInputFields->push_back("mask_index");

	psOutputFields->push_back("next_sent");
	psOutputFields->push_back("masked_words");

	return true;
}

KBool BertFeeder::m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) {
	if (sFieldName == "tokens") {	// 입력 토큰들, [mbsize, maxpos, 3]의 형상을 가지며 embed 레이어에 (단어, 위치, 문장구분)의 튜플 정보를 알려준다.
		pFieldInfo->m_bIsFloat = false;
		pFieldInfo->m_shape = KaiShape{ 3 };
		pFieldInfo->m_bIsSeq = true;
		pFieldInfo->m_nTimesteps = m_max_position;
	}
	else if (sFieldName == "mask_index") {	// 마스크 처리된 위치를 나타내는 플래그 정보, [mbsize, maxpos, 1]의 형상을 가지며 채점용 정보이지만 마지막 select 레이어에 제공되어 필요한 정보만 걸러내게 한다.
		pFieldInfo->m_bIsFloat = false;
		pFieldInfo->m_shape = KaiShape{ 1 };
		pFieldInfo->m_bIsSeq = true;
		pFieldInfo->m_nTimesteps = m_max_position;
		pFieldInfo->m_bFreeShape = true;
	}
	else if (sFieldName == "next_sent") {	// 인접 문장 여부를 나타내는 정답 정보, [mbsize, 2]의 형상
		pFieldInfo->m_bIsFloat = true;
		pFieldInfo->m_shape = KaiShape{ 2 };
		pFieldInfo->m_bIsSeq = false;
		pFieldInfo->m_nTimesteps = 0;
	}
	else if (sFieldName == "masked_words") {	// 마스크된 단어들의 실제 단어를 나타내는 정답 정보, [가변크기, 2]의 형상
		pFieldInfo->m_bIsFloat = false;
		pFieldInfo->m_shape = KaiShape{ 1 };
		pFieldInfo->m_bIsSeq = true;
		pFieldInfo->m_nTimesteps = m_max_position;
		pFieldInfo->m_bFreeShape = true;
	}

	return true;
}

KBool BertFeeder::m_informDataIndexes(void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount) {
	m_nMarkCount = 0;

	while (m_nMarkCount < 3) {
		m_nMarkCount = 0;

		delete[] m_pTokens;
		delete[] m_pMaskIndex;
		delete[] m_pMaskedWords;
		delete[] m_pNextSent;

		KInt mb_size = nDatIndexs.size();

		m_pTokens = new KInt[mb_size * m_max_position * 3];
		m_pMaskIndex = new KInt[mb_size * m_max_position];
		m_pMaskedWords = new KInt[mb_size * m_max_position];
		m_pNextSent = new KFloat[mb_size * 2];

		memset(m_pTokens, 0, sizeof(KInt) * mb_size * m_max_position * 3);
		memset(m_pMaskIndex, 0, sizeof(KInt) * mb_size * m_max_position);
		memset(m_pMaskedWords, 0, sizeof(KInt) * mb_size * m_max_position);
		memset(m_pNextSent, 0, sizeof(KFloat) * mb_size * 2);

		KInt nFrom, nTo;
		KInt pos = 0, start_pos = 0, end_pos = m_max_position;
		KInt toke_cnt = 0;

		for (KInt n = 0; n < mb_size; n++) {
			KInt sent1_idx = nDatIndexs[n] * 2;
			KInt sent2_idx = sent1_idx + 1;

			KBool next_sent = (rand() % 2 == 0);

			m_pNextSent[n * 2 + (next_sent ? 0 : 1)] = 1.0f;

			if (!next_sent) {
				if (sent1_idx < m_va_start) nFrom = 0, nTo = m_va_start;
				else if (sent1_idx < m_te_start) nFrom = m_va_start, nTo = m_te_start;
				else nFrom = m_te_start, nTo = m_dat_count;

				sent2_idx = (sent1_idx + 3 + rand() % ((nTo - nFrom - 2) / 2) * 2);
				if (sent2_idx >= nTo) sent2_idx -= nTo - nFrom;

			}

			pos = start_pos;

			m_pTokens[pos++ * 3] = m_CLS;

			if (pos < end_pos) m_fill_sentence(end_pos, sent1_idx, 1, pos);
			if (pos < end_pos) m_fill_sentence(end_pos, sent2_idx, 2, pos);

			assert(pos <= end_pos);

			toke_cnt += pos - start_pos;

			start_pos = end_pos;
			end_pos += m_max_position;
		}
	}

	//printf("pos = %lld, tokens = %lld, m_nMarkCount = %lld\n", pos, toke_cnt, m_nMarkCount);

	return true;
}

KBool BertFeeder::m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	KInt mb_size = nDatIndexs.size();
	KInt dat_size = shape.total_size();

	if (sFieldName == "tokens") {
		memcpy(pnBuffer, m_pTokens, sizeof(KInt) * mb_size * dat_size);
	}
	else THROW(KERR_INTERNAL_LOGIC_ERROR);

	return true;
}

KBool BertFeeder::m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
	KInt mb_size = nDatIndexs.size();
	KInt dat_size = shape.total_size();

	if (sFieldName == "next_sent") {
		memcpy(pfBuffer, m_pNextSent, sizeof(KFloat) * mb_size * dat_size);
	}
	else THROW(KERR_INTERNAL_LOGIC_ERROR);

	return true;
}

KBool BertFeeder::m_feedFreeIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer) {
	if (sFieldName == "mask_index") {
		pnBuffer = m_pMaskIndex;
		shape = KaiShape{ m_nMarkCount };
	}
	else if (sFieldName == "masked_words") {
		pnBuffer = m_pMaskedWords;
		shape = KaiShape{ m_nMarkCount };
	}
	else THROW(KERR_INTERNAL_LOGIC_ERROR);

	return true;
}

KBool BertFeeder::m_feedFreeFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer) {
	THROW(KERR_INTERNAL_LOGIC_ERROR);

	return true;
}

void BertFeeder::m_fill_sentence(KInt end_pos, KInt sent_idx, KInt sent_mark, KInt& pos) {
	std::vector<KInt> sent = m_sentences[sent_idx];

	for (KInt n = 0; n < (KInt)sent.size(); n++) {
		if (sent[n] < 4 || sent[n] >= m_voc_count) THROW(KERR_ASSERT);

		m_pTokens[pos * 3] = sent[n];
		m_pTokens[pos * 3 + 1] = pos % m_max_position;
		m_pTokens[pos * 3 + 2] = sent_mark;

		if (rand() % 100 < 15) {
			m_pMaskIndex[m_nMarkCount] = pos;
			m_pMaskedWords[m_nMarkCount] = sent[n];

			m_nMarkCount++;

			if (rand() % 100 < 80) {
				m_pTokens[pos * 3] = m_MSK;
			}
			else if (rand() % 100 < 50) {
				KInt other_token = sent[n] + rand() % (m_voc_count - 5);
				if (other_token >= m_voc_count) other_token -= (m_voc_count - 4);
				
				if (other_token < 4 || other_token >= m_voc_count) THROW(KERR_ASSERT);

				m_pTokens[pos * 3] = other_token;
			}
		}

		pos++;

		if (pos >= end_pos) return;
	}

	m_pTokens[pos * 3] = m_SEP;
	m_pTokens[pos * 3 + 1] = pos % m_max_position;
	m_pTokens[pos * 3 + 2] = sent_mark;

	pos++;
}
