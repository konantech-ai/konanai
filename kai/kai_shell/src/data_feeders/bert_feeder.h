/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../data_feeder.h"
#include "../utils/utils.h"

class BertFeeder : public DataFeeder {
public:
	BertFeeder();
	virtual ~BertFeeder();

	void setModel(KString sub_model) { m_sub_model = sub_model; }
	void setMaxPosition(KInt max_position) { m_max_position = max_position; }

	virtual void loadData(KString data_path, KString cache_path);

	KInt voc_count() { return m_voc_count; }

	KString getTokenWord(KInt token) { return m_idx_to_word[token]; }

protected:
	virtual KBool m_getDataCount(void* pAux, KInt* pnDataCount);
	virtual KBool m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod);
	virtual KBool m_getSecDataCount(void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount);
	virtual KBool m_getSecBatchSize(void* pAux, Ken_data_section section, KInt* pnBatchSize);
	virtual KBool m_getExtraFields(void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields);
	virtual KBool m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);
	virtual KBool m_informDataIndexes(void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount);
	virtual KBool m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	virtual KBool m_feedFreeIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer);
	virtual KBool m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);
	virtual KBool m_feedFreeFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer);

protected:
	int m_version;
	static int ms_checkCode;

protected:
	KString m_sub_model;

	KString m_sDataName;

protected:
	KInt m_max_position;

	KInt m_dat_count;
	KInt m_voc_count;
	
	KInt m_tr_start;
	KInt m_va_start;
	KInt m_te_start;

	std::map<std::string,KInt> m_word_to_idx;
	std::map<KInt,std::string> m_idx_to_word;

	std::vector<std::vector<KInt>> m_sentences;

	KInt m_EMT;
	KInt m_CLS;
	KInt m_SEP;
	KInt m_MSK;

	void m_load_data(KString data_path);

	bool m_load_cache(KString cache_path);
	void m_save_cache(KString cache_path);

	void m_reset();

	KInt m_load_text_file(KString data_path, KString filename);

protected:
	KInt m_nMarkCount;

	KInt* m_pTokens;
	KInt* m_pMaskIndex;
	KInt* m_pMaskedWords;

	KFloat* m_pNextSent;

	void m_fill_sentence(KInt end_pos, KInt sent_idx, KInt sent_mark, KInt& pos);

};
