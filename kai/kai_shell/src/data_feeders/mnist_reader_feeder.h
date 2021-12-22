/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../data_feeder.h"
#include "../utils/utils.h"

class MnistReaderFeeder : public DataFeeder {
public:
	MnistReaderFeeder();
	virtual ~MnistReaderFeeder();

	virtual void loadData(KString data_path, KString cache_path);

protected:
	virtual KBool m_getDataCount(void* pAux, KInt* pnDataCount);
	virtual KBool m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);
	virtual KBool m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	virtual KBool m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);

protected:
	void m_load_data(KString data_path);

	bool m_load_cache(KString cache_path);
	void m_save_cache(KString cache_path);

	int m_version;
	static int ms_checkCode;

protected:
	int m_data_count;

	KFloat* m_pImages;
	unsigned char* m_pLabels;

	KFloat m_digit_words[10][6][27];

	void m_load_image_file(KString filepath);
	void m_load_label_file(KString filepath);

	int m_fread_int_msb(FILE* fid);

	void m_fill_digit_words();
};
