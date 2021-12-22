/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../data_feeder.h"
#include "../utils/utils.h"

class _WavedataPicker {
public:
	_WavedataPicker(WaveInfo* pWaveInfo, KInt start_offset, KInt nFetchRate);
	virtual ~_WavedataPicker();

	float fetch(KInt nth);

protected:
	float m_get_sample(KInt nth);

	KInt m_start_offset;
	KInt m_fetch_rate;

	WaveInfo* m_pWaveInfo;
};

class UrbanDataFeeder : public DataFeeder {
public:
	UrbanDataFeeder(KHSession hSession);
	virtual ~UrbanDataFeeder();

	virtual void loadData(KString data_path, KString cache_path);
	virtual KaiList getTargetNames();

protected:
	virtual KBool m_getDataCount(void* pAux, KInt* pnDataCount);
	virtual KBool m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod);
	virtual KBool m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);
	virtual KBool m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	virtual KBool m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);

protected:
	void m_load_data(KString data_path);

	bool m_load_cache(KString cache_path);
	void m_save_cache(KString cache_path);

	void m_load_wave_data(KString wav_path, KFloat* pWaveBuffer, KInt fetch_width, KInt sample_rate);;

	KHSession m_hSession;

	int m_version;
	static int ms_checkCode;

	KaiList m_dat_categorys;
	KaiList m_category_names;

	KInt m_file_count;
	KInt m_step_cnt;
	KInt m_freq_cnt;

	KInt m_data_size;

	KFloat* m_pFFT;
};
