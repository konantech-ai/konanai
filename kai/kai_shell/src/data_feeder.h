/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class DataFeeder {
public:
	DataFeeder();
	virtual ~DataFeeder();

	void ConnectToKai(KHSession hSession, KHDataset );

protected:
	static KBool ms_cbGetDataCount(void* pInst, void* pAux, KInt* pnDataCount);
	static KBool ms_cbGetDataSuffleMethod(void* pInst, void* pAux, Ken_data_suffle_method* pSuffleMethod);
	static KBool ms_cbGetSecDataCount(void* pInst, void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount);
	static KBool ms_cbGetSecBatchSize(void* pInst, void* pAux, Ken_data_section section, KInt* pnBatchSize);
	static KBool ms_cbGetExtraFields(void* pInst, void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields);
	static KBool ms_cbGetFieldSpec(void* pInst, void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);
	static KBool ms_cbInformDataIndexes(void* pInst, void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount);
	static KBool ms_cbFeedIntData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	static KBool ms_cbFreeFeedIntData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer);
	static KBool ms_cbFeedFloatData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);
	static KBool ms_cbFeedFreeFloatData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer);

	virtual KBool m_getDataCount(void* pAux, KInt* pnDataCount) = 0;
	virtual KBool m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod);
	virtual KBool m_getSecDataCount(void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount);
	virtual KBool m_getSecBatchSize(void* pAux, Ken_data_section section, KInt* pnBatchSize);
	virtual KBool m_getExtraFields(void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields);
	virtual KBool m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) = 0;
	virtual KBool m_informDataIndexes(void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount);
	virtual KBool m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	virtual KBool m_feedFreeIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer);
	virtual KBool m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);
	virtual KBool m_feedFreeFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer);

protected:
};

