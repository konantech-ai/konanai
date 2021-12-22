/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "data_feeder.h"
#include "utils/utils.h"

#include <algorithm>

DataFeeder::DataFeeder() {
}

DataFeeder::~DataFeeder() {
}

void DataFeeder::ConnectToKai(KHSession hSession, KHDataset hDataset) {
	// Added by Hyung-jae, Son (2021-09-14)
	void* param = (void*)hDataset;

	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::get_data_cnt, this, reinterpret_cast<void*>(ms_cbGetDataCount), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::get_suffle_method, this, reinterpret_cast<void*>(ms_cbGetDataSuffleMethod), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::get_sec_data_cnt, this, reinterpret_cast<void*>(ms_cbGetSecDataCount), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::get_sec_batch_size, this, reinterpret_cast<void*>(ms_cbGetSecBatchSize), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::get_extra_fields, this, reinterpret_cast<void*>(ms_cbGetExtraFields), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::get_field_spec, this, reinterpret_cast<void*>(ms_cbGetFieldSpec), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::inform_data_indexes, this, reinterpret_cast<void*>(ms_cbInformDataIndexes), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::feed_int_data, this, reinterpret_cast<void*>(ms_cbFeedIntData), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::feed_int_free_data, this, reinterpret_cast<void*>(ms_cbFreeFeedIntData), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::feed_float_data, this, reinterpret_cast<void*>(ms_cbFeedFloatData), param));
	KERR_CHK(KAI_Component_set_datafeed_callback(hDataset, Ken_datafeed_cb_event::feed_float_free_data, this, reinterpret_cast<void*>(ms_cbFeedFreeFloatData), param));
}

KBool DataFeeder::ms_cbGetDataCount(void* pInst, void* pAux, KInt* pnDataCount) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_getDataCount(pAux, pnDataCount);
}

KBool DataFeeder::ms_cbGetDataSuffleMethod(void* pInst, void* pAux, Ken_data_suffle_method* pSuffleMethod) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_getDataSuffleMethod(pAux, pSuffleMethod);
}

KBool DataFeeder::ms_cbGetSecDataCount(void* pInst, void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_getSecDataCount(pAux, section, nDatCount, pnDataCount);
}

KBool DataFeeder::ms_cbGetSecBatchSize(void* pInst, void* pAux, Ken_data_section section, KInt* pnBatchSize) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_getSecBatchSize(pAux, section, pnBatchSize);
}

KBool DataFeeder::ms_cbGetExtraFields(void* pInst, void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_getExtraFields(pAux, pbUseDefInput, psInputFields, pbUseDefOutput, psOutputFields);
}

KBool DataFeeder::ms_cbGetFieldSpec(void* pInst, void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_getFieldSpec(pAux, bInput, sFieldName, pFieldInfo, pMatchInfo);
}

KBool DataFeeder::ms_cbInformDataIndexes(void* pInst, void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_informDataIndexes(pAux, nDatIndexs, nRangeStart, nRangeCount);
}

KBool DataFeeder::ms_cbFeedIntData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_feedIntData(pAux, bInput, sFieldName, shape, nDatIndexs, pnBuffer);
}

KBool DataFeeder::ms_cbFreeFeedIntData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_feedFreeIntData(pAux, bInput, sFieldName, shape, nDatIndexs, pnBuffer);
}

KBool DataFeeder::ms_cbFeedFloatData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_feedFloatData(pAux, bInput, sFieldName, shape, nDatIndexs, pfBuffer);
}

KBool DataFeeder::ms_cbFeedFreeFloatData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer) {
	DataFeeder* pInstance = (DataFeeder*)pInst;
	return pInstance->m_feedFreeFloatData(pAux, bInput, sFieldName, shape, nDatIndexs, pfBuffer);
}

KBool DataFeeder::m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod) {
	*pSuffleMethod = Ken_data_suffle_method::random;
#ifdef SON_BAD_TOUCH
	////////////////////////////////////////////////////////////////
	// Before modification
	//*pSuffleMethod = Ken_data_suffle_method::random;
	////////////////////////////////////////////////////////////////

	// Added by Hyung-jae, Son (2021-09-01)
	KHDataset hDataset = (KHDataset)pAux;
	KaiValue shuffle_method;
	if (KAI_Component_get_property(hDataset, "data_split", &shuffle_method) != KRetOK
		|| shuffle_method.type() != Ken_value_type::string) {
		//THROW (KERR_BAD_TYPE_IN_VALUE_CONVERSION);
		// Default setting
		*pSuffleMethod = Ken_data_suffle_method::random;
		return true;
	}

	if (shuffle_method == KString("sequential"))
		*pSuffleMethod = Ken_data_suffle_method::sequential;
	if (shuffle_method == KString("random"))
		*pSuffleMethod = Ken_data_suffle_method::random;
#endif
	
	return true;
}

KBool DataFeeder::m_getSecDataCount(void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount) {
	switch (section) {
	case Ken_data_section::train:
		*pnDataCount = (KInt)(nDatCount * 0.8);
		break;
	case Ken_data_section::validate:
		*pnDataCount = (KInt)(nDatCount * 0.1);
		break;
	case Ken_data_section::test:
		*pnDataCount = (KInt)(nDatCount * 0.1);
		break;
	}
#ifdef SON_BAD_TOUCH
	////////////////////////////////////////////////////////////////
	// Before modification
	/*
	switch (section) {
	case Ken_data_section::train:
		*pnDataCount = (KInt)(nDatCount * 0.8);
		break;
	case Ken_data_section::validate:
		*pnDataCount = (KInt)(nDatCount * 0.1);
		break;
	case Ken_data_section::test:
		*pnDataCount = (KInt)(nDatCount * 0.1);
		break;
	}
	*/
	////////////////////////////////////////////////////////////////
	
	// Modified by Hyung-jae, Son (2021-09-01)
	
	// Get user-defined ratios
	KHDataset hDataset = (KHDataset)pAux;
	KaiValue tr_ratio, te_ratio, va_ratio;
	if (KAI_Component_get_property(hDataset, "tr_ratio", &tr_ratio) != KRetOK
		|| KAI_Component_get_property(hDataset, "te_ratio", &te_ratio) != KRetOK
		|| KAI_Component_get_property(hDataset, "va_ratio", &va_ratio) != KRetOK) {
		//THROW (KERR_KEY_NOT_FOUND_ON_GET_POPERTY);
		// Default setting
		switch (section) {
		case Ken_data_section::train:
			*pnDataCount = (KInt)(nDatCount * 0.8);
			break;
		case Ken_data_section::validate:
			*pnDataCount = (KInt)(nDatCount * 0.1);
			break;
		case Ken_data_section::test:
			*pnDataCount = (KInt)(nDatCount * 0.1);
			break;
		}
		return true;
	}

	// Convert the values to data counts
	KInt tr_count = (KInt)((KFloat)nDatCount * (KFloat)tr_ratio);
	KInt te_count = (KInt)((KFloat)nDatCount * (KFloat)te_ratio);

	// Check the validation
	if ((KFloat)tr_ratio + (KFloat)te_ratio >= 1.0f || tr_count + te_count >= nDatCount)
		THROW (KERR_INDEX_OUT_OF_RANGE);

	switch (section) {
	case Ken_data_section::train:
		*pnDataCount = tr_count;
		break;
	case Ken_data_section::test:
		*pnDataCount = te_count;
		break;
	case Ken_data_section::validate:
		{
			KInt va_count = (KInt)((KFloat)nDatCount * (KFloat)va_ratio);

			// Auto correction
			if (tr_count + te_count + va_count != nDatCount) {
				printf("[NOTE] The count of validate data is automatically corrected from %lld to %lld.\n",
					va_count, nDatCount - (tr_count + te_count));
				va_count = nDatCount - (tr_count + te_count);
			}

			*pnDataCount = va_count;
		}
		break;
	}
	
	////////////////////////////////////////////////////////////////

	#endif
	return true;
}

KBool DataFeeder::m_getSecBatchSize(void* pAux, Ken_data_section section, KInt* pnBatchSize) {
	*pnBatchSize = 10;
#ifdef SON_BAD_TOUCH
	////////////////////////////////////////////////////////////////
	// Before modification
	//*pnBatchSize = 10;
	////////////////////////////////////////////////////////////////

	// Modified by Hyung-jae, Son (2021-09-01)
	
	// Get the target keyword
	KString sKey;
	switch (section) {
	case Ken_data_section::train:
		sKey = KString("tr_batch_size");
		break;
	case Ken_data_section::test:
		sKey = KString("te_batch_size");
		break;
	case Ken_data_section::validate:
		sKey = KString("va_batch_size");
		break;
	}

	// Get user-defined batch size
	KHDataset hDataset = (KHDataset)pAux;
	KInt batch_size = 0;
	if (KAI_Component_get_int_property(hDataset, sKey, &batch_size) != KRetOK) {
		//THROW (KERR_KEY_NOT_FOUND_ON_GET_POPERTY);
		// Default setting
		*pnBatchSize = 10;
		return true;
	}
	*pnBatchSize = batch_size;
	
	////////////////////////////////////////////////////////////////
#endif

	return true;
}

KBool DataFeeder::m_getExtraFields(void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields) {
	return true;
}

KBool DataFeeder::m_informDataIndexes(void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount) {
	return true;
}

KBool DataFeeder::m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	THROW(KERR_CALLBACK_NOT_DEFINED_FOR_FEEDING_DATA);
	return false;
}

KBool DataFeeder::m_feedFreeIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer) {
	THROW(KERR_CALLBACK_NOT_DEFINED_FOR_FEEDING_DATA);
	return false;
}

KBool DataFeeder::m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
	THROW(KERR_CALLBACK_NOT_DEFINED_FOR_FEEDING_DATA);
	return false;
}

KBool DataFeeder::m_feedFreeFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer) {
	THROW(KERR_CALLBACK_NOT_DEFINED_FOR_FEEDING_DATA);
	return false;
}

