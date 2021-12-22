/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "callback.h"
#include "../exec/exec_context.h"
#include "../math/kmath.h"
#include "../utils/kutil.h"

KaiCallbackAgent::KaiCallbackAgent(KaiModelInstance* pModelInst) {
	KaiDict families = pModelInst->get_property("#callback", KaiDict());
	for (auto& it1 : families) {
		KaiDict events = it1.second;
		std::map<int, KaiList> family;
		for (auto& it2 : events) {
			KaiList cb_infos = it2.second;
			family[std::stoi(it2.first)] = cb_infos;
		}
		m_cbFamilies[std::stoi(it1.first)] = family;
	}
	KaiDict m_cbFamilies = pModelInst->get_property("#callback", KaiDict());
}

KaiCallbackAgent::~KaiCallbackAgent() {
}

KBool KaiCallbackAgent::get_data_count(KInt* pnDataCount, KBool bThrow) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::get_data_cnt, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbGetDataCount* pFunc = (KCbGetDataCount*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (pFunc(pInst, pAux, pnDataCount))
				return true;
		}
	}

	if (bThrow)
		throw KaiException(KERR_NO_ACTIVE_CALLBACK_FUNCTION, "GetDataCount");

	return false;
}

KBool KaiCallbackAgent::get_data_suffle_method(Ken_data_suffle_method* penumSuffleMethod, KBool bThrow) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::get_suffle_method, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbGetDataSuffleMethod* pFunc = (KCbGetDataSuffleMethod*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (pFunc(pInst, pAux, penumSuffleMethod)) return true;
		}
	}

	if (bThrow) throw KaiException(KERR_NO_ACTIVE_CALLBACK_FUNCTION, "GetDataSuffleMethod");

	return false;
}

KBool KaiCallbackAgent::get_section_data_count(KInt nDataCount, KInt* pnTrCount, KInt* pnTeCount, KInt* pnVaCount, KBool bThrow) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::get_sec_data_cnt, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbGetSectionDataCount* pFunc = (KCbGetSectionDataCount*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (!pFunc(pInst, pAux, Ken_data_section::train, nDataCount, pnTrCount)) break;
			if (!pFunc(pInst, pAux, Ken_data_section::test, nDataCount, pnTeCount)) break;
			if (!pFunc(pInst, pAux, Ken_data_section::validate, nDataCount, pnVaCount)) break;
		}
	}

	if (bThrow) throw KaiException(KERR_NO_ACTIVE_CALLBACK_FUNCTION, "GetSectionDataCount");

	return false;
}

KBool KaiCallbackAgent::get_section_batch_size(KInt* pnTrBatch, KInt* pnTeBatch, KInt* pnVaBatch, KBool bThrow) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::get_sec_batch_size, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbGetSectionBatchSize* pFunc = (KCbGetSectionBatchSize*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (!pFunc(pInst, pAux, Ken_data_section::train, pnTrBatch)) break;
			if (!pFunc(pInst, pAux, Ken_data_section::test, pnTeBatch)) break;
			if (!pFunc(pInst, pAux, Ken_data_section::validate, pnVaBatch)) break;
		}
	}

	if (bThrow) throw KaiException(KERR_NO_ACTIVE_CALLBACK_FUNCTION, "GetSectionBatchSize");

	return false;
}

KBool KaiCallbackAgent::get_extra_fields(KStrList& inFieldNames, KStrList& outFieldNames, KBool bThrow) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::get_extra_fields, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbGetExtraFields* pFunc = (KCbGetExtraFields*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			KBool bUseDefInput = true;
			KBool bUseDefOutput = true;

			if (pFunc(pInst, pAux, &bUseDefInput, &inFieldNames, &bUseDefOutput, &outFieldNames)) {
				if (bUseDefInput) inFieldNames.push_back("#default");
				if (bUseDefOutput) outFieldNames.push_back("#default");

				return true;
			}
		}
	}

	if (bThrow) throw KaiException(KERR_NO_ACTIVE_CALLBACK_FUNCTION, "GetExtraFields");

	inFieldNames.push_back("#default");
	outFieldNames.push_back("#default");

	return false;
}

KBool KaiCallbackAgent::get_field_spec(KBool bInput, KString sFieldName, KaiDict& field_info, KBool bThrow) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::get_field_spec, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbGetFieldSpec* pFunc = (KCbGetFieldSpec*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			KCbFieldInfo fieldInfo;
			KCbFieldInfo matchInfo;

			fieldInfo.m_bIsFloat = true;
			fieldInfo.m_bIsSeq = false;
			fieldInfo.m_nTimesteps = 0;
			fieldInfo.m_bFreeShape = false;

			//matchInfo.m_bFilled = false;

			if (pFunc(pInst, pAux, bInput, sFieldName, &fieldInfo, &matchInfo)) {
				field_info["dtype"] = fieldInfo.m_bIsFloat ? "float" : "int";
				field_info["shape"] = fieldInfo.m_shape;
				field_info["seq"] = fieldInfo.m_bIsSeq;
				field_info["timesteps"] = fieldInfo.m_nTimesteps;
				field_info["feed_shape"] = fieldInfo.m_bIsSeq ? fieldInfo.m_shape.insert_head(fieldInfo.m_nTimesteps) : fieldInfo.m_shape;
				field_info["free_shape"] = fieldInfo.m_bFreeShape;
				//field_info["match_shape"] = matchInfo.m_bFilled ? matchInfo.m_shape : fieldInfo.m_shape;

				return true;
			}
		}
	}

	if (bThrow) throw KaiException(KERR_NO_ACTIVE_CALLBACK_FUNCTION, "GetFieldSpec");

	return false;
}

KBool KaiCallbackAgent::train_start(KString sName, KString sTimestamp, KInt epoch_count, KInt data_count, KInt batch_size, KInt batch_count) {
	KaiList cbFuncs;
	
	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_start, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbTrainStart* pFunc = (KCbTrainStart*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sName, sTimestamp, epoch_count, data_count);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::train_end(KString sName, KString sTimestamp) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_end, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbTrainEnd* pFunc = (KCbTrainEnd*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sName, sTimestamp);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::train_epoch_start(KInt epoch_count, KInt epoch_index, KaiDict tr_data) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_epoch_start, cbFuncs)) {
		KaiCallbackTransaction cbTran;

		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbEpochStart* pFunc = (KCbEpochStart*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sTimestamp, epoch_count, epoch_index, tr_data["batch_size"], cbTran.create_token(tr_data["data_index"]));
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::train_epoch_end(KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_epoch_end, cbFuncs)) {
		KaiCallbackTransaction cbTran;

		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbEpochEnd* pFunc = (KCbEpochEnd*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sTimestamp, epoch_count, epoch_index, dat_count, loss, accuracy);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::train_batch_start(KInt batch_count, KInt batch_index) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_batch_start, cbFuncs)) {
		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbBatchStart* pFunc = (KCbBatchStart*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sTimestamp, batch_count, batch_index);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::train_batch_end(KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_batch_end, cbFuncs)) {
		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbBatchEnd* pFunc = (KCbBatchEnd*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sTimestamp, batch_count, batch_index, batch_size, loss, accuracy);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::train_validate_start(KInt data_count, KInt batch_size) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_validate_start, cbFuncs)) {
		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbValidateStart* pFunc = (KCbValidateStart*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sTimestamp, data_count, batch_size);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::train_validate_end(KaiDict accuracy) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::train, (int)Ken_train_cb_event::train_validate_end, cbFuncs)) {
		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbValidateEnd* pFunc = (KCbValidateEnd*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			pFunc(pInst, pAux, sTimestamp, accuracy);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::test_start(KString sName, KString sTimestamp, KInt data_count, KaiDict te_data) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::test, (int)Ken_test_cb_event::test_start, cbFuncs)) {
		KaiCallbackTransaction cbTran;

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbTestStart* pFunc = (KCbTestStart*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			KaiDict token_info = cbTran.create_token(te_data["data_index"]);

			pFunc(pInst, pAux, sName, sTimestamp, data_count, token_info);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::test_end(KString sName, KString sTimestamp, KaiDict accuracy) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::test, (int)Ken_test_cb_event::test_end, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbTestEnd* pFunc = (KCbTestEnd*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

pFunc(pInst, pAux, sName, sTimestamp, accuracy);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::visualize_start(KString sName, Ken_visualize_mode mode, KInt nDatCount, KaiDict data_info) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::visualize, (int)Ken_visualize_cb_event::visualize_start, cbFuncs)) {
		KaiCallbackTransaction cbTran;

		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbVisualizeStart* pFunc = (KCbVisualizeStart*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			KaiDict token_info = (data_info.size() > 0) ? cbTran.create_token(data_info["data_index"]) : data_info;

			pFunc(pInst, pAux, sName, sTimestamp, mode, nDatCount, token_info);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::visualize_end(KString sName, KaiDict xs, KaiDict ys, KaiDict outs, KaiDict vis_dict) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::visualize, (int)Ken_visualize_cb_event::visualize_end, cbFuncs)) {
		KaiCallbackTransaction cbTran;

		KString sTimestamp = kutil.get_timestamp(time(NULL));

		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbVisualizeEnd* pFunc = (KCbVisualizeEnd*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			KaiDict xs_tokens = cbTran.conv_to_token_dict(xs);
			KaiDict ys_tokens = cbTran.conv_to_token_dict(ys);
			KaiDict os_tokens = cbTran.conv_to_token_dict(outs);
			KaiDict vd_tokens = cbTran.conv_to_token_dict(vis_dict);

			pFunc(pInst, pAux, sName, sTimestamp, xs_tokens, ys_tokens, os_tokens, vd_tokens);
		}

		return true;
	}

	return false;
}

KBool KaiCallbackAgent::m_feed_data(KaiDict& dats, KString sFieldName, KaiDict field_info, KaiList cbFloatFuncs, KaiList cbIntFuncs, KInt mb_size, KBool bInput, KIntList indexs, KaiMath* pMath) {
	KaiShape fshape = field_info["feed_shape"];
	KaiShape dshape = fshape.insert_head(mb_size);

	if (field_info["dtype"] == "float") {
		for (auto& it2 : cbFloatFuncs) {
			KaiArray<KFloat> arr = KaiArray<KFloat>::zeros(dshape);
			KFloat* p_dat = arr.data_ptr();

			KaiList cb_info = it2;

			KCbFeedFloatData* pFunc = (KCbFeedFloatData*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (!pFunc(pInst, pAux, bInput, sFieldName, fshape, indexs, p_dat)) continue;

			dats[sFieldName] = pMath->to_cuda(arr).get_core();

			return true;
		}
	}
	else {
		for (auto& it2 : cbIntFuncs) {
			KaiArray<KInt> arr = KaiArray<KInt>::zeros(dshape);
			KInt* p_dat = arr.data_ptr();

			KaiList cb_info = it2;

			KCbFeedIntData* pFunc = (KCbFeedIntData*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (!pFunc(pInst, pAux, bInput, sFieldName, fshape, indexs, p_dat)) continue;

			dats[sFieldName] = pMath->to_cuda(arr).get_core();

			return true;
		}
	}

	return false;
}

KBool KaiCallbackAgent::m_feed_free_shape_data(KaiDict& dats, KString sFieldName, KaiDict field_info, KaiList cbFloatFuncs, KaiList cbIntFuncs, KInt mb_size, KBool bInput, KIntList indexs, KaiMath* pMath) {
	if (field_info["dtype"] == "float") {
		for (auto& it2 : cbFloatFuncs) {
			KaiShape fshape;
			KFloat* p_dat;

			KaiList cb_info = it2;

			KCbFeedFreeShapeFloatData* pFunc = (KCbFeedFreeShapeFloatData*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (!pFunc(pInst, pAux, bInput, sFieldName, fshape, indexs, p_dat)) continue;

			KaiArray<KFloat> arr(fshape, p_dat);

			dats[sFieldName] = pMath->to_cuda(arr).get_core();

			return true;
		}
	}
	else {
		for (auto& it2 : cbIntFuncs) {
			KaiShape fshape;
			KInt* p_dat;

			KaiList cb_info = it2;

			KCbFeedFreeShapeIntData* pFunc = (KCbFeedFreeShapeIntData*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (!pFunc(pInst, pAux, bInput, sFieldName, fshape, indexs, p_dat)) continue;

			KaiArray<KInt> arr(fshape, p_dat);

			dats[sFieldName] = pMath->to_cuda(arr).get_core();

			return true;
		}
	}

	return false;
}

KBool KaiCallbackAgent::inform_data_indexes(KIntList data_indexes, KInt nRangeStart, KInt nRangeCount) {
	KaiList cbFuncs;

	if (m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::inform_data_indexes, cbFuncs)) {
		for (auto& it : cbFuncs) {
			KaiList cb_info = it;

			KCbInformDataIndexes* pFunc = (KCbInformDataIndexes*)(KInt)cb_info[0];
			void* pInst = (void*)(KInt)cb_info[1];
			void* pAux = (void*)(KInt)cb_info[2];

			if (pFunc(pInst, pAux, data_indexes, nRangeStart, nRangeCount)) return true;
		}
	}

	return false;
}

KBool KaiCallbackAgent::fetchData(KIntList data_indexes, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext, KBool bThrow) {
	KaiList cbFloatFuncs;
	KaiList cbFloatFreeFuncs;
	KaiList cbIntFuncs;
	KaiList cbIntFreeFuncs;

	m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::feed_float_data, cbFloatFuncs);
	m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::feed_float_free_data, cbFloatFreeFuncs);
	m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::feed_int_data, cbIntFuncs);
	m_fetch_functions((int)Ken_cb_family::datafeed, (int)Ken_datafeed_cb_event::feed_int_free_data, cbIntFreeFuncs);

	KaiMath* pMath = pContext->get_math();

	KInt mb_size = data_indexes.size();

	KaiDict input_fields = pContext->get_property("input_fields");
	KaiDict output_fields = pContext->get_property("output_fields");

	KString sFieldName;
	KaiDict field_info;

	for (auto& it : input_fields) {
		sFieldName = it.first;
		field_info = it.second;

		if (!(KBool)field_info["free_shape"]) {
			if (!m_feed_data(xs, sFieldName, field_info, cbFloatFuncs, cbIntFuncs, mb_size, true, data_indexes, pMath)) {
				if (bThrow) throw KaiException(KERR_FAILURE_ON_DATA_FEEDING_INPUT_FIELD, sFieldName);
			}
		}
		else {
			if (!m_feed_free_shape_data(xs, sFieldName, field_info, cbFloatFreeFuncs, cbIntFreeFuncs, mb_size, true, data_indexes, pMath)) {
				if (bThrow) throw KaiException(KERR_FAILURE_ON_DATA_FEEDING_INPUT_FIELD, sFieldName);
			}
		}
	}

	for (auto& it : output_fields) {
		sFieldName = it.first;
		field_info = it.second;

		if (!(KBool)field_info["free_shape"]) {
			if (!m_feed_data(ys, sFieldName, field_info, cbFloatFuncs, cbIntFuncs, mb_size, false, data_indexes, pMath)) {
				if (bThrow) throw KaiException(KERR_FAILURE_ON_DATA_FEEDING_INPUT_FIELD, sFieldName);
			}
		}
		else {
			if (!m_feed_free_shape_data(ys, sFieldName, field_info, cbFloatFreeFuncs, cbIntFreeFuncs, false, mb_size, data_indexes, pMath)) {
				if (bThrow) throw KaiException(KERR_FAILURE_ON_DATA_FEEDING_INPUT_FIELD, sFieldName);
			}
		}
	}

	return true;
}

KBool KaiCallbackAgent::m_fetch_functions(int family, int event, KaiList& cbFuncs) {
	if (m_cbFamilies.find(family) == m_cbFamilies.end()) return false;
	std::map<int, KaiList> eventDict = m_cbFamilies[family];
	if (eventDict.find(event) == eventDict.end()) return false;
	cbFuncs = eventDict[event];
	return true;
}

int KaiCallbackTransaction::ms_checkCode = 33862128;
int KaiCallbackTransaction::ms_tokenCode = 61707164;

KaiCallbackTransaction::KaiCallbackTransaction() {
	m_nCheckCode = ms_checkCode;
}

KaiCallbackTransaction::~KaiCallbackTransaction() {
	for (auto& pToken : m_tokens) delete pToken;
	m_nCheckCode = 0;
}

KaiDict KaiCallbackTransaction::create_token(KaiValue value) {
	if (value.type() != Ken_value_type::object) throw KaiException(KERR_ARR_TOKEN_REQUEST_FOR_NOT_ARRAY);

	KaiObject* pObject = (KHObject)value;

	_ArrToken* pToken = new _ArrToken;

	pToken->m_pTransaction = this;
	pToken->m_index = (int)m_tokens.size();
	pToken->m_nCheckCode = ms_tokenCode;

	m_tokens.push_back(pToken);
	m_values.push_back(value);

	KaiDict token_info;

	if (pObject->get_type() == Ken_object_type::farray) {
		KaiArray<KFloat> farr = FARRAY(value);
		token_info["is_float"] = true;
		token_info["shape"] = farr.shape();
		token_info["token"] = (KInt)pToken;
	}
	else if (pObject->get_type() == Ken_object_type::narray) {
		KaiArray<KInt> narr = NARRAY(value);
		token_info["is_float"] = false;
		token_info["shape"] = narr.shape();
		token_info["token"] = (KInt)pToken;
	}
	else {
		throw KaiException(KERR_ARR_TOKEN_REQUEST_FOR_NOT_ARRAY);
	}

	return token_info;
}

KaiDict KaiCallbackTransaction::conv_to_token_dict(KaiDict dict) {
	KaiDict token_dict;

	for (auto& it : dict) {
		if (it.second.type() != Ken_value_type::object) continue;
		Ken_object_type obj_type = ((KaiObject*)it.second)->get_type();
		if (obj_type == Ken_object_type::farray || obj_type == Ken_object_type::narray) {
			token_dict[it.first] = create_token(it.second);
		}
	}

	return token_dict;
}

void KaiCallbackTransaction::download_float_data(_ArrToken* pToken, KInt nSize, KFloat* pBuffer) {
	if (m_nCheckCode != ms_checkCode) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);
	if (pToken->m_nCheckCode != ms_tokenCode) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);
	if (pToken->m_index < 0 || pToken->m_index >= m_tokens.size()) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);
	if (m_tokens[pToken->m_index] != pToken) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);

	KaiArray<KFloat> arr = FARRAY(m_values[pToken->m_index]);

	if (arr.total_size() != nSize) throw KaiException(KERR_BAD_BUFFER_SIZE_FOR_DOWNLOAD_DATA);

	KaiArray<KFloat> host_arr = arr.to_host();

	KFloat* pSrc = host_arr.data_ptr();

	memcpy(pBuffer, pSrc, nSize * sizeof(KFloat));
}

void KaiCallbackTransaction::download_int_data(_ArrToken* pToken, KInt nSize, KInt* pBuffer) {
	if (m_nCheckCode != ms_checkCode) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);
	if (pToken->m_nCheckCode != ms_tokenCode) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);
	if (pToken->m_index < 0 || pToken->m_index >= m_tokens.size()) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);
	if (m_tokens[pToken->m_index] != pToken) throw KaiException(KERR_INVALID_TOKEN_INFO_FOR_DOWNLOAD_DATA);

	KaiArray<KInt> arr = NARRAY(m_values[pToken->m_index]);

	if (arr.total_size() != nSize) throw KaiException(KERR_BAD_BUFFER_SIZE_FOR_DOWNLOAD_DATA);

	KaiArray<KInt> host_arr = arr.to_host();

	KInt* pSrc = host_arr.data_ptr();

	memcpy(pBuffer, pSrc, nSize * sizeof(KInt));
}

