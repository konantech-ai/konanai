/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kmodel_instance.h"

#include "../components/kmodel.h"
#include "../components/kdataset.h"
#include "../components/knetwork.h"
#include "../components/koptimizer.h"
#include "../components/kexpression.h"
#include "../components/klayer.h"
#include "../math/kmath.h"
#include "../utils/kutil.h"

int KaiModelInstance::ms_checkCode = 17352731;

KStrList KaiModelInstance::ms_builtin = {};
//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif
KaiModelInstance::KaiModelInstance(KaiSession* pSession, KaiModel* pModel, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::model_instance, Ken_object_type::model_instance, kwArgs) {
	m_checkCode = ms_checkCode;

	m_pCbAgent = NULL;

	KaiDataset* pDataset = KaiDataset::HandleToPointer(m_pSession->get_bound_component(pModel, "dataset"));
	KaiNetwork* pNetwork = KaiNetwork::HandleToPointer(m_pSession->get_bound_component(pModel, "network"));
	KaiOptimizer* pOptimizer = KaiOptimizer::HandleToPointer(m_pSession->get_bound_component(pModel, "optimizer"));
	KaiExpression* pLossFunc = KaiExpression::HandleToPointer(m_pSession->get_bound_component(pModel, "loss_exp"));
	KaiExpression* pAccuracyFunc = KaiExpression::HandleToPointer(m_pSession->get_bound_component(pModel, "accuracy_exp"));
	KaiExpression* pVisualFunc = KaiExpression::HandleToPointer(m_pSession->get_bound_component(pModel, "visualize_exp"), true);
	KaiExpression* pPredictFunc = KaiExpression::HandleToPointer(m_pSession->get_bound_component(pModel, "predict_exp"), true);

	if (pDataset == NULL) throw KaiException(KERR_NO_DATASET_FOR_MODEL_EXEC);
	if (pNetwork == NULL) throw KaiException(KERR_NO_NETWOK_FOR_MODEL_EXEC);
	if (pOptimizer == NULL) throw KaiException(KERR_NO_OPTIMIZER_FOR_MODEL_EXEC);
	if (pLossFunc == NULL) throw KaiException(KERR_NO_LOSS_FUNC_FOR_MODEL_EXEC);
	if (pAccuracyFunc == NULL) throw KaiException(KERR_NO_ACCURACY_FUNC_FOR_MODEL_EXEC);

	pModel->refreshDirty();
	pDataset->refreshDirty();
	pNetwork->refreshDirty();
	pOptimizer->refreshDirty();
	pLossFunc->refreshDirty();
	pAccuracyFunc->refreshDirty();
	if(pVisualFunc != nullptr) pVisualFunc->refreshDirty(); //hs.cho
 	if(pPredictFunc != nullptr) pPredictFunc->refreshDirty(); //hs.cho

	pModel->collect_properties(this, "model");
	pDataset->collect_properties(this, "dataset");
	pNetwork->collect_properties(this, "network");
	pOptimizer->collect_properties(this, "optimizer");
	pLossFunc->collect_properties(this, "loss_exp");
	pAccuracyFunc->collect_properties(this, "accuracy_exp");
	if (pVisualFunc != nullptr)  pVisualFunc->collect_properties(this, "visualize_exp"); //hs.cho
	if (pPredictFunc != nullptr) pPredictFunc->collect_properties(this, "predict_exp"); //hs.cho

	m_pCbAgent = new KaiCallbackAgent(this);

	if ((KString)pDataset->get_property("builtin") != "feeding") m_setDataInfo();
	else pDataset->setDataFeedingInfo(this, m_pCbAgent);

	m_createParameters(pNetwork, pDataset, pOptimizer);

	/*
	* // expression 쪽에서 아래와 같이 수식그래프 만든 상태, 
	m_propDict["dict_exprs"] = dict_exprs;
	m_propDict["dict_terms"] = dict_terms;

	if (m_propDict.find("postproc") != m_propDict.end()) {
		m_propDict["postproc_exp"] = new KaiExpDefRoot((KString)m_propDict["postproc"]);
	}
	*/
}

KaiModelInstance::~KaiModelInstance() {
	delete m_pCbAgent;
	m_checkCode = 0;
}

KaiModelInstance* KaiModelInstance::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "ModelInstance");

	KaiModelInstance* pModelInstance = (KaiModelInstance*)hObject;

	if (pModelInstance->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "ModelInstance");
	if (pModelInstance->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "ModelInstance");

	return pModelInstance;
}

KaiModelInstance* KaiModelInstance::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "ModelInstance");

	KaiModelInstance* pModelInstance = (KaiModelInstance*)hObject;

	if (pModelInstance->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "ModelInstance");

	return pModelInstance;
}

KaiModelInstance* KaiModelInstance::incRefCount() {
	m_nRefCount++;
	return this;
}

void KaiModelInstance::destroy() {
	if (this && --m_nRefCount <= 0) delete this;
}

void KaiModelInstance::train(KaiDict kwArgs, KBool bAsync) {
	KaiExecContext* pContext = new KaiExecContext(m_pSession, this, exec_mode::em_train, kwArgs, m_pCbAgent);

	if (bAsync) {
		std::thread* pTrainThread = new std::thread(ms_trainThreadMain, pContext);
		//pTrainThread->join();
		//delete pTrainThread;
		// 스레드 종료 처리 방법 찾아랏
		// 단일 exec_context의 작업 수행에서 내부적  싱크 유지 방법 찾아랏
		// 두 개 이상의 xontext 동시 수행 확인해랏
	}
	else {
		pContext->train();
	}
}

void KaiModelInstance::ms_trainThreadMain(void* aux) {
	KaiExecContext* pContext = (KaiExecContext*)aux;
	KaiSession* pSession = pContext->get_session();

	try {
		try {
			pContext->train();
		}
		catch (KValueException ex) { throw KaiException(ex.m_nErrCode); }
	}
	catch (KaiException ex) {
		pSession->SetLastError(ex);
		//return ex.GetErrorCode();
		throw KaiException(KERR_UNIMPEMENTED_YET, "메인 스레드에 예외 발생 전달할 방법 찾아랏");
	}
	catch (...) {
		//return KERR_UNKNOWN_ERROR;
		throw KaiException(KERR_UNIMPEMENTED_YET, "메인 스레드에 예외 발생 전달할 방법 찾아랏");
	}
}

void KaiModelInstance::test(KaiDict kwArgs, KBool bAsync) {
	KaiExecContext* pContext = new KaiExecContext(m_pSession, this, exec_mode::em_test, kwArgs, m_pCbAgent);

	if (bAsync) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
		//std::thread* pTESTThread = new std::thread(ms_testThreadMain, pContext);
		//pTrainThread->join();
		//delete pTrainThread;
		// 스레드 종료 처리 방법 찾아랏
		// 단일 exec_context의 작업 수행에서 내부적  싱크 유지 방법 찾아랏
		// 두 개 이상의 xontext 동시 수행 확인해랏
	}
	else {
		pContext->test();
	}
}

void KaiModelInstance::visualize(KaiDict kwArgs, KBool bAsync) {
	KaiExecContext* pContext = new KaiExecContext(m_pSession, this, exec_mode::em_visualize, kwArgs, m_pCbAgent);

	if (bAsync) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
		//std::thread* pTESTThread = new std::thread(ms_testThreadMain, pContext);
		//pTrainThread->join();
		//delete pTrainThread;
		// 스레드 종료 처리 방법 찾아랏
		// 단일 exec_context의 작업 수행에서 내부적  싱크 유지 방법 찾아랏
		// 두 개 이상의 xontext 동시 수행 확인해랏
	}
	else {
		pContext->visualize();
	}
}

KaiList KaiModelInstance::predict(KaiDict kwArgs) {
	KaiExecContext* pContext = new KaiExecContext(m_pSession, this, exec_mode::em_visualize, kwArgs, m_pCbAgent);
	return pContext->predict();
}

KInt KaiModelInstance::get_trained_epoch_count() {
	KaiList train_history = m_propDict["train_history"];
	return (KInt)train_history.size();
}

KInt KaiModelInstance::get_int_property(KString sKey) {
	if (m_propDict.find(sKey) != m_propDict.end()) return m_propDict[sKey];
	throw KaiException(KERR_MISSING_PROPERTY_ON_MODEL_CONTEXT, sKey);
}

KInt KaiModelInstance::get_int_property(KString sKey, KInt nDefault) {
	if (m_propDict.find(sKey) != m_propDict.end()) return m_propDict[sKey];
	return nDefault;
}

KaiValue KaiModelInstance::get_property(KString sKey, KaiValue vDefault) {
	if (m_propDict.find(sKey) != m_propDict.end()) {
		return m_propDict[sKey];
	}
	else if (vDefault.type() == Ken_value_type::none) {
		throw KaiException(KERR_MISSING_PROPERTY_ON_MODEL_CONTEXT, sKey);
	}
	else {
		return vDefault;
	}
}

KaiValue KaiModelInstance::get_property(KStrList slKeys) {
	KaiDict dict = m_propDict;

	KInt nLast = slKeys.size() - 1;

	for (KInt n = 0; n < nLast; n++) {
		dict = dict[slKeys[n]];
	}

	return dict[slKeys[nLast]];
}

void KaiModelInstance::m_setDataInfo() {
	KaiDict data = get_property(KStrList{ "dataset", "data" });
	KaiArray<KFloat> data_arr = FARRAY(data["#default"]);

	KInt data_count = data_arr.axis_size(0);

	KFloat tr_ratio = get_property(KStrList{ "dataset", "tr_ratio" });
	KFloat te_ratio = get_property(KStrList{ "dataset", "te_ratio" });
	KFloat va_ratio = get_property(KStrList{ "dataset", "va_ratio" });

	KInt tr_count = (KInt)(int)(data_count * tr_ratio);
	KInt te_count = (KInt)(int)(data_count * te_ratio);
	KInt va_count = (KInt)(int)(data_count * va_ratio);

	if (tr_count + te_count > data_count) throw KaiException(KERR_TVT_DATA_CNT_EXCEEDS_TOTAL_DATA_CNT);
	if (tr_count + te_count + va_count < data_count) va_count = data_count - (tr_count + te_count);

	KaiMath* pMath = KaiMath::GetHostMath();

	KaiArray<KInt> total_index = pMath->arange(data_count);

	KString sDataSplit = get_property(KStrList{ "dataset", "data_split" });

	if (sDataSplit != "sequential") {
		pMath->shuffle(total_index);
		if (sDataSplit != "random") {
			logger.Print("Need to support Dataloder::data_split::%s option", sDataSplit.c_str());
		}
	}

	m_propDict["data_count"] = data_count;
	m_propDict["data_index"] = total_index.get_core();

	// TRACE
	//KaiArray<KInt> index_list = total_index;
	//printf("[TRACE] %s(%u): index_list[%lld] : { ", __FUNCTION__, __LINE__, index_list.total_size());
	//for (int i=0; i<10 && i<index_list.total_size(); ++i)
	//	printf("%lld ", index_list.get_at(i));
	//if (index_list.total_size() > 11)
	//	printf("... ");
	//if (index_list.total_size() > 10)
	//	printf("%lld ", index_list.get_at( index_list.total_size() - 1 ));
	//printf("}\n");

	KaiDict tr_data, te_data, va_data;
	KInt tr_start = 0;

	tr_data["data_start"] = tr_start; // (KInt)0;
	tr_data["data_count"] = tr_count;
	tr_data["batch_size"] = get_property("tr_batch_size");

	te_data["data_start"] = tr_count;
	te_data["data_count"] = te_count;
	te_data["batch_size"] = get_property("te_batch_size");

	va_data["data_start"] = tr_count + te_count;
	va_data["data_count"] = va_count;
	va_data["batch_size"] = get_property("va_batch_size");

	m_propDict["tr_data"] = tr_data;
	m_propDict["te_data"] = te_data;
	m_propDict["va_data"] = va_data;
}

void KaiModelInstance::m_createParameters(KaiNetwork* pNetwork, KaiDataset* pDataset, KaiOptimizer* pOptimizer) {
	KaiList train_history;

	KaiDict call_info = m_copy_properties();

	KaiDict input_fields = get_property("input_fields");
	KaiDict output_fields = get_property("output_fields");

	KaiShape hshape;

	KaiDict pack;

	for (auto& it : input_fields) {
		KaiDict field_info = it.second;

		if (it.first == "#default") {
			hshape = field_info["shape"];
			if (hshape.size() == 0) throw KaiException(KERR_DEFAULT_INPUT_SHAPE_NOT_DECLARED);
			if ((KString)field_info["dtype"] != "float") throw KaiException(KERR_DEFAULT_INPUT_DTYPE_NOT_FLOAT);
		}
		else {
			pack[it.first] = field_info["shape"];
		}
	}

	/*
	KaiShape hshape_net = pNetwork->get_property("input_shape", KaiShape{});
	KaiShape hshape = get_property("input_shape", hshape_net);

	if (hshape.size() == 0) throw KaiException(KERR_INPUT_SHAPE_NOT_DECLARED);
	if (hshape_net.size() > 0 && hshape != hshape_net) throw KaiException(KERR_DUP_INPUT_SHAPE_DECL_MISMATCHED);
	*/

	KaiList info_pair = pNetwork->prepare_net_exec_info(hshape, pOptimizer, call_info, pack);

	KaiList layerInfos = info_pair[0];
	KaiList layerParams = info_pair[1];
	KaiDict netInfo = info_pair[2];

	KaiShape yshape;

	if (output_fields.find("#default") != output_fields.end()) {
		KaiDict field_info = output_fields["#default"];
		yshape = field_info["shape"];
		if (yshape.size() == 0) throw KaiException(KERR_OUTPUT_SHAPE_NOT_DECLARED);
		if ((KString)field_info["dtype"] != "float") throw KaiException(KERR_DEFAULT_OUTPUT_DTYPE_NOT_FLOAT);
	}

	/*
	KaiShape yshape_net = pNetwork->get_property("output_shape", KaiShape{});
	KaiShape yshape = get_property("output_shape", yshape_net);

	if (yshape.size() == 0) throw KaiException(KERR_OUTPUT_SHAPE_NOT_DECLARED);
	if (yshape_net.size() > 0 && yshape != yshape_net) throw KaiException(KERR_DUP_OUTPUT_SHAPE_DECL_MISMATCHED);
	*/

	KInt hsize = hshape.total_size();
	KInt ysize = yshape.total_size();

	if ((KBool)get_property("use_output_layer", true)) {
		KaiDict layerInfo;
		KaiDict layerParam;

		layerInfo["builtin"] = "dense";
		layerInfo["actfunc"] = "none";
		layerInfo["output"] = true;
		layerInfo["input_shape"] = KaiShape{ hsize };
		layerInfo["output_shape"] = KaiShape{ ysize };
		layerInfo["width"] = ysize;
		layerInfo["actfunc_id"] = (KInt)Ken_actfunc::none;

		KaiShape wshape = KaiShape{ hsize, ysize };
		KString init_weight = kutil.seek_set_dict(call_info, "init_weight", "gaussian");
		KFloat init_std = kutil.seek_set_dict(call_info, "init_std", 0.030f);

		layerParam["weight"] = pOptimizer->createAffineParams(wshape, true, false, init_weight, init_std);

		layerInfos.push_back(layerInfo);
		layerParams.push_back(layerParam);
	}
	else if (ysize > 0 && hsize != ysize) {
		throw KaiException(KERR_WRONG_NETWORK_OUTPUT_VEC_SIZE);
	}

	m_propDict["output_shape"] = yshape;

	m_propDict["layerInfos"] = layerInfos;
	m_propDict["layerParams"] = layerParams;
	m_propDict["netInfo"] = netInfo;
	m_propDict["train_history"] = train_history;
}

void KaiModelInstance::updateParams(KaiList layerParams, KaiList train_historyToAdd) {
	m_propDict["layerParams"] = layerParams;

	KaiList train_history = m_propDict["train_history"];

	for (auto& it : train_historyToAdd) {
		train_history.push_back(it);
	}
	m_propDict["train_history"] = train_history;
}

KString KaiModelInstance::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf,128, "<KaiComponent ModelInstance %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

