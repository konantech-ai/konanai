/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "exec_context.h"
#include "../session/kcommon.h"
#include "../components/kmodel_instance.h"
#include "../components/kmodel.h"
#include "../components/kdataset.h"
#include "../components/knetwork.h"
#include "../components/koptimizer.h"
#include "../components/kexpression.h"
#include "../components/klayer.h"
#include "../math/kmath.h"
#include "../utils/kutil.h"
#include "../exec/callback.h"
#include "../nightly/nightly_utils.h"

KBool KaiExecContext::ms_debugTrace = false;

KaiExecContext::KaiExecContext(KaiSession* pSession, KaiModelInstance* pModelInst, exec_mode mode, KaiDict kwArgs, KaiCallbackAgent* pCbAgent)
: m_shortTermContextInfo(), m_timeMap(), m_tr_data(), m_te_data(), m_va_data() {
	m_pSession = pSession;
	m_pModelInstance = pModelInst->incRefCount();
	m_mode = mode;

	m_pMath = KaiMath::Allocate(pModelInst);
	m_pCbAgent = pCbAgent;

	m_initProperties(kwArgs);
}

KaiExecContext::~KaiExecContext() {
	m_pModelInstance->destroy();
	delete m_pMath;
}

void KaiExecContext::m_initProperties(KaiDict kwArgs) {
	for (auto& it : kwArgs) {
		m_shortTermContextInfo[it.first] = it.second;
	}

	m_tr_data = get_property("tr_data");
	m_te_data = get_property("te_data");
	m_va_data = get_property("va_data");

	KaiList hostParams = m_pModelInstance->get_property("layerParams");
	//KString sDump = hostParams.desc();
	KaiList cudaParams = conv_to_cuda(hostParams);

	set_property("layerParams", cudaParams);
}

void KaiExecContext::set_property(KString sKey, KaiValue vValue) {
	m_shortTermContextInfo[sKey] = vValue;
}

KInt KaiExecContext::get_int_property(KString sKey) {
	if (m_shortTermContextInfo.find(sKey) != m_shortTermContextInfo.end()) return m_shortTermContextInfo[sKey];
	return m_pModelInstance->get_int_property(sKey);
}

KInt KaiExecContext::get_int_property(KString sKey, KInt nDefault) {
	if (m_shortTermContextInfo.find(sKey) != m_shortTermContextInfo.end()) return m_shortTermContextInfo[sKey];
	return m_pModelInstance->get_int_property(sKey, nDefault);
}

KaiValue KaiExecContext::get_property(KString sKey, KaiValue vDefault) {
	if (m_shortTermContextInfo.find(sKey) != m_shortTermContextInfo.end()) return m_shortTermContextInfo[sKey];
	return m_pModelInstance->get_property(sKey, vDefault);
}

KaiDict KaiExecContext::get_component_property(KString sKey) {
	return m_pModelInstance->get_property(sKey, KaiDict());
}

void KaiExecContext::train() {
	KInt epoch_count = get_int_property("epoch_count", 10);
	KInt start_epoch = get_int_property("start_epoch", 0);
	KInt start_batch = get_int_property("start_batch", 0);

	m_train_job_start();

	KInt batch_size = m_tr_data["batch_size"];
	KInt batch_count = m_tr_data["batch_count"];

	for (KInt nEpoch = start_epoch; nEpoch < epoch_count; nEpoch++) {
		m_train_epoch_start(epoch_count, nEpoch);

		m_tr_data["curr_epoch"] = nEpoch;

		// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_TRAIN)
		{
			KaiList layerParams = get_property("layerParams");
			print_klist(layerParams, "[TRACE]", 2, "layerParams");

			printf("[TRACE]  **** Details of \"layerParams\" ****\n");
			for (KInt idxLayer=0; idxLayer<layerParams.size(); ++idxLayer) {
				KaiDict pm_weight = ((KaiDict)layerParams[idxLayer])["weight"];
				printf("[TRACE]  layerParams[%lld][\"weight\"][\"%s\"] = %lld\n", idxLayer, "train", (KInt)pm_weight["train"]);

				KaiList pm_keys = { KString("b"), KString("w") };
				for (KInt idxKey=0; idxKey<pm_keys.size(); ++idxKey) {
					KaiDictIter it = pm_weight.find((KString)pm_keys[idxKey]);
					
					if (it == pm_weight.end())
						continue;

					KaiDict pm = it->second;

					printf("[TRACE]  layerParams[%lld][\"weight\"][\"%s\"] = %s\n",
						idxLayer, ((KString)pm_keys[idxKey]).c_str(), "(KaiDict)");

					KaiArray<KFloat> _pm_ = FARRAY(pm["_pm_"]);
					KFloat           n    = pm["n"];			// step
					KaiArray<KFloat> s    = FARRAY(pm["s"]);	// 1st momentum
					KaiArray<KFloat> t    = FARRAY(pm["t"]);	// 2nd momentum
					
					size_t indent_size = KString("[TRACE]  ").length() + 4;

					print_karray_kfloat(&_pm_, "", indent_size, "[\"_pm_\"]", 10);
					printf("%s[\"n\"] = %.6lf\n", KString(indent_size, ' ').c_str(), n);
					print_karray_kfloat(&s, "", indent_size, "[\"s\"]", 10);
					print_karray_kfloat(&t, "", indent_size, "[\"t\"]", 10);
				}
			}
			printf("[TRACE]  **** End of details ****\n");
		}
#endif

		for (KInt nBatch = start_batch; nBatch < batch_count; nBatch++) {
			m_tr_data["curr_batch"] = nBatch;

			KaiDict xs, ys;
			KInt mb_size;

			m_fetch_data(m_tr_data, xs, ys, mb_size);

			m_train_minibatch(batch_count, nBatch, xs, ys, mb_size);

			KBool bValidate = m_invoke_check("in_batch_validate", nBatch);
			KBool bReport = m_invoke_check("in_batch_report", nBatch);
			
			m_exec_report(bValidate, bReport, nEpoch, nBatch);

			if (m_invoke_check("in_batch_visualize", nBatch)) m_exec_visualize(Ken_visualize_mode::train, nEpoch, nBatch);
			if (m_invoke_check("in_batch_save", nBatch)) m_exec_save(nEpoch, nBatch);
		}

		KBool bValidate = m_invoke_check("epoch_validate", nEpoch);
		KBool bReport = m_invoke_check("epoch_report", nEpoch);

		// Print out the results
		m_exec_report(bValidate, bReport, nEpoch);
		
		if (m_invoke_check("epoch_visualize", nEpoch)) m_exec_visualize(Ken_visualize_mode::train, nEpoch);
		if (m_invoke_check("epoch_save", nEpoch)) m_exec_save(nEpoch);

		start_batch = 0;

		m_train_epoch_finish(epoch_count, nEpoch);
	}

	if (m_not_invoke_check("epoch_report", epoch_count-1) && m_not_invoke_check("epoch_validate", epoch_count - 1)) m_exec_report(false, true, epoch_count-1);

	m_train_job_finish();
}

void KaiExecContext::test() {
	m_test_job_start();

	KInt batch_size = m_te_data["batch_size"];
	KInt batch_count = m_te_data["batch_count"];

	for (KInt nBatch = 0; nBatch < batch_count; nBatch++) {
		m_te_data["curr_batch"] = nBatch;

		KaiDict xs, ys;
		KInt mb_size;

		m_fetch_data(m_te_data, xs, ys, mb_size);
		m_test_minibatch(xs, ys, mb_size);
	}

	m_test_job_finish();
}

void KaiExecContext::visualize() {
	m_exec_visualize(Ken_visualize_mode::visualize, -1, -1);
}

KaiList KaiExecContext::predict() {
	m_predict_job_start();

	KBool bVisualize = get_property("visualize", false);

	KInt mb_size;

	KaiDict xs = KaiDataset::fetch_predict_data(this, &mb_size);
	KaiDict outs = m_predict_minibatch(xs);

	if (bVisualize) {
		KString sName = m_pModelInstance->get_property(KStrList{ "model", "name" });

		if (m_pCbAgent->visualize_start(sName, Ken_visualize_mode::predict, mb_size, KaiDict())) {
			KaiDict vis_info = m_pModelInstance->get_property("visualize_exp", KaiDict());
			KaiDict pred_info = m_pModelInstance->get_property("predict_exp", vis_info);
			KaiDict pred_dict = KaiExpression::evaluate("#predict", pred_info, xs, KaiDict(), outs, this, false, mb_size);

			m_pCbAgent->visualize_end(sName, xs, KaiDict(), outs, pred_dict);
		}
		else {
			logger.Print("predict visualize report:");
			m_visualize_output(xs, KaiDict(), outs);
		}
	}

	m_predict_job_finish();

	return KaiList();
}

void KaiExecContext::m_train_minibatch(KInt batch_count, KInt batch_index, KaiDict xs, KaiDict ys, KInt mb_size) {
	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_TRAIN_MINIBATCH)
	{
		printf("****************************************************************\n");
		printf(" %s() begins.\n", __FUNCTION__);
		printf("****************************************************************\n");
	}
#endif

	m_pCbAgent->train_batch_start(batch_count, batch_index);

	KBool debug_trace = false;

	KaiDict grads;
	
	KaiDict pack;
	for (auto& it : xs) pack[it.first] = it.second;

	m_forward_neuralnet(xs, true, pack);

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_TRAIN_MINIBATCH)
	printf("[TRACE]  m_forward_neuralnet() has been completed.\n");
	KaiArray<KFloat> pack_default = FARRAY(pack["#default"]);
	print_karray_kfloat(&pack_default, "[TRACE]", 2, "pack[\"#default\"]", 10);
	KaiList aux = pack["aux"];
	for (KInt i=0; i<aux.size(); ++i) {
		KaiDict subdict = aux[i];
		std::printf("[TRACE]  pack[\"aux\"][%lld][\"xshape\"].desc() : %s\n", i, subdict["xshape"].desc().c_str());

		char szArrayName[256] = {0, };

		sprintf(szArrayName, "pack[\"aux\"][%lld][\"%s\"]", i, "post_act");
		KaiArray<KFloat> post_act = FARRAY(subdict["post_act"]);
		print_karray_kfloat(&post_act, "[TRACE]", 2, szArrayName, 10);

		sprintf(szArrayName, "pack[\"aux\"][%lld][\"%s\"]", i, "pre_act");
		KaiArray<KFloat> pre_act = FARRAY(subdict["pre_act"]);
		print_karray_kfloat(&pre_act, "[TRACE]", 2, szArrayName, 10);

		sprintf(szArrayName, "pack[\"aux\"][%lld][\"%s\"]", i, "temp_x");
		KaiArray<KFloat> temp_x = FARRAY(subdict["temp_x"]);
		print_karray_kfloat(&temp_x, "[TRACE]", 2, szArrayName, 10);

		sprintf(szArrayName, "pack[\"aux\"][%lld][\"%s\"]", i, "x");
		KaiArray<KFloat> x = FARRAY(subdict["x"]);
		print_karray_kfloat(&x, "[TRACE]", 2, szArrayName, 10);
	}
#endif

	// Not supported yet (commented out by Hyung-jae, Son)
	//KaiArray<KFloat> map1 = FARRAY(pack["feature_map_1"]);
	//KaiArray<KFloat> map2 = FARRAY(pack["feature_map_2"]);
	//KaiArray<KFloat> map3 = FARRAY(pack["feature_map_3"]);

	KaiDict loss = m_eval_loss_grad(xs, ys, pack, grads, mb_size);
	KaiDict accs = m_eval_accuracy(xs, ys, pack, mb_size);
	
	m_backprop_neuralnet(pack, grads);

	m_accumulate_loss(m_tr_loss, loss, mb_size);
	m_accumulate_accs(m_tr_accs, accs, mb_size);

	m_update_parameter();

	m_pCbAgent->train_batch_end(batch_count, batch_index, mb_size, loss, accs);
	//m_pCbAgent->train_batch_end(batch_count, batch_index, mb_size, loss, accs);

#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_TRAIN_MINIBATCH)
	{
		printf("****************************************************************\n");
		printf(" %s() has been completed.\n", __FUNCTION__);
		printf("****************************************************************\n");
	}
#endif
}

void KaiExecContext::m_train_job_start() {
	KString sName = m_pModelInstance->get_property(KStrList{ "model", "name" });
	KString sTimestamp = kutil.get_timestamp(time(NULL));

	KInt epoch_count = get_int_property("epoch_count", 10);
	KInt data_count = m_tr_data["data_count"];
	KInt batch_size = m_tr_data["batch_size"];
	KInt batch_count = (data_count - 1) / batch_size + 1;

	if (!m_pCbAgent->train_start(sName, sTimestamp, epoch_count, data_count, batch_size, batch_count)) {
		logger.Print("Model %s %s started: %s", sName.c_str(), m_job_name().c_str(), sTimestamp.c_str());
	}

	m_tr_data["batch_count"] = batch_count;
	m_tr_data["exec_count"] = data_count;

	m_timeMap["train_start"] = high_resolution_clock::now();
}

void KaiExecContext::m_train_job_finish() {
	KString sName = m_pModelInstance->get_property(KStrList{ "model", "name" });
	KString sTimestamp = kutil.get_timestamp(time(NULL));

	if (!m_pCbAgent->train_end(sName, sTimestamp)) {
		KFloat mtime = m_duration_milisec("train_start");
		logger.Print("Model %s %s ended:   %s (%5.3f secs)", sName.c_str(), m_job_name().c_str(), sTimestamp.c_str(), mtime);
	}

	KaiList layerParams = get_property("layerParams");

	m_pModelInstance->updateParams(layerParams, m_train_history);
}

void KaiExecContext::m_train_epoch_start(KInt epoch_count, KInt epoch_index) {
	m_timeMap["epoch_start"] = high_resolution_clock::now();

	m_shuffle_data(m_tr_data);

	m_pCbAgent->train_epoch_start(epoch_count, epoch_index, m_tr_data);
}

void KaiExecContext::m_train_epoch_finish(KInt epoch_count, KInt epoch_index) {
	// 에포크 처리 직후 필요한 일 수행, 
	// 추후 콜백 호출도 이 함수 앞 뒤에 추가

	KaiDict train_record;
	KaiDict train_loss;
	KaiDict train_accs;

	KInt nDataCount = m_tr_data["data_count"];

	for (auto& it : m_tr_loss) {
		train_loss[it.first] = (KFloat)it.second / (KFloat)nDataCount;
	}

	for (auto& it : m_tr_accs) {
		train_accs[it.first] = (KFloat)it.second / (KFloat)nDataCount;
	}

	train_record["data_index"] = m_tr_data["data_index"];
	train_record["loss"] = train_loss;
	train_record["acc"] = train_accs;

	m_train_history.push_back(train_record);

	m_tr_loss.clear();
	m_tr_accs.clear();

	m_pCbAgent->train_epoch_end(epoch_count, epoch_index, nDataCount, train_loss, train_accs);
}

void KaiExecContext::m_test_minibatch(KaiDict xs, KaiDict ys, KInt mb_size) {
	KaiDict grads;

	KaiDict pack;
	for (auto& it : xs) pack[it.first] = it.second;

	m_forward_neuralnet(xs, false, pack);
	KaiDict accuracy = m_eval_accuracy(xs, ys, pack, mb_size);

	m_accumulate_accs(m_te_accs, accuracy, mb_size);
}

void KaiExecContext::m_test_job_start() {
	// 실행 시작 직전에 필요한 일 수행, 
	// 추후 콜백 호출도 이 함수 앞 뒤에 추가
	KString sName = m_pModelInstance->get_property(KStrList{ "model", "name" });
	KString sTimestamp = kutil.get_timestamp(time(NULL));

	m_timeMap["test_start"] = high_resolution_clock::now();

	KInt data_count = m_te_data["data_count"];
	KInt batch_size = m_te_data["batch_size"];
	KInt batch_count = (data_count - 1) / batch_size + 1;

	m_te_data["batch_count"] = batch_count;
	m_te_data["exec_count"] = data_count;

	m_shuffle_data(m_te_data);

	if (!m_pCbAgent->test_start(sName, sTimestamp, data_count, m_te_data)) {
		logger.Print("Model %s %s started: %s", sName.c_str(), m_job_name().c_str(), sTimestamp.c_str());
	}
}

void KaiExecContext::m_test_job_finish() {
	// 학습 실행 직후 필요한 일 수행, 
	// 추후 콜백 호출도 이 함수 앞 뒤에 추가
	KString sName = m_pModelInstance->get_property(KStrList{ "model", "name" });
	KString sTimestamp = kutil.get_timestamp(time(NULL));

	KInt nDataCount = m_te_data["exec_count"];
	KFloat mtime = m_duration_milisec("test_start");
	
	KaiDict te_acc_means;

	for (auto& it : m_te_accs) {
		te_acc_means[it.first] = (KFloat)it.second / (KFloat)nDataCount;
	}

	m_pCbAgent->test_end(sName, sTimestamp, te_acc_means);

	m_te_accs.clear();
}

void KaiExecContext::m_predict_job_start() {
	// 실행 시작 직전에 필요한 일 수행, 
	// 추후 콜백 호출도 이 함수 앞 뒤에 추가
}

void KaiExecContext::m_predict_job_finish() {
	// 학습 실행 직후 필요한 일 수행, 
	// 추후 콜백 호출도 이 함수 앞 뒤에 추가
}

void KaiExecContext::m_fetch_data(KaiDict dataSection, KaiDict& xs, KaiDict& ys, KInt& mb_size) {
	KInt exec_count = dataSection["exec_count"];
	KInt batch_size = dataSection["batch_size"];
	KInt curr_batch = dataSection["curr_batch"];

	if ((KInt)dataSection["data_count"] < exec_count) exec_count = dataSection["data_count"];

	KInt batch_start = curr_batch * batch_size;

	mb_size = (batch_start + batch_size > exec_count) ? exec_count - batch_start : batch_size;

	KaiMath* pMath = KaiMath::GetHostMath();

	KaiArray<KInt> data_index = NARRAY(dataSection["data_index"]);
	KaiArray<KInt> batch_index = pMath->subrange(data_index, batch_start, mb_size);

	KIntList data_index_list;
	for (KInt m = 0; m < mb_size; m++) data_index_list.push_back(batch_index.get_at(m));

	m_pCbAgent->inform_data_indexes(data_index_list, (KInt)dataSection["data_start"], (KInt)dataSection["data_count"]);

	if ((KString)m_pModelInstance->get_property(KStrList{ "dataset", "builtin" }) == "feeding") {
		m_pCbAgent->fetchData(data_index_list, xs, ys, this, true);
	}
	else {
		KaiDataset::fetchData(batch_index, xs, ys, this);
	}
}

void KaiExecContext::m_forward_neuralnet(KaiDict xs, KBool bIsTraining, KaiDict& pack) {
	set_property("xs", xs);

	KaiList layerInfos = get_property("layerInfos");
	KaiList layerParams = get_property("layerParams");
	KaiList layerAuxs;

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_FORWARD_NEURALNET)
	{
		printf("[TRACE]  %s(%u) {\n", __FUNCTION__, __LINE__);
		print_klist(layerInfos, "", 4, "layerInfos");
		print_klist(layerParams, "", 4, "layerParams");
		printf("}\n\n");
	}
#endif

	if (layerInfos.size() != layerParams.size()) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

	KaiArray<KFloat> harr;
	if (xs.find("#default") != xs.end()) harr = FARRAY(xs["#default"]);

	for (KInt n = 0; n < (KInt) layerInfos.size(); n++) {
		KaiDict layerAux;
		KaiDict* pLayerAux = bIsTraining ? &layerAux : NULL;

		harr = KaiLayer::forward(harr, layerInfos[n], layerParams[n], this, pLayerAux, pack);

		KString sShape = harr.shape().desc();
		if (bIsTraining) layerAuxs.push_back(layerAux);
		//hidden = pLayer->forward(hidden);
	}

	pack["#default"] = harr.get_core();
	pack["aux"] = layerAuxs;
}

KaiDict KaiExecContext::m_eval_loss_grad(KaiDict xs, KaiDict ys, KaiDict outs, KaiDict& grads, KInt mb_size) {
	KaiDict loss_info = m_pModelInstance->get_property("loss_exp");
	KaiDict loss_arr = KaiExpression::evaluate_with_grad(loss_info, xs, ys, outs, this, grads, mb_size);
	return m_fetch_loss_values(loss_arr);
}

KaiDict KaiExecContext::m_eval_accuracy(KaiDict xs, KaiDict ys, KaiDict outs, KInt mb_size) {
	KaiDict acc_info = m_pModelInstance->get_property("accuracy_exp");
	KaiDict acc_dict = KaiExpression::evaluate("#accuracy", acc_info, xs, ys, outs, this, true, mb_size);
	return acc_dict;
}

void KaiExecContext::m_backprop_neuralnet(KaiDict outs, KaiDict grads) {
	KaiList layerInfos = get_property("layerInfos");
	KaiList layerParams = get_property("layerParams");
	KaiList layerAuxs = outs["aux"];

	KaiArray<KFloat> garr;
	KaiDict pack;
	
	for (auto& it : grads) {
		if (it.first == "#default") {
			garr = FARRAY(it.second);
		}
		else pack[it.first] = it.second;
	}

	for (KInt n = (KInt)layerInfos.size()-1; n >= 0; n--) {
		KaiDict layerInfo = layerInfos[n];
		KaiDict layerParam = layerParams[n];
		KaiDict layerAux = layerAuxs[n];

		garr = KaiLayer::backprop(garr, layerInfo, layerParam, layerAux, this, pack);
	}
}

void KaiExecContext::m_update_parameter() {
	KaiList layerParams = get_property("layerParams");
	KaiOptimizer::update_parameter(layerParams, this);
}

KaiValue KaiExecContext::conv_to_cuda(KaiValue value) {
	return m_conv_to_cuda(value, 0);
}

KaiValue KaiExecContext::m_conv_to_cuda(KaiValue value, int depth) {
	Ken_value_type val_type = value.type();

	switch (val_type) {
	case Ken_value_type::dict:
	{
		KaiDict dict = value;
		KaiDict clone;
		//logger.Print("%*s[", depth*2, "");
		for (auto& it : dict) {
			//logger.Print("%*s%s:", depth * 2, "", it.first.c_str());
			clone[it.first] = m_conv_to_cuda(it.second, depth+1);
		}
		//logger.Print("%*s]", depth * 2, "");
		return clone;
	}
		break;
	case Ken_value_type::list:
	{
		KaiList list = value;
		KaiList clone;
		//logger.Print("%*s<", depth * 2, "");
		for (auto& it : list) {
			clone.push_back(m_conv_to_cuda(it, depth+1));
		}
		//logger.Print("%*s>", depth * 2, "");
		return clone;
	}
	break;
	case Ken_value_type::object:
	{
		KaiObject* pObject = value;
		Ken_object_type obj_type = pObject->get_type();
		if (obj_type == Ken_object_type::narray) {
			KaiArray<KInt> host_arr = NARRAY(value);
			KaiArray<KInt> cuda_arr = m_pMath->to_cuda(host_arr);
			return cuda_arr.get_core();
		}
		else if (obj_type == Ken_object_type::farray) {
			KaiArray<KFloat> host_arr = FARRAY(value);
			//logger.Print("%*s  %s", depth * 2, "", host_arr.shape().desc().c_str());
			KaiArray<KFloat> cuda_arr = m_pMath->to_cuda(host_arr);
			return cuda_arr.get_core();
		}
		else {
			//logger.PrintWait("%s", value.desc().c_str());
			return value;
		}
	}
	break;
	default:
		return value;
	}
}

KString KaiExecContext::m_job_name() {
	switch (m_mode) {
	case exec_mode::em_train: return "train";
	case exec_mode::em_validate: return "validate";
	case exec_mode::em_test: return "test";
	case exec_mode::em_visualize: return "visualize";
	}

	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "KaiExecContext::m_job_name()");
}

bool KaiExecContext::m_invoke_check(KString sKey, KInt curr)
{
	KInt period = get_int_property(sKey, 0);
	return period > 0 && (curr + 1) % period == 0;
}

bool KaiExecContext::m_not_invoke_check(KString sKey, KInt curr)
{
	KInt period = get_int_property(sKey, 0);
	return period > 0 && (curr + 1) % period != 0;
}

void KaiExecContext::m_exec_save(KInt nEpoch, KInt nBatch) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiExecContext::m_exec_report(KBool bValidate, KBool bReport, KInt nEpoch, KInt nBatch) {
	if (!bValidate && !bReport) return;

	KInt nDataCount = m_tr_data["data_count"];

	KaiDict loss_means;
	KaiDict tr_acc_means;
	KaiDict va_acc_means;

	//KFloat loss_mean = m_pMath->sum(m_tr_loss) / (KFloat)nDataCount;
	
	for (auto& it : m_tr_loss) {
		loss_means[it.first] = (KFloat)it.second / (KFloat)nDataCount;
	}

	for (auto& it : m_tr_accs) {
		tr_acc_means[it.first] = (KFloat)it.second / (KFloat)nDataCount;
	}

	// TRACE
	//KFloat loss_mean = m_pMath->sum(m_tr_losses) / (KFloat)nDataCount;
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_EXEC_REPORT)
	printf("[TRACE] %s(%u) {\n", __FUNCTION__, __LINE__);
	print_kdict(m_tr_data, "", 4, "m_tr_data");
	print_kdict(loss_means, "", 4, "loss_means");
	print_kdict(tr_acc_means, "", 4, "tr_acc_means");
	printf("}\n\n");
	
#endif

	KaiDict acc_info = m_pModelInstance->get_property("accuracy_exp");

	if (acc_info.find("postproc") != acc_info.end()) {
		KaiDict arr_dict;
		for (auto& it : tr_acc_means) {
			KaiArray<KFloat> arr = m_pMath->ones(KaiShape{ 1 }, (KFloat)it.second);
			arr_dict[it.first] = arr.get_core();
		}
		tr_acc_means = KaiExpression::postproc(acc_info["postproc"], this, arr_dict);
	}

	KBool bValidateReported = false;
	//KBool train_validate_start(KInt batch_count, KInt batch_index);
	//KBool train_validate_end(KInt batch_count, KInt batch_index, KInt batch_size, KFloat loss, KaiDict accuracy);

	if (bValidate) {
		m_validate_start(nEpoch, nBatch);

		KInt batch_count = m_va_data["batch_count"];

		for (KInt nBatch = 0; nBatch < batch_count; nBatch++) {
			m_va_data["curr_batch"] = nBatch;

			KaiDict xs, ys;
			KInt mb_size;

			m_fetch_data(m_va_data, xs, ys, mb_size);
			m_validate_minibatch(xs, ys, mb_size);
		}

		KInt nDataCount = m_va_data["exec_count"];

		for (auto& it : m_va_accs) {
			va_acc_means[it.first] = (KFloat)it.second / (KFloat)nDataCount;
		}

		KaiDict acc_info = m_pModelInstance->get_property("accuracy_exp");
		if (acc_info.find("postproc") != acc_info.end()) {
			KaiDict arr_dict;
			for (auto& it : va_acc_means) {
				KaiArray<KFloat> arr = m_pMath->ones(KaiShape{ 1 }, (KFloat)it.second);
				arr_dict[it.first] = arr.get_core();
			}
			va_acc_means = KaiExpression::postproc(acc_info["postproc"], this, arr_dict);
		}

		bValidateReported = m_pCbAgent->train_validate_end(va_acc_means);
	}

	/*
	if (bReport || !bValidateReported) {
		// Fixed by Hyung-jae, Son (2021-08-26)
		//char buffer[1024];			// before, the result of strlen() is invalid
		char buffer[1024] = { 0, };		// after
		size_t pos = 0;

		if (tr_acc_means.find("#default") != tr_acc_means.end()) {
			sprintf(buffer + pos, "acc = %5.3f", (KFloat)tr_acc_means["#default"]); pos += strlen(buffer + pos);
			if (bValidate) sprintf(buffer + pos, "/%5.3f", (KFloat)va_acc_means["#default"]); pos += strlen(buffer + pos);
		}

		for (auto& it : tr_acc_means) {
			KString sKey = it.first;
			if (sKey == "#default") continue;
			if (pos > 0) sprintf(buffer + pos, ", "); pos += strlen(buffer + pos);
			sprintf(buffer + pos, "%s = %5.3f", sKey.c_str(), (KFloat)tr_acc_means[sKey]); pos += strlen(buffer + pos);
			if (bValidate) sprintf(buffer + pos, "/%5.3f", (KFloat)va_acc_means[sKey]); pos += strlen(buffer + pos);
		}

		if (nBatch < 0) logger.PrintWait("    Epoch(%lld): ", nEpoch + 1);
		else logger.PrintWait("    Batch(%lld/%lld): ", nEpoch, nBatch + 1);

		KFloat mtime1 = m_duration_milisec("epoch_start");
		KFloat mtime2 = m_duration_milisec("train_start");

		logger.Print("[TRAIN] loss = %8.6f, %s (%5.3f/%5.3f secs)", (KFloat)loss_means["#default"], buffer, mtime1, mtime2);
	}
	*/
}

void KaiExecContext::m_validate_minibatch(KaiDict xs, KaiDict ys, KInt mb_size) {
	KaiDict pack;
	for (auto& it : xs) pack[it.first] = it.second;

	m_forward_neuralnet(xs, false, pack);
	KaiDict accuracy = m_eval_accuracy(xs, ys, pack, mb_size);

	m_accumulate_accs(m_va_accs, accuracy, mb_size);
}

void KaiExecContext::m_validate_start(KInt nEpoch, KInt nBatch) {
	m_shuffle_data(m_va_data);

	KInt valid_count = get_int_property("validate_count", 20);

	if ((KInt)m_va_data["data_count"] < valid_count) valid_count = m_va_data["data_count"];

	KInt batch_size = m_va_data["batch_size"];
	KInt batch_count = (valid_count - 1) / batch_size + 1;

	m_va_data["exec_count"] = valid_count;
	m_va_data["batch_count"] = batch_count;

	m_va_accs.clear();

	m_pCbAgent->train_validate_start(valid_count, batch_size);
}

/*
void KaiExecContext::m_validate_finish(KInt nEpoch, KInt nBatch) {
	KInt nDataCount = m_va_data["exec_count"];

	throw KaiException(KERR_UNIMPEMENTED_YET);
	KFloat acc_mean = m_pMath->sum(m_va_accs) / (KFloat)nDataCount;

	m_va_accs.clear();

	if (nBatch < 0) logger.PrintWait("    Epoch(%lld): ", nEpoch + 1);
	else logger.PrintWait("    Batch(%lld/%lld): ", nEpoch, nBatch + 1);

	KFloat mtime = m_duration_milisec("valid_start");

	logger.Print("[VALIDATE] acc = %8.6f (%5.3f secs)", acc_mean, mtime);
}
*/

void KaiExecContext::m_exec_visualize(Ken_visualize_mode mode, KInt nEpoch, KInt nBatch) {
	m_visualize_start(mode, nEpoch, nBatch);

	KaiDict xs, ys;
	KInt mb_size;

	m_fetch_data(m_va_data, xs, ys, mb_size);
	m_visualize_minibatch(xs, ys, mb_size);

	m_visualize_finish(nEpoch, nBatch);
}

void KaiExecContext::m_visualize_minibatch(KaiDict xs, KaiDict ys, KInt mb_size) {
	KaiDict pack;
	for (auto& it : xs) pack[it.first] = it.second;

	m_forward_neuralnet(xs, false, pack);

	KaiDict vis_info = m_pModelInstance->get_property("visualize_exp", KaiDict());
	KaiDict vis_dict = KaiExpression::evaluate("#visualize", vis_info, xs, ys, pack, this, false, mb_size);

	KString sName = m_pModelInstance->get_property(KStrList{ "model", "name" });

	if (!m_pCbAgent->visualize_end(sName, xs, ys, pack, vis_dict)) {
		m_visualize_output(xs, ys, pack);
	}
	
	//throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	if (cbman.familyExist(this, Ken_cb_family::visualize)) {
		_CB_INFO cb;

		if (cbman.get_cb_info(this, Ken_cb_family::visualize, 0, cb)) {
			KCbVisualizeStart* pFunc = (KCbVisualizeStart*)cb.pFunc;
			pFunc(cb.pInst, cb.pAux, nCount);
		}

		if (cbman.get_cb_info(this, Ken_cb_family::visualize, 1, cb)) {
			KCbVisualizeData* pFunc = (KCbVisualizeData*)cb.pFunc;
			for (KInt n = 0; n < nCount; n++) {
				pFunc(cb.pInst, cb.pAux, n);
			}
		}

		if (cbman.get_cb_info(this, Ken_cb_family::visualize, 2, cb)) {
			KCbVisualizeEnd* pFunc = (KCbVisualizeEnd*)cb.pFunc;
			pFunc(cb.pInst, cb.pAux);
		}
	}
	else {
		m_visualize_output(xs, ys, outs);
	}
	*/
}

KaiDict KaiExecContext::m_predict_minibatch(KaiDict xs) {
	KaiDict pack;
	for (auto& it : xs) pack[it.first] = it.second;

	m_forward_neuralnet(xs, false, pack);

	return pack;
}

void KaiExecContext::m_visualize_start(Ken_visualize_mode mode, KInt nEpoch, KInt nBatch) {
	m_shuffle_data(m_va_data);

	KInt visual_count = get_int_property("visualize_count", 5);
	KInt nZero = 0;

	m_va_data["exec_count"] = visual_count;
	m_va_data["batch_size"] = visual_count;
	m_va_data["batch_count"] = 1;
	m_va_data["curr_batch"] = nZero;

	KString sName = m_pModelInstance->get_property(KStrList{ "model", "name" });

	if (!m_pCbAgent->visualize_start(sName, mode, visual_count, m_va_data)) {
		printf("[WARNING] visualization by engine is depreciated. Before release this function will be deleted and visualization will be supported only by the callback.\n");
		if (nEpoch >= 0) {
			if (nBatch < 0) logger.PrintWait("    Epoch(%lld): ", nEpoch + 1);
			else logger.PrintWait("    Batch(%lld/%lld): ", nEpoch, nBatch + 1);
		}

		logger.Print("[Visualize] %lld records\n", visual_count);
	}

	m_timeMap["visual_start"] = high_resolution_clock::now();
}

void KaiExecContext::m_visualize_finish(KInt nEpoch, KInt nBatch) {
}

void KaiExecContext::m_visualize_output(KaiDict xs, KaiDict ys, KaiDict outs) {
	throw KaiException(KERR_DEPRECIATED_FUNCTION_IS_CALLED, "KaiExecContext::m_visualize_output");

	printf("[WARNING] visualization by engine is depreciated. Before release this function will be deleted and visualization will be supported only by the callback.\n");
	KaiArray<KFloat> input = FARRAY(xs["#default"]);
	KaiArray<KFloat> estimate = FARRAY(outs["#default"]);

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_VISUALIZE_OUTPUT)
	printf("[TRACE] %s(%u) {\n", __FUNCTION__, __LINE__);
	printf("    xs.shape()       : %s\n", xs.desc().c_str());
	printf("    ys.shape()       : %s\n", ys.desc().c_str());
	printf("    outs.shape()     : %s\n", outs.desc().c_str());
	printf("    input.shape()    : %s\n", input.shape().desc().c_str());
	printf("    estimate.shape() : %s\n", estimate.shape().desc().c_str());
	printf("}\n\n");
#endif

	KaiDict est_visual_exp = get_property("visualize_exp", KaiDict());

	if (est_visual_exp.size() > 0) {
		KaiDict exp_info = m_pModelInstance->get_property("visualize_exp");
		//KaiDict values = ("#visualize", exp_info, xs, ys, outs, this, false);
		KaiValue value = KaiExpression::evaluate_value(exp_info, xs, ys, outs, this);
		estimate = FARRAY(value);
	}

	input = m_pMath->to_host(input);
	estimate = m_pMath->to_host(estimate);

	KaiList header = get_property("header", KaiList());

	if (header.size() > 0) {
		logger.PrintWait("       ");
		for (auto& it : header) logger.PrintWait(" %s", ((KString)it).c_str());
		logger.Print("");
	}

	KInt batch_size = input.axis_size(0);
	KInt input_size = input.total_size() / batch_size;
	KInt est_size = estimate.axis_size(1);

	if (input.dim() != 2) {
		input = input.reshape(KaiShape{ batch_size , input_size });
	}

	KaiArray<KFloat> answer;
	KInt ans_size;

	KBool bAnswer = ys.size() > 0;

	if (bAnswer) {
		answer = FARRAY(ys["#default"]);
		answer = m_pMath->to_host(answer);
		ans_size = answer.axis_size(1);
	}

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_VISUALIZE_OUTPUT)
	printf("[TRACE] %s(%u) {\n", __FUNCTION__, __LINE__);
	printf("    input.shape()    : %s\n", input.shape().desc().c_str());
	printf("    estimate.shape() : %s\n", estimate.shape().desc().c_str());
	print_kdict(est_visual_exp, "", 4, "est_visual_exp");
	print_klist(header, "", 4, "header");
	printf("    batch_size     : %lld\n", batch_size);
	printf("    input_size     : %lld\n", input_size);
	printf("    est_size       : %lld\n", est_size);
	printf("    bAnswer        : %s\n", bAnswer ? "true" : "false");
	printf("    answer.shape() : %s\n", answer.shape().desc().c_str());
	printf("}\n\n");
#endif

	if (input.dim() != 2 || estimate.dim() != 2 || answer.dim() != 2) {
		printf("[WARNING] visualize of dimension over 2 will be implemented later.\n");
		return;
	}

	for (KInt m = 0; m < batch_size; m++) {
		logger.PrintWait("       ");
		if (input_size <= 10) {
			for (KInt n = 0; n < input_size; n++) {
				logger.PrintWait(" %5.3f", input.get_at(m, n));
			}
		}
		else {
			for (KInt n = 0; n < 3; n++) {
				logger.PrintWait(" %5.3f", input.get_at(m, n));
			}
			logger.PrintWait(" ...");
			for (KInt n = input_size-3; n < input_size; n++) {
				logger.PrintWait(" %5.3f", input.get_at(m, n));
			}
		}
		logger.PrintWait(" => Est");
		KString sPrefix = "[";
		if (est_size <= 10) {
			for (KInt n = 0; n < est_size; n++) {
				logger.PrintWait("%s%5.3f", sPrefix.c_str(), estimate.get_at(m, n));
				sPrefix = ",";
			}
		}
		else {
			for (KInt n = 0; n < 3; n++) {
				logger.PrintWait("%s%5.3f", sPrefix.c_str(), estimate.get_at(m, n));
				sPrefix = ",";
			}
			logger.PrintWait(",...");
			for (KInt n = est_size - 3; n < est_size; n++) {
				logger.PrintWait("%s%5.3f", sPrefix.c_str(), estimate.get_at(m, n));
			}
		}
		logger.PrintWait("]");
		
		if (bAnswer) {
			logger.PrintWait("] vs.Ans");
			sPrefix = "[";
			if (ans_size <= 10) {
				for (KInt n = 0; n < ans_size; n++) {
					logger.PrintWait("%s%5.3f", sPrefix.c_str(), answer.get_at(m, n));
					sPrefix = ",";
				}
			}
			else {
				for (KInt n = 0; n < 3; n++) {
					logger.PrintWait("%s%5.3f", sPrefix.c_str(), answer.get_at(m, n));
					sPrefix = ",";
				}
				logger.PrintWait(",...");
				for (KInt n = ans_size - 3; n < ans_size; n++) {
					logger.PrintWait("%s%5.3f", sPrefix.c_str(), answer.get_at(m, n));
				}
			}
		}
		logger.Print("]");
	}
}

void KaiExecContext::m_shuffle_data(KaiDict& dataSection) {
	KInt data_start = dataSection["data_start"];
	KInt data_count = dataSection["data_count"];

	KaiMath* pMath = KaiMath::GetHostMath();

	// Get data indexes of all data
	KaiArray<KInt> total_index = NARRAY(get_property("data_index"));

	// Extract some data indexes from the all data indexes
	KaiArray<KInt> data_index = pMath->subrange(total_index, data_start, data_count);

	
#if (ACTIVATE_TEST && TEST_DISABLE_SHUFFLE)
	// Test code for debugging : No shuffle
#else
	// Shuffle
	pMath->shuffle(data_index);
#endif
	
	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXEC_CONTEXT_M_SHUFFLE_DATA)
	KaiArray<KInt> index_list = data_index;
	printf("[TRACE] %s(%u): index_list[%lld] : { ", __FUNCTION__, __LINE__, index_list.total_size());
	for (int i=0; i<10 && i<index_list.total_size(); ++i)
		printf("%lld ", index_list.get_at(i));
	if (index_list.total_size() > 11)
		printf("... ");
	if (index_list.total_size() > 10)
		printf("%lld ", index_list.get_at( index_list.total_size() - 1 ));
	printf("}\n");
#endif

	dataSection["data_index"] = data_index.get_core();
}

KFloat KaiExecContext::m_duration_milisec(KString sKey) {
	high_resolution_clock::time_point end_time = high_resolution_clock::now();
	duration<double, std::ratio<1, 1000>> time_span = duration_cast<duration<double, std::ratio<1, 1000>>>(end_time - m_timeMap[sKey]);
	return (KFloat)time_span.count() / 1000.0f;
}

void KaiExecContext::m_accumulate_loss(KaiDict& acc_loss, KaiDict loss, KInt mb_size) {
	for (auto& it : loss) {
		KString sKey = it.first;
		if (acc_loss.find(sKey) == acc_loss.end()) acc_loss[sKey] = 0.0f;
		acc_loss[sKey] = (KFloat)acc_loss[sKey] + mb_size * (KFloat)loss[sKey];
	}
}

void KaiExecContext::m_accumulate_accs(KaiDict& acc_accs, KaiDict accuracy, KInt mb_size) {
	for (auto& it : accuracy) {
		KString sKey = it.first;
		if (acc_accs.find(sKey) == acc_accs.end()) acc_accs[sKey] = 0.0f;
		KaiArray<KFloat> acc_arr = FARRAY(accuracy[sKey]);
		acc_accs[sKey] = (KFloat)acc_accs[sKey] + mb_size * m_pMath->fetch(acc_arr); // (KFloat)accuracy[sKey];
		accuracy[sKey] = m_pMath->fetch(acc_arr);
	}
}

KaiDict KaiExecContext::m_fetch_loss_values(KaiDict loss_arrs) {
	KaiDict loss_vals;

	for (auto& it : loss_arrs) {
		KString sKey = it.first;
		KaiArray<KFloat> loss_arr = FARRAY(it.second);
		if (loss_arr.total_size() != 1) throw KaiException(KERR_LOSS_FUNCTION_WITH_NONSCALAR_OUTPUT);
		KFloat loss = m_pMath->fetch(loss_arr, 0);
		loss_vals[sKey] = loss;
	}

	return loss_vals;
}