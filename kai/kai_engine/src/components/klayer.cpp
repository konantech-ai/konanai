/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "klayer.h"
#include "knetwork.h"
#include "koptimizer.h"
#include "../session/session.h"
#include "../utils/kutil.h"
#include "../exec/exec_context.h"
#include "../math/kshape.h"
#include "../math/kmath.h"

//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif


int KaiLayer::ms_checkCode = 27761801;

KStrList KaiLayer::ms_builtin = {
	"dense", "conv", "max", "avg", "globalavg", "batchnormal" , "activate", "dropout", "custom", "subnet",
	"rnn", "lstm", "gru", "self_attention", "extract", "embed", "select", "expand", "pass" };

KaiLayer::KaiLayer(KaiSession* pSession, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::layer, Ken_object_type::layer, kwArgs) {
	m_checkCode = ms_checkCode;
	m_set_default("name", "layer" + to_string(m_nComponentSeqId));
}

KaiLayer::KaiLayer(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiComponent(pSession, Ken_component_type::layer, Ken_object_type::layer, pLib, pComponentInfo) {
	m_checkCode = ms_checkCode;
	m_set_default("name", "layer" + to_string(m_nComponentSeqId));
}

/*
KaiLayer::KaiLayer(KaiLayer* pSrc, KaiDict kwArgs) : KaiComponent(pSrc->m_pSession, Ken_component_type::layer, Ken_object_type::layer, pSrc->m_propDict) {
	m_checkCode = ms_checkCode;
	m_set_default("name", "layer" + to_string(m_nComponentSeqId));
	m_replace_macro_args(kwArgs);
}
*/

KaiLayer::~KaiLayer() {
	m_checkCode = 0;
}

KaiLayer* KaiLayer::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Layer");

	KaiLayer* pLayer = (KaiLayer*)hObject;

	if (pLayer->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Layer");
	if (pLayer->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Layer");

	return pLayer;
}

KaiLayer* KaiLayer::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Layer");

	KaiLayer* pLayer = (KaiLayer*)hObject;

	if (pLayer->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Layer");

	return pLayer;
}

void KaiLayer::GetBuiltinNames(KStrList* pslNames) {
	for (auto it = ms_builtin.begin(); it != ms_builtin.end(); it++) {
		pslNames->push_back(*it);
	}
}

KaiLayer* KaiLayer::CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs) {
	KaiLayer* pInstance = NULL;

	if (sBuiltin == "dense") pInstance = new KaiDenseLayer(pSession, kwArgs);
	else if (sBuiltin == "conv") pInstance = new KaiConvLayer(pSession, kwArgs);
	else if (sBuiltin == "max") pInstance = new KaiMaxLayer(pSession, kwArgs);
	else if (sBuiltin == "avg") pInstance = new KaiAvgLayer(pSession, kwArgs);
	else if (sBuiltin == "globalavg") pInstance = new KaiGlobalAvgLayer(pSession, kwArgs);
	else if (sBuiltin == "batchnormal") pInstance = new KaiBatchNormalLayer(pSession, kwArgs);
	else if (sBuiltin == "activate") pInstance = new KaiActivateLayer(pSession, kwArgs);
	else if (sBuiltin == "dropout") pInstance = new KaiDropoutLayer(pSession, kwArgs);
	else if (sBuiltin == "custom") pInstance = new KaiCustomLayer(pSession, kwArgs);
	else if (sBuiltin == "subnet") pInstance = new KaiSubnetLayer(pSession, kwArgs);
	else if (sBuiltin == "rnn") pInstance = new KaiRnnLayer(pSession, kwArgs);
	else if (sBuiltin == "lstm") pInstance = new KaiLstmLayer(pSession, kwArgs);
	else if (sBuiltin == "gru") pInstance = new KaiGruLayer(pSession, kwArgs);
	else if (sBuiltin == "self_attention") pInstance = new KaiSelfAttentionLayer(pSession, kwArgs);
	else if (sBuiltin == "extract") pInstance = new KaiExtractLayer(pSession, kwArgs);
	else if (sBuiltin == "embed") pInstance = new KaiEmbedLayer(pSession, kwArgs);
	else if (sBuiltin == "select") pInstance = new KaiSelectLayer(pSession, kwArgs);
	else if (sBuiltin == "expand") pInstance = new KaiExpandLayer(pSession, kwArgs);
	else if (sBuiltin == "pass") pInstance = new KaiPassLayer(pSession, kwArgs);
	else if (sBuiltin == "stack") pInstance = new KaiStackLayer(pSession, kwArgs);

	if (!pInstance) throw KaiException(KERR_UNKNOWN_LAYER_SUBCLASS_NAME);

	pInstance->m_propDict["builtin"] = sBuiltin;
	pInstance->m_propDict["desc"] = pInstance->desc();

	return pInstance;
}

KString KaiLayer::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf, 128, "<KaiComponent Layer %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

KaiList KaiLayer::prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	//printf("layer %s: in shape %s\n", ((KString)info["builtin"]).c_str(), shape.desc().c_str());

	if (info.find("get") != info.end() && (KString)info["get"] != "") {
		KString sFieldName = info["get"];
		if (pack.find(sFieldName) == pack.end()) throw KaiException(KERR_FIELD_FOR_SET_ATTRIBUTE_NOT_FOUND, sFieldName);
		shape = pack[sFieldName];
	}

	KaiList info_pair = m_prepare_exec_info(shape, pOptimizer, info, pack);

	if (info.find("set") != info.end() && (KString)info["set"] != "") {
		KString sFieldName = info["set"];
		pack[sFieldName] = shape.copy();
	}

	//printf("layer %s: out shape %s\n", ((KString)info["builtin"]).c_str(), shape.desc().c_str());

	return info_pair;
}

KaiList KaiLayer::wrap_repeated_in_parallel_branch(KaiLayer* pLayer, KInt nRepeat, KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict subnetInfo, networkInfo, subnetParam;
	KaiList layerInfos, layerParams;

	networkInfo["builtin"] = "serial";

	subnetInfo["builtin"] = "subnet";
	subnetInfo["netinfo"] = networkInfo;
	subnetInfo["input_shape"] = shape.copy();

	for (KInt n = 0; n < nRepeat; n++) {
		KaiList info_pair = pLayer->m_prepare_exec_info(shape, pOptimizer, info, pack);
		layerInfos.push_back(info_pair[0]);
		layerParams.push_back(info_pair[1]);
	}

	subnetInfo["output_shape"] = shape.copy();

	subnetInfo["subnet"] = layerInfos;
	subnetParam["subnet"] = layerParams;

	return KaiList{ subnetInfo, subnetParam };
}

KaiArray<KFloat> KaiLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	static KInt nth_seed = 0;
	KInt nth = nth_seed++;
	layerInfo["nth"] = nth;
	KString subName;
	if (layerInfo.find("subnet") != layerInfo.end()) {
		subName = " " + (KString)((KaiDict)layerInfo["netinfo"])["builtin"];
		if (layerInfo.find("macro_name") != layerInfo.end()) subName += ":" + (KString)layerInfo["macro_name"];
	}
	
	//printf("layer %s%s(%lld): forward input shape %s\n", ((KString)layerInfo["builtin"]).c_str(), subName.c_str(), nth, xarr.shape().desc().c_str());

	if (layerInfo.find("get") != layerInfo.end()) {
		KString getName = layerInfo["get"];
		if (getName != "") {
			if (pack.find(getName) == pack.end()) throw KaiException(KERR_SET_ATTR_VALUE_NOT_FOUND, getName);
			KaiValue value = pack[getName];
			if (value.is_farray()) {
				xarr = FARRAY(value);
				//printf("get(%s)\n", getName.c_str());
			}
			else if (value.is_narray()) {
				pack["#narr"] = value;
				xarr = KaiArray<KFloat>();
				//printf("get(%s)\n", getName.c_str());
			}
			else throw KaiException(KERR_BAD_TYPE_DATA_FOR_SET_ATTR_VALUE, getName);
		}
	}

	KaiArray<KFloat> harr;
	KString sBuiltin = layerInfo["builtin"];

	if (sBuiltin == "dense") harr = KaiDenseLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "conv") harr = KaiConvLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "max") harr = KaiMaxLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "avg") harr = KaiAvgLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "globalavg") harr = KaiGlobalAvgLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "batchnormal") harr = KaiBatchNormalLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "activate") harr = KaiActivateLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "dropout") harr = KaiDropoutLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "subnet") harr = KaiSubnetLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "custom") harr = KaiCustomLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "rnn") harr = KaiRnnLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "lstm") harr = KaiLstmLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "gru") harr = KaiGruLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "self_attention") harr = KaiSelfAttentionLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "extract") harr = KaiExtractLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "embed") harr = KaiEmbedLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "select") harr = KaiSelectLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "expand") harr = KaiExpandLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "pass") harr = KaiPassLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else if (sBuiltin == "stack") harr = KaiStackLayer::forward(xarr, layerInfo, layerParam, pContext, pAux, pack);
	else throw KaiException(KERR_UNKNOWN_LAYER_NAME_IN_FORWARD);

	if (layerInfo.find("set") != layerInfo.end()) {
		KString setName = layerInfo["set"];
		if (setName != "") {
			pack[setName] = harr.get_core();
			//printf("set(%s)\n", setName.c_str());
		}
	}

	//printf("layer %s%s(%lld): forward output shape %s\n", ((KString)layerInfo["builtin"]).c_str(), subName.c_str(), nth, harr.shape().desc().c_str());

	return harr;
}

// 현재 버그으 원인은 set, get이매크로 단위 첫 레이어에 뿌려지는 것, 현재 DBL
// get은 첫 레이어, set은 마지막 레이어?

KaiArray<KFloat> KaiLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	//KInt nth = layerInfo["nth"];
	//KString subName;
	//if (layerInfo.find("subnet") != layerInfo.end()) {
	//	subName = " " + (KString)((KaiDict)layerInfo["netinfo"])["builtin"];
	//	if (layerInfo.find("macro_name") != layerInfo.end()) subName += ":" + (KString)layerInfo["macro_name"];
	//}
	//printf("layer %s%s(%lld): backprop y-grad %s\n", ((KString)layerInfo["builtin"]).c_str(), subName.c_str(), nth, garr.shape().desc().c_str());

	if (layerInfo.find("set") != layerInfo.end()) {
		KString setName = layerInfo["set"];
		if (setName != "" && pack.find(setName) != pack.end()) {
			//printf("set(%s)\n", setName.c_str());
			KaiArray<KFloat> grad = FARRAY(pack[setName]);
			if (garr.is_empty()) garr = grad;
			else garr = pContext->get_math()->add(garr, grad);
		}
	}

	KString sBuiltin = layerInfo["builtin"];

	if (sBuiltin == "subnet") garr = KaiSubnetLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "custom") garr = KaiCustomLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (garr.is_empty()) {
		// get 입력으로 입력이 대치되는 레이어는 기울기를 알려주지 못하며 여기 직렬로 연결된 단순 레이어들은 set에 의한 기울기 획득시까지 역전파 처리 않음 
		//printf("layer %s%s(%lld): backprop passing by empty grad!!!\n", ((KString)layerInfo["builtin"]).c_str(), subName.c_str(), nth);
	}
	else if (sBuiltin == "dense") garr = KaiDenseLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "conv") garr = KaiConvLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "max") garr = KaiMaxLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "avg") garr = KaiAvgLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "globalavg") garr = KaiGlobalAvgLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "batchnormal") garr = KaiBatchNormalLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "activate") garr = KaiActivateLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "dropout") garr = KaiDropoutLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "rnn") garr = KaiRnnLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "lstm") garr = KaiLstmLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "gru") garr = KaiGruLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "self_attention") garr = KaiSelfAttentionLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "extract") garr = KaiExtractLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "embed") garr = KaiEmbedLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "select") garr = KaiSelectLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "expand") garr = KaiExpandLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "pass") garr = KaiPassLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else if (sBuiltin == "stack") garr = KaiStackLayer::backprop(garr, layerInfo, layerParam, layerAux, pContext, pack);
	else throw KaiException(KERR_UNKNOWN_LAYER_NAME_IN_BACKPROP);

	if (layerInfo.find("get") != layerInfo.end()) {
		KString getName = layerInfo["get"];
		if (getName != "") {
			pack[getName] = garr.get_core();
			garr = KaiArray<KFloat>();
			//printf("get(%s)\n", getName.c_str());
		}
	}

	//printf("layer %s%s(%lld): backprop x-grad %s\n", ((KString)layerInfo["builtin"]).c_str(), subName.c_str(), nth, garr.shape().desc().c_str());

	return garr;
}

KInt KaiLayer::m_to_actfunc_id(KString funcname) {
	if (funcname == "none" || funcname == "") return (KInt) Ken_actfunc::none;
	else if (funcname == "relu") return (KInt)Ken_actfunc::relu;
	else if (funcname == "sigmoid") return (KInt)Ken_actfunc::sigmoid;
	else if (funcname == "tanh") return (KInt)Ken_actfunc::tanh;
	else if (funcname == "leaky_relu") return (KInt)Ken_actfunc::leaky_relu;
	else if (funcname == "gelu") return (KInt)Ken_actfunc::gelu;
	else if (funcname == "custom") return (KInt)Ken_actfunc::custom;

	throw KaiException(KERR_UNKNOWN_ACTFUNCNAME, funcname);
}

KaiArray<KFloat> KaiLayer::ms_extract_weight(KaiDict pm) {
	KaiDict pm_w = pm["w"];
	KaiArray<KFloat> weight = FARRAY(pm_w["_pm_"]);
	return weight;
}

KaiArray<KFloat> KaiLayer::ms_extract_bias(KaiDict pm) {
	KaiDict pm_b = pm["b"];
	KaiArray<KFloat> weight = FARRAY(pm_b["_pm_"]);
	return weight;
}

KaiArray<KFloat> KaiLayer::ms_extract_weight_grad(KaiDict pm) {
	KaiDict pm_w = pm["w"];
	KaiArray<KFloat> weight = FARRAY(pm_w["_grad_"]);
	return weight;
}

KaiArray<KFloat> KaiLayer::ms_extract_bias_grad(KaiDict pm) {
	KaiDict pm_b = pm["b"];
	KaiArray<KFloat> weight = FARRAY(pm_b["_grad_"]);
	return weight;
}

KBool KaiLayer::ms_get_debug_trace(KaiDict layerInfo, KString phase) {
	if (layerInfo.find("debug_trace") == layerInfo.end()) return false;

	KaiDict debug_trace = layerInfo["debug_trace"];

	if (debug_trace.find("phase") != debug_trace.end()) {
		KaiList phases = debug_trace["phase"];
		if (!phases.find_string(phase)) return false;
	}

	if (debug_trace.find("targets") != debug_trace.end()) {
		KaiList target = debug_trace["targets"];
		if (target.find_string((KString)layerInfo["builtin"])) return true;
	}

	if (debug_trace.find("checked") != debug_trace.end()) {
		KaiList checked = debug_trace["checked"];
		if (checked.find_string((KString)layerInfo["builtin"])) return false;
		return true;
	}

	return false;
}

void KaiLayer::set_aux(KaiDict* pAux, KString sKey, KaiValue value) {
	if (pAux) {
		KaiDict& dict = *pAux;
		dict[sKey] = value;
	}
}

KaiShape KaiLayer::get_2d_option(KaiDict call_info, KString sKey, KaiShape sDef) {
	if (call_info.find(sKey) == call_info.end()) return sDef;
	KaiValue val = call_info[sKey];
	if (val.type() == Ken_value_type::kint) {
		KInt nVal = val;
		return KaiShape{ nVal, nVal };
	}
	else if (val.type() == Ken_value_type::shape) {
		KaiShape shape = val;
		if (shape.size() != 2) throw KaiException(KERR_BAD_2D_OPTION_VALUE, sKey);
		return shape;
	}
	else {
		throw KaiException(KERR_BAD_2D_OPTION_VALUE, sKey);
	}
}

KaiArray<KFloat> KaiLayer::get_param(KaiDict layerParam, KString sParamName, KString sSubName) {
	KaiDict paramset = layerParam[sParamName];
	KaiDict param = paramset[sSubName];
	KaiArray<KFloat> pm =FARRAY( param["_pm_"]);
	return pm;
}

KaiDenseLayer::KaiDenseLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiDenseLayer::~KaiDenseLayer() {
}

KaiList KaiDenseLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["actfunc_id"] = m_to_actfunc_id(get_info_prop(info, "actfunc", "relu"));

	KaiShape xshape = shape.copy();
	KaiShape yshape = xshape.replace_end(info["width"]);

	info["input_shape"] = xshape;
	info["output_shape"] = yshape;

	if (xshape.size() != 1 || yshape.size() != 1) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

	KaiShape wshape = KaiShape{ xshape[-1], yshape[-1] };
	KBool use_bias = kutil.seek_set_dict(info, "use_bias", true);
	KString init_weight = kutil.seek_set_dict(info, "init_weight", "gaussian");
	KFloat init_std = kutil.seek_set_dict(info, "init_std", 0.030f);

	param["weight"] = pOptimizer->createAffineParams(wshape, true, use_bias, init_weight, init_std);
	
	shape = yshape;

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

KaiArray<KFloat> KaiDenseLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	set_aux(pAux, "temp_x", xarr.get_core());

	KaiShape xshape = xarr.shape();

	set_aux(pAux, "xshape", xshape);

	if (xshape.total_size() % input_shape.total_size() != 0) throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "출력 형상 불일치");

	if (xshape.size() != 2 || xshape[1] != input_shape[0]) {
		KInt mb_size = xshape.total_size() / input_shape.total_size();
		KaiShape xshape_2d{ mb_size , input_shape[0] };
		xarr = xarr.reshape(xshape_2d);
	}

	KaiShape yshape = xshape.replace_tail_by_size(input_shape, output_shape);

	//KaiDict optimizer_info = pContext->get_component_property("optimizer");

	KaiArray<KFloat> yarr = KaiOptimizer::forward_affine(xarr, layerInfo, layerParam["weight"], pContext, pAux);

	if (debug_trace) {
		KaiArray<KFloat> pm_w = ms_extract_weight(layerParam["weight"]);
		KaiArray<KFloat> pm_b = ms_extract_bias(layerParam["weight"]);

		pm_w.dump("pm_w");
		pm_b.dump("pm_b");
		xarr.dump("xarr");
		yarr.dump("yarr");
	}

	set_aux(pAux, "pre_act", yarr.get_core());

	yarr = pContext->get_math()->acivate(yarr, layerInfo["actfunc_id"], pContext);
	yarr = yarr.reshape(yshape);

	if (debug_trace) {
		yarr.dump("yarr_activated");
	}

	set_aux(pAux, "post_act", yarr.get_core());

	return yarr;
}

KaiArray<KFloat> KaiDenseLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	KaiArray<KFloat> pre_act = FARRAY(layerAux["pre_act"]);
	KaiArray<KFloat> post_act = FARRAY(layerAux["post_act"]);

	KaiShape output_shape = layerInfo["output_shape"];

	KaiShape yshape = gyarr.shape();

	if (yshape[-1] != output_shape[0]) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

	if (yshape.size() != 2) {
		KInt mb_size = yshape.total_size() / output_shape[0];
		KaiShape yshape_2d{ mb_size , output_shape[0] };
		gyarr = gyarr.reshape(yshape_2d);
	}

	KaiArray<KFloat> gaarr = pContext->get_math()->acivate_backprop(gyarr, pre_act, post_act, layerInfo["actfunc_id"], pContext);

	KaiDict pm = layerParam["weight"];

	KaiArray<KFloat> gxarr = KaiOptimizer::backprop_affine(gaarr, layerInfo, layerAux, pContext, pm);

	KaiShape xshape = layerAux["xshape"];
	gxarr = gxarr.reshape(xshape);

	if (debug_trace) {
		KaiArray<KFloat> temp_x = FARRAY(layerAux["temp_x"]);

		KaiArray<KFloat> pm_w = ms_extract_weight(layerParam["weight"]);
		KaiArray<KFloat> pm_b = ms_extract_bias(layerParam["weight"]);
		KaiArray<KFloat> pm_w_grad = ms_extract_weight_grad(layerParam["weight"]);
		KaiArray<KFloat> pm_b_grad = ms_extract_bias_grad(layerParam["weight"]);

		temp_x.dump("xarr");
		pm_w.dump("pm_w");
		pm_b.dump("pm_b");
		pm_w_grad.dump("pm_w_grad");
		pm_b_grad.dump("pm_b_grad");

		pre_act.dump("pre_act");
		post_act.dump("post_act");

		gyarr.dump("gyarr");
		gaarr.dump("gaarr");
		gxarr.dump("gxarr");
	}

	return gxarr;
}

KaiConvLayer::KaiConvLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiConvLayer::~KaiConvLayer() {
}

KaiList KaiConvLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	if (shape.size() < 3) throw KaiException(KERR_TOO_SHORT_INPUT_SHAPE_FOR_CNN_LAYER);

	info["actfunc_id"] = m_to_actfunc_id(get_info_prop(info, "actfunc", "relu"));
	info["padding"] = get_info_prop(info, "padding", "same");
	info["ksize"] = get_2d_option(info, "ksize");
	info["stride"] = get_2d_option(info, "stride");

	KaiShape xshape = shape.copy();
	KaiShape yshape = shape.copy();

	KInt yh = yshape[-3], yw = yshape[-2];

	KaiShape ksize = info["ksize"];

	if (info["padding"] == "valid") {
		yh -= (KInt)ksize[0] - 1;
		yw -= (KInt)ksize[1] - 1;
	}

	KaiShape stride = info["stride"];

	KInt sh = stride[0], sw = stride[1];

	yshape[-3] = (yh + sh / 2) / sh;
	yshape[-2] = (yw + sw / 2) / sw;
	yshape[-1] = info["chn"];

	info["input_shape"] = xshape;
	info["output_shape"] = yshape;

	KaiShape kshape = KaiShape{ ksize[0], ksize[1], xshape[-1], yshape[-1] };
	KBool use_bias = kutil.seek_set_dict(info, "use_bias", true);
	KString init_weight = kutil.seek_set_dict(info, "init_weight", "gaussian");
	KFloat init_std = kutil.seek_set_dict(info, "init_std", 0.030f);

	param["kernel"] = pOptimizer->createAffineParams(kshape, true, use_bias, init_weight, init_std);

	shape = yshape.copy();

	return KaiList{ info, param };
}

KaiArray<KFloat> KaiConvLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_tail(input_shape, output_shape);

	//KaiDict optimizer_info = pContext->get_component_property("optimizer");

	//xarr.dump("Conv Input", true);
	KaiArray<KFloat> yarr = KaiOptimizer::forward_conv(xarr, layerInfo, layerParam["kernel"], pContext, pAux);
	//yarr.dump("Conv output", true);

	if (layerInfo["padding"] == "valid") {
		set_aux(pAux, "pre_valid", yarr.shape().copy());
		KaiShape ksize = layerInfo["ksize"];
		yarr = pContext->get_math()->subrange(yarr, -3, ((KInt)ksize[0] - 1) / 2, xshape[-3] - (KInt)ksize[0] + 1);
		yarr = pContext->get_math()->subrange(yarr, -2, ((KInt)ksize[1] - 1) / 2, xshape[-2] - (KInt)ksize[1] + 1);
	}

	KaiShape stride = layerInfo["stride"];

	if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
		set_aux(pAux, "pre_stride", yarr.shape().copy());
		yarr = pContext->get_math()->stride(yarr, stride);
	}

	if (yarr.shape() != yshape) throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "출력 형상 불일치");

	set_aux(pAux, "pre_act", yarr.get_core());
	yarr = pContext->get_math()->acivate(yarr, layerInfo["actfunc_id"], pContext);
	set_aux(pAux, "post_act", yarr.get_core());
	//yarr.dump("Actfunc output", true);

	return yarr;
}

KaiArray<KFloat> KaiConvLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiArray<KFloat> pre_act = FARRAY(layerAux["pre_act"]);
	KaiArray<KFloat> post_act = FARRAY(layerAux["post_act"]);

	//garr.dump("Conv backprop input", true);
	garr = pContext->get_math()->acivate_backprop(garr, pre_act, post_act, layerInfo["actfunc_id"], pContext);
	//garr.dump("Conv backprop activate output", true);

	KaiShape stride = layerInfo["stride"];

	if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
		KaiShape xshape = layerAux["pre_stride"];
		garr = pContext->get_math()->stride_derv(garr, stride, xshape);
	}

	if (layerInfo["padding"] == "valid") {
		KaiShape ksize = layerInfo["ksize"];
		KaiShape xshape = layerAux["pre_valid"];
		garr = pContext->get_math()->subrange_derv(garr, -2, ((KInt)ksize[1] - 1) / 2, xshape[-2] - (KInt)ksize[1] + 1);
		garr = pContext->get_math()->subrange_derv(garr, -3, ((KInt)ksize[0] - 1) / 2, xshape[-3] - (KInt)ksize[0] + 1);
	}

	//KaiDict optimizer_info = pContext->get_component_property("optimizer");
	KaiDict pm = layerParam["kernel"];

	garr = KaiOptimizer::backprop_conv(garr, layerInfo, layerAux, pContext, pm);
	//garr.dump("Conv backprop output", true);

	return garr;
}

KaiMaxLayer::KaiMaxLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiMaxLayer::~KaiMaxLayer() {
}

KaiList KaiMaxLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["padding"] = get_info_prop(info, "padding", "same");
	info["stride"] = get_2d_option(info, "stride");
	info["ksize"] = get_2d_option(info, "ksize", info["stride"]);

	if (shape.size() < 3) throw KaiException(KERR_TOO_SHORT_INPUT_SHAPE_FOR_CNN_LAYER);

	KaiShape xshape = shape.copy();
	KaiShape yshape = shape.copy();

	KInt yh = yshape[-3], yw = yshape[-2];

	KaiShape ksize = info["ksize"];

	if (info["padding"] == "valid") {
		yh -= (KInt)ksize[0] - 1;
		yw -= (KInt)ksize[1] - 1;
	}

	KaiShape stride = info["stride"];

	KInt sh = stride[0], sw = stride[1];

	yshape[-3] = (yh + sh / 2) / sh;
	yshape[-2] = (yw + sw / 2) / sw;

	info["input_shape"] = xshape;
	info["output_shape"] = yshape.copy();

	shape = yshape;

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

KaiArray<KFloat> KaiMaxLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_tail(input_shape, output_shape);

	KaiShape ksize = layerInfo["ksize"];

	//xarr.dump("Max input", true);

	KaiArray<KInt> max_map;
	KaiArray<KFloat> yarr = pContext->get_math()->max_pool(xarr, &max_map, ksize);
	//yarr.dump("Max output", true);
	//max_map.dump("Max map output", true);

	set_aux(pAux, "map", max_map.get_core());

	if (layerInfo["padding"] == "valid") {
		set_aux(pAux, "pre_valid", yarr.shape().copy());
		yarr = pContext->get_math()->subrange(yarr, -3, ((KInt)ksize[0] - 1) / 2, xshape[-3] - (KInt)ksize[0] + 1);
		yarr = pContext->get_math()->subrange(yarr, -2, ((KInt)ksize[1] - 1) / 2, xshape[-2] - (KInt)ksize[1] + 1);
	}

	KaiShape stride = layerInfo["stride"];

	if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
		set_aux(pAux, "pre_stride", yarr.shape().copy());
		yarr = pContext->get_math()->stride(yarr, stride);
	}

	//yarr.dump("Max after stride output", true);

	if (yarr.shape() != yshape) throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "출력 형상 불일치");

	return yarr;
}

KaiArray<KFloat> KaiMaxLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape stride = layerInfo["stride"];

	//garr.dump("Max backprop input", true);
	if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
		KaiShape xshape = layerAux["pre_stride"];
		garr = pContext->get_math()->stride_derv(garr, stride, xshape);
	}
	//garr.dump("Max backprop stride output", true);

	if (layerInfo["padding"] == "valid") {
		KaiShape ksize = layerInfo["ksize"];
		KaiShape xshape = layerAux["pre_valid"];
		garr = pContext->get_math()->subrange_derv(garr, -2, ((KInt)ksize[1] - 1) / 2, xshape[-2] - (KInt)ksize[1] + 1);
		garr = pContext->get_math()->subrange_derv(garr, -3, ((KInt)ksize[0] - 1) / 2, xshape[-3] - (KInt)ksize[0] + 1);
	}

	KaiShape kshape = layerInfo["ksize"];
	KaiArray<KInt> max_map = NARRAY(layerAux["map"]);

	garr = pContext->get_math()->max_pool_derv(garr, max_map, kshape);
	//garr.dump("Max backprop output", true);

	return garr;
}

KaiAvgLayer::KaiAvgLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiAvgLayer::~KaiAvgLayer() {
}

KaiList KaiAvgLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["padding"] = get_info_prop(info, "padding", "same");
	info["stride"] = get_2d_option(info, "stride");
	info["ksize"] = get_2d_option(info, "ksize", info["stride"]);

	if (shape.size() < 3) throw KaiException(KERR_TOO_SHORT_INPUT_SHAPE_FOR_CNN_LAYER);

	KaiShape xshape = shape.copy();
	KaiShape yshape = shape.copy();

	KInt yh = yshape[-3], yw = yshape[-2];

	KaiShape ksize = info["ksize"];

	if (info["padding"] == "valid") {
		yh -= (KInt)ksize[0] - 1;
		yw -= (KInt)ksize[1] - 1;
	}

	KaiShape stride = info["stride"];

	KInt sh = stride[0], sw = stride[1];

	yshape[-3] = (yh + sh / 2) / sh;
	yshape[-2] = (yw + sw / 2) / sw;

	info["input_shape"] = xshape;
	info["output_shape"] = yshape;

	info["parameters"] = KaiList();

	shape = yshape;

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

KaiArray<KFloat> KaiAvgLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_tail(input_shape, output_shape);

	KaiShape ksize = layerInfo["ksize"];

	//xarr.dump("Avg input", true);
	KaiArray<KInt> avg_map;
	KaiArray<KFloat> yarr = pContext->get_math()->avg_pool(xarr, &avg_map, ksize);
	//yarr.dump("Avg output", true);

	set_aux(pAux, "map", avg_map.get_core());

	if (layerInfo["padding"] == "valid") {
		set_aux(pAux, "pre_valid", yarr.shape().copy());
		yarr = pContext->get_math()->subrange(yarr, -3, ((KInt)ksize[0] - 1) / 2, xshape[-3] - (KInt)ksize[0] + 1);
		yarr = pContext->get_math()->subrange(yarr, -2, ((KInt)ksize[1] - 1) / 2, xshape[-2] - (KInt)ksize[1] + 1);
	}

	KaiShape stride = layerInfo["stride"];

	if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
		set_aux(pAux, "pre_stride", yarr.shape().copy());
		yarr = pContext->get_math()->stride(yarr, stride);
	}
	//yarr.dump("Avg stride output", true);

	if (yarr.shape() != yshape) throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "출력 형상 불일치");

	return yarr;
}

KaiArray<KFloat> KaiAvgLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape stride = layerInfo["stride"];

	//garr.dump("Avg backprop input", true);
	if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
		KaiShape xshape = layerAux["pre_stride"];
		garr = pContext->get_math()->stride_derv(garr, stride, xshape);
	}
	//garr.dump("Avg backprop stride output", true);

	KaiShape ksize = layerInfo["ksize"];

	if (layerInfo["padding"] == "valid") {
		KaiShape xshape = layerAux["pre_valid"];
		garr = pContext->get_math()->subrange_derv(garr, -2, ((KInt)ksize[1] - 1) / 2, xshape[-2] - (KInt)ksize[1] + 1);
		garr = pContext->get_math()->subrange_derv(garr, -3, ((KInt)ksize[0] - 1) / 2, xshape[-3] - (KInt)ksize[0] + 1);
	}

	KaiArray<KInt> avg_map = NARRAY(layerAux["map"]);

	garr = pContext->get_math()->avg_pool_derv(garr, avg_map, ksize);
	//garr.dump("Avg backprop output", true);

	return garr;
}

KaiGlobalAvgLayer::KaiGlobalAvgLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiGlobalAvgLayer::~KaiGlobalAvgLayer() {
}

KaiList KaiGlobalAvgLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	if (shape.size() < 3) throw KaiException(KERR_TOO_SHORT_INPUT_SHAPE_FOR_CNN_LAYER);

	KaiShape xshape = shape.copy();
	KaiShape yshape = shape.cut_tail(3).append(shape[-1]);

	info["input_shape"] = xshape;
	info["output_shape"] = yshape;

	info["parameters"] = KaiList();

	shape = yshape;

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

KaiArray<KFloat> KaiGlobalAvgLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_tail(input_shape, output_shape);

	//xarr.dump("Globalavg input", true);
	KaiArray<KFloat> yarr = pContext->get_math()->globalavg(xarr);
	//yarr.dump("Globalavg output", true);

	if (yarr.shape() != yshape) throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "출력 형상 불일치");

	return yarr;
}

KaiArray<KFloat> KaiGlobalAvgLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KaiShape yshape = garr.shape();
	KaiShape xshape = yshape.replace_tail(output_shape, input_shape);

	//garr.dump("Globalavg backprop input", true);
	garr = pContext->get_math()->globalavg_derv(garr, xshape);
	//garr.dump("Globalavg backprop output", true);

	if (garr.shape() != xshape) throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "출력 형상 불일치");

	return garr;
}

KaiBatchNormalLayer::KaiBatchNormalLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiBatchNormalLayer::~KaiBatchNormalLayer() {
}

KaiList KaiBatchNormalLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["rescale"] = get_info_prop(info, "rescale", true);
	info["epsilon"] = get_info_prop(info, "epsilon", 0.001f);
	info["momentum"] = get_info_prop(info, "momentum", 0.99f);

	info["input_shape"] = shape;
	info["output_shape"] = shape;

	KaiShape nshape = KaiShape{ shape[-1] };

	param["mavg"] = pOptimizer->createAffineParams(nshape, false, false, "zeros");
	param["mvar"] = pOptimizer->createAffineParams(nshape, false, false, "ones");

	if ((KBool)info["rescale"]) {
		param["scale"] = pOptimizer->createAffineParams(nshape, true, false, "ones");
		param["shift"] = pOptimizer->createAffineParams(nshape, true, false, "zeros");
	}

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

KaiArray<KFloat> KaiBatchNormalLayer::forward(KaiArray<KFloat> harr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KaiArray<KFloat> mavg = get_param(layerParam, "mavg");
	KaiArray<KFloat> mvar = get_param(layerParam, "mvar");

	set_aux(pAux, "org_x", harr.get_core());

	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (debug_trace) {
		harr.dump("harr_in");
		mavg.dump("mavg_pre");
		mvar.dump("mvar_pre");
	}

	KaiMath* pMath = pContext->get_math();

	if (pAux) {
		KaiArray<KFloat> var;
		harr = pMath->BNCollectNorm(harr, mavg, mvar, var, layerInfo["momentum"], layerInfo["epsilon"]);
		set_aux(pAux, "var", var.get_core());
	}
	else harr = pMath->BnNormalize(harr, mavg, mvar, layerInfo["epsilon"]);

	if (debug_trace) {
		harr.dump("harr_normed");
		mavg.dump("mavg_post");
		mvar.dump("mvar_post");
	}

	set_aux(pAux, "norm_x", harr.get_core());

	if ((KBool)layerInfo["rescale"]) {
		KaiArray<KFloat> scale = get_param(layerParam, "scale");
		KaiArray<KFloat> shift = get_param(layerParam, "shift");

		if (debug_trace) {
			scale.dump("scale_pre");
			shift.dump("shift_pre");
		}

		harr = pMath->BnScale(harr, scale, shift);

		if (debug_trace) {
			scale.dump("scale_post");
			shift.dump("shift_post");
			harr.dump("harr_scaled");
		}
	}

	return harr;
}

KaiArray<KFloat> KaiBatchNormalLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	KaiArray<KFloat> gnarr = gyarr;

	if ((KBool)layerInfo["rescale"]) {
		KaiDict pm_scale = layerParam["scale"];
		KaiDict pm_shift = layerParam["shift"];

		KaiArray<KFloat> org_x = FARRAY(layerAux["org_x"]);
		KaiArray<KFloat> norm_x = FARRAY(layerAux["norm_x"]);

		gnarr = KaiOptimizer::backprop_rescale(gyarr, norm_x, pm_scale, pm_shift, pContext);

		if (debug_trace) {
			KaiArray<KFloat> pm_sc = ms_extract_weight(pm_scale);
			KaiArray<KFloat> pm_sh = ms_extract_weight(pm_shift);
			KaiArray<KFloat> pm_sc_g = ms_extract_weight_grad(pm_scale);
			KaiArray<KFloat> pm_sh_g = ms_extract_weight_grad(pm_shift);

			gyarr.dump("gyarr");
			pm_sc.dump("pm_scale");
			pm_sh.dump("pm_shift");
			org_x.dump("org_x");
			norm_x.dump("norm_x");
			pm_sc_g.dump("pm_scale_grad");
			pm_sh_g.dump("pm_shift_grad");
		}
	}

	KaiArray<KFloat> var = FARRAY(layerAux["var"]);
	
	KaiArray<KFloat> gxarr = pContext->get_math()->BnNormDerv(gnarr, var, layerInfo["epsilon"]);
	
	if (debug_trace) {
		var.dump("var");
		gnarr.dump("gnarr");
		gxarr.dump("gxarr");
	}

	return gxarr;
}

KaiActivateLayer::KaiActivateLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiActivateLayer::~KaiActivateLayer() {
}

KaiList KaiActivateLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["actfunc_id"] = m_to_actfunc_id(get_info_prop(info, "actfunc", "relu"));

	info["input_shape"] = shape;
	info["output_shape"] = shape;

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

KaiArray<KFloat> KaiActivateLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	set_aux(pAux, "pre_act", xarr.get_core());
	xarr = pContext->get_math()->acivate(xarr, layerInfo["actfunc_id"], pContext);
	set_aux(pAux, "post_act", xarr.get_core());

	return xarr;
}

KaiArray<KFloat> KaiActivateLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiArray<KFloat> pre_act = FARRAY(layerAux["pre_act"]);
	KaiArray<KFloat> post_act = FARRAY(layerAux["post_act"]);

	garr = pContext->get_math()->acivate_backprop(garr, pre_act, post_act, layerInfo["actfunc_id"], pContext);

	return garr;
}

KaiDropoutLayer::KaiDropoutLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiDropoutLayer::~KaiDropoutLayer() {
}

KaiList KaiDropoutLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["keep_ratio"] = get_info_prop(info, "keep_ratio", 0.9f);

	info["input_shape"] = shape;
	info["output_shape"] = shape;

	return KaiList{ info, param };
}

KaiArray<KFloat> KaiDropoutLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (pAux) {	// means in training
		KFloat keep_ratio = layerInfo["keep_ratio"];
		KaiShape xshape = xarr.shape();

		KaiArray<KFloat> mask = pContext->get_math()->random_bernoulli(xshape, keep_ratio);
		KaiArray<KFloat> yarr = pContext->get_math()->dropout(xarr, mask, keep_ratio);

		set_aux(pAux, "mask", mask.get_core());

		if (debug_trace) {
			xarr.dump("xarr");
			mask.dump("mask");
			yarr.dump("yarr");
		}

		return yarr;
	}
	else {	// means not in training
		return xarr;
	}
}

KaiArray<KFloat> KaiDropoutLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	KFloat keep_ratio = layerInfo["keep_ratio"];
	KaiArray<KFloat> mask = FARRAY(layerAux["mask"]);
	KaiArray<KFloat> gxarr = pContext->get_math()->dropout_derv(gyarr, mask, keep_ratio);

	if (debug_trace) {
		gyarr.dump("gyarr");
		mask.dump("mask");
		gxarr.dump("gxarr");
	}

	return gxarr;
}

KaiRecurrentLayer::KaiRecurrentLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
	/*
	* 아래의 유형들을 지원할 수 있도록 옵션화 및 확장 필요
Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification). => not RNN
Sequence output (e.g. image captioning takes an image and outputs a sentence of words).
Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment)
Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French).
Synced sequence input and output (e.g. video classification where we wish to label each frame of the video).	*/
}

KaiRecurrentLayer::~KaiRecurrentLayer() {
}

KaiList KaiRecurrentLayer::m_prepare_recurrent_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KInt nBlocks, KInt nExtra) {
	KaiDict param;

	info["input_seq"] = get_info_prop(info, "input_seq", true);
	info["output_seq"] = get_info_prop(info, "output_seq", true);

	info["actfunc_id"] = m_to_actfunc_id(get_info_prop(info, "actfunc", "tanh"));	// only for RNN
	info["use_state"] = kutil.seek_set_dict(info, "use_state", false);	// only for lstm

	KaiShape xshape = shape.copy();
	KaiShape yshape = xshape.replace_end(info["width"]);

	info["input_shape"] = xshape.copy();
	info["output_shape"] = yshape.copy();

	KInt inp_size = xshape.total_size();
	KInt out_size = yshape.total_size();
	KInt exp_size = inp_size + out_size;

	info["inp_size"] = inp_size;
	info["out_size"] = out_size;
	info["exp_size"] = exp_size;

	KBool use_bias = kutil.seek_set_dict(info, "use_bias", true);
	KString init_weight = kutil.seek_set_dict(info, "init_weight", "gaussian");
	KFloat init_std = kutil.seek_set_dict(info, "init_std", 0.030f);

	KaiShape wshape = KaiShape{ exp_size, nBlocks * out_size };
	param["weight"] = pOptimizer->createAffineParams(wshape, true, use_bias, init_weight, init_std);

	if (nExtra > 0) {
		KaiShape eshape = KaiShape{ exp_size, nExtra * out_size };
		param["extra"] = pOptimizer->createAffineParams(eshape, true, use_bias, init_weight, init_std);
	}

	shape = yshape;

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

#include "../math/khostmath.h"

KaiArray<KFloat> KaiRecurrentLayer::ms_forward(rec_cell cell, KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KInt inp_size = layerInfo["inp_size"];
	KInt out_size = layerInfo["out_size"];
	KInt exp_size = layerInfo["exp_size"];

	KBool input_seq = layerInfo["input_seq"];
	KBool output_seq = layerInfo["output_seq"];

	KInt mb_size = xarr.axis_size(0);
	KInt actfunc_id = layerInfo["actfunc_id"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape;
	KInt timesteps;

	KaiMath* pMath = pContext->get_math();

	KaiArray<KFloat> yarr;

	if (input_seq) {
		if (xshape.size() < input_shape.size() + 2) throw KaiException(KERR_NO_SEQ_DIM_IN_RECURRENT_LAYER_DATA);
		timesteps = xshape[-input_shape.size() - 1];
		if (output_seq) {
			yshape = xshape.replace_tail(input_shape, output_shape);
			yarr = pMath->zeros(yshape);
		}
		else {
			yshape = xshape.cut_tail(input_shape.size() + 1).append(output_shape);
		}
	}
	else {
		if (output_seq) {
			if (layerInfo.find("timesteps") != layerInfo.end()) timesteps = layerInfo["timesteps"];
			else if (layerInfo.find("output_timesteps") != layerInfo.end()) timesteps = layerInfo["output_timesteps"];
			else throw KaiException(KERR_OUTPUT_TIMESTEPS_UNKNOWN_IN_RNN);
			yshape = xshape.cut_tail(input_shape.size()).append(timesteps).append(output_shape);
			yarr = pMath->zeros(yshape);
		}
		else throw KaiException(KERR_RECURRENT_LAYER_WITHOUT_SEQ_DATA);
	}

	set_aux(pAux, "xshape", xshape);
	set_aux(pAux, "yshape", yshape);
	set_aux(pAux, "timesteps", timesteps);

	KaiList seqAuxs;

	KaiArray<KFloat> recurrent = pMath->zeros(KaiShape{ mb_size, out_size });
	KaiArray<KFloat> state;
	KBool use_state = layerInfo["use_state"];

	if (cell == rec_cell::lstm) {
		/*
		recurrent = hostmath.random_uniform(KaiShape{ mb_size, out_size });
		state = hostmath.random_uniform(KaiShape{ mb_size, out_size }); //pMath->random_uniform(KaiShape{ mb_size, out_size }); // will be used only in LSTM
		recurrent = pMath->to_cuda(recurrent);
		state = pMath->to_cuda(state);
		*/
		state = pMath->zeros(KaiShape{ mb_size, out_size }); // will be used only in LSTM
	}

	KaiList stepAuxes;

	KaiDict pm = layerParam["weight"];

	for (KInt n = 0; n < timesteps; n++) {
		KaiDict auxStep;
		KaiDict* pAuxStep = pAux ? &auxStep : NULL;

		KaiArray<KFloat> exp_input = pMath->CombineExtendedInput(recurrent, input_seq, xarr, n);
		KaiArray<KFloat> affine = KaiOptimizer::forward_affine(exp_input, layerInfo, pm, pContext, pAuxStep);

		if (cell == rec_cell::rnn) {
			set_aux(pAuxStep, "pre_act", affine.get_core());
			recurrent = pMath->acivate(affine, actfunc_id, pContext);
			set_aux(pAuxStep, "post_act", recurrent.get_core());
		}
		else if (cell == rec_cell::lstm) {
			set_aux(pAuxStep, "pre_state", state.get_core());
			KaiArray<KFloat> gates = pMath->lstm_gates(affine);
			set_aux(pAuxStep, "gates", gates.get_core());
			recurrent = pMath->lstm_proc(gates, state, use_state);
			set_aux(pAuxStep, "post_recur", recurrent.get_core());
		}
		else if (cell == rec_cell::gru) {
			set_aux(pAuxStep, "pre_recur", recurrent.get_core());
			KaiArray<KFloat> gates = pMath->sigmoid(affine);
			set_aux(pAuxStep, "gates", gates.get_core());
			KaiArray<KFloat> exp_inp_extra = pMath->gru_combine_extra(exp_input, gates);
			set_aux(pAuxStep, "exp_inp_extra", exp_inp_extra.get_core());
			KaiArray<KFloat> extra_affine = KaiOptimizer::forward_affine(exp_inp_extra, layerInfo, layerParam["extra"], pContext, pAuxStep);
			set_aux(pAuxStep, "extra_affine", extra_affine.get_core());
			recurrent = pMath->gru_proc(gates, recurrent, extra_affine);
		}
		else throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

		if (output_seq) {
			if (cell == rec_cell::lstm && use_state) pMath->CopyIntoTimeSlice(yarr, state, n);
			else pMath->CopyIntoTimeSlice(yarr, recurrent, n);
		}

		stepAuxes.push_back(auxStep);
	}

	set_aux(pAux, "stepAuxes", stepAuxes);

	if (!output_seq) {
		if (cell == rec_cell::lstm && use_state) yarr = state;
		yarr = recurrent;
	}

	return yarr;
}

KaiArray<KFloat> KaiRecurrentLayer::ms_backprop(rec_cell cell, KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KInt inp_size = layerInfo["inp_size"];
	KInt out_size = layerInfo["out_size"];
	KInt exp_size = layerInfo["exp_size"];

	KBool input_seq = layerInfo["input_seq"];
	KBool output_seq = layerInfo["output_seq"];

	KInt mb_size = garr.axis_size(0);
	KInt actfunc_id = layerInfo["actfunc_id"];

	KaiShape xshape = layerAux["xshape"];
	KaiShape yshape = layerAux["yshape"];

	KInt timesteps = layerAux["timesteps"];

	KaiMath* pMath = pContext->get_math();

	KaiArray<KFloat> g_x = pMath->zeros(xshape);
	KaiArray<KFloat> g_recurrent = output_seq ? pMath->zeros(KaiShape{ mb_size, out_size }) : pMath->copy(garr);
	KaiArray<KFloat> g_state;

	KBool use_state = layerInfo["use_state"];

	if (cell == rec_cell::lstm) {
		g_state = pMath->zeros(KaiShape{ mb_size, out_size }); // will be used only in LSTM
		if (use_state && !output_seq) {
			g_state = g_recurrent;
			g_recurrent = pMath->zeros(KaiShape{ mb_size, out_size }); // will be used only in LSTM
		}
	}

	KaiList stepAuxes = layerAux["stepAuxes"];
	KaiDict pm = layerParam["weight"];

	KBool bAcc = false;

	for (KInt n = timesteps-1; n >= 0; n--) {
		KaiDict stepAux = stepAuxes[n];

		if (output_seq) {
			if (cell == rec_cell::lstm && use_state) pMath->add_time_slice_on_dest(g_state, garr, n);
			else pMath->add_time_slice_on_dest(g_recurrent, garr, n);
		}

		KaiArray<KFloat> g_affine;

		if (cell == rec_cell::rnn) {
			KaiArray<KFloat> pre_act = FARRAY(stepAux["pre_act"]);
			KaiArray<KFloat> post_act = FARRAY(stepAux["post_act"]);

			g_affine = pMath->acivate_backprop(g_recurrent, pre_act, post_act, actfunc_id, pContext);

			KaiArray<KFloat> g_exp_input = KaiOptimizer::backprop_affine(g_affine, layerInfo, stepAux, pContext, pm, bAcc);
			g_recurrent = pMath->SplitExtendedInputGrad(g_exp_input, input_seq, g_x, n);
		}
		else if (cell == rec_cell::lstm) {
			KaiArray<KFloat> gates = FARRAY(stepAux["gates"]);
			KaiArray<KFloat> pre_state = FARRAY(stepAux["pre_state"]);
			KaiArray<KFloat> post_recur = FARRAY(stepAux["post_recur"]);

			KaiArray<KFloat> g_gates = pMath->lstm_proc_derv(g_state, g_recurrent, gates, pre_state, post_recur, use_state);
			g_affine = pMath->lstm_gates_derv(g_gates, gates);

			KaiArray<KFloat> g_exp_input = KaiOptimizer::backprop_affine(g_affine, layerInfo, stepAux, pContext, pm, bAcc);
			g_recurrent = pMath->SplitExtendedInputGrad(g_exp_input, input_seq, g_x, n);
		}
		else if (cell == rec_cell::gru) {
			KaiArray<KFloat> gates = FARRAY(stepAux["gates"]);
			KaiArray<KFloat> pre_recur = FARRAY(stepAux["pre_recur"]);
			KaiArray<KFloat> extra_affine = FARRAY(stepAux["extra_affine"]);
			KaiArray<KFloat> exp_inp_extra = FARRAY(stepAux["exp_inp_extra"]);

			KaiDict pm_extra = layerParam["extra"];
			KaiArray<KFloat> g_gates;
			KaiArray<KFloat> g_new_rec;

			KaiArray<KFloat> g_extra_affine = pMath->gru_proc_derv(g_gates, g_new_rec, g_recurrent, gates, pre_recur, extra_affine);
			KaiArray<KFloat> g_exp_inp_extra = KaiOptimizer::backprop_affine(g_extra_affine, layerInfo, stepAux, pContext, pm_extra, bAcc);
			pMath->gru_combine_extra_derv(g_exp_inp_extra, g_gates, g_recurrent, gates, exp_inp_extra);
			g_affine = pMath->sigmoid_derv_grad(g_gates, gates);
			KaiArray<KFloat> g_exp_input = KaiOptimizer::backprop_affine(g_affine, layerInfo, stepAux, pContext, pm, bAcc);
			g_recurrent = pMath->SplitExtendedInputGrad(g_exp_input, input_seq, g_x, n);
			pMath->add_on(g_exp_input, g_exp_inp_extra);
			pMath->add_on(g_recurrent, g_new_rec);
		}
		else throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

		bAcc = true;
	}

	return g_x;
}

KaiRnnLayer::KaiRnnLayer(KaiSession* pSession, KaiDict kwArgs) : KaiRecurrentLayer(pSession, kwArgs) {
}

KaiRnnLayer::~KaiRnnLayer() {
}

KaiList KaiRnnLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	return m_prepare_recurrent_exec_info(shape, pOptimizer, info, 1);
}

KaiArray<KFloat> KaiRnnLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	return ms_forward(rec_cell::rnn, xarr, layerInfo, layerParam, pContext, pAux);
}

KaiArray<KFloat> KaiRnnLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	return ms_backprop(rec_cell::rnn, garr, layerInfo, layerParam, layerAux, pContext);
}

KaiLstmLayer::KaiLstmLayer(KaiSession* pSession, KaiDict kwArgs) : KaiRecurrentLayer(pSession, kwArgs) {
}

KaiLstmLayer::~KaiLstmLayer() {
}

KaiList KaiLstmLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiList info_pair = m_prepare_recurrent_exec_info(shape, pOptimizer, info, 4);

	KFloat forget_bias = get_info_prop(info, "forget_bias", 0.5);

	if (forget_bias != 0) {
		KaiDict params = info_pair[1];
		KaiDict pm_weight = params["weight"];
		KaiDict pm_b = pm_weight["b"];
		KaiArray<KFloat> pm = FARRAY(pm_b["_pm_"]);

		KInt psize = pm.total_size();
		KFloat* p_pm = pm.data_ptr();
		for (KInt n = 0; n < psize; n += 4) p_pm[n] = forget_bias;
	}

	return info_pair;
}

#include "../math/kcudamath.h"
#include "../../src2020/core/host_math.h"

KaiArray<KFloat> KaiLstmLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	/*
	KaiArray<KFloat> yarr = ms_forward(rec_cell::lstm, xarr, layerInfo, layerParam, pContext, pAux);
	
	KaiCudaMath* pMath = (KaiCudaMath*)pContext->get_math();
	List aux = pMath->lstm_forward_test(xarr, yarr, layerParam["weight"]);

	KaiArray<KFloat> gyarr = yarr;
	KaiArray<KFloat> gxarr = ms_backprop(rec_cell::lstm, gyarr, layerInfo, layerParam, *pAux, pContext);
	
	pMath->lstm_backprop_test(gyarr, gxarr, layerParam["weight"], aux);

	return yarr;
	*/
	return ms_forward(rec_cell::lstm, xarr, layerInfo, layerParam, pContext, pAux);
}

KaiArray<KFloat> KaiLstmLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	return ms_backprop(rec_cell::lstm, garr, layerInfo, layerParam, layerAux, pContext);
}

KaiGruLayer::KaiGruLayer(KaiSession* pSession, KaiDict kwArgs) : KaiRecurrentLayer(pSession, kwArgs) {
}

KaiGruLayer::~KaiGruLayer() {
}

KaiList KaiGruLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	return m_prepare_recurrent_exec_info(shape, pOptimizer, info, 2, 1);
}

KaiArray<KFloat> KaiGruLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	return ms_forward(rec_cell::gru, xarr, layerInfo, layerParam, pContext, pAux);
}

KaiArray<KFloat> KaiGruLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	return ms_backprop(rec_cell::gru, garr, layerInfo, layerParam, layerAux, pContext);
}

KaiSelfAttentionLayer::KaiSelfAttentionLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiSelfAttentionLayer::~KaiSelfAttentionLayer() {
}

KaiList KaiSelfAttentionLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	info["input_shape"] = shape.copy();
	info["output_shape"] = shape.copy();

	KaiDict param;

	KInt head_cnt = get_set_info_prop(info, "multi_heads", 8);

	KInt vec_size = shape[1];			// word vector
	KInt vec_per_head = vec_size / head_cnt;

	if (vec_size % head_cnt != 0) KaiException(KERR_BAD_HEAD_CNT_FOR_SELF_ATTENTION);

	KString init_weight = kutil.seek_set_dict(info, "init_weight", "gaussian");
	KFloat init_std = kutil.seek_set_dict(info, "init_std", 0.030f);

	param["QKV"] = pOptimizer->createAffineParams(KaiShape{ vec_size, 3 * vec_size }, true, true, init_weight, init_std);
	param["O"] = pOptimizer->createAffineParams(KaiShape{ vec_size, vec_size }, true, true, init_weight, init_std);

	info["coef"] = 1.0f / sqrt(float(vec_per_head));
	info["keep_ratio"] = 1.0f - (KFloat)kutil.seek_set_dict(info, "dropout", 0.0f);

	return KaiList{ info, param };
}

KaiArray<KFloat> KaiSelfAttentionLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	KaiShape xshape = xarr.shape();

	KInt head_cnt = layerInfo["multi_heads"];		// heads for self attention

	KaiMath* pMath = pContext->get_math();

	if (debug_trace) {
		KaiArray<KFloat> qkv_w = ms_extract_weight(layerParam["QKV"]);
		KaiArray<KFloat> qkv_b = ms_extract_bias(layerParam["QKV"]);
		KaiArray<KFloat> o_w = ms_extract_weight(layerParam["O"]);
		KaiArray<KFloat> o_b = ms_extract_bias(layerParam["O"]);

		xarr.dump("xarr");
		qkv_w.dump("qkv_w");
		qkv_b.dump("qkv_b");
		o_w.dump("o_w");
		o_b.dump("o_b");
	}

	KaiArray<KFloat> qkv_in_one = KaiOptimizer::forward_affine(xarr, layerInfo, layerParam["QKV"], pContext, pAux);

	KaiList qkv_pieces = pMath->split_array(qkv_in_one, 3);

	KaiArray<KFloat> query = FARRAY(qkv_pieces[0]);
	KaiArray<KFloat> key   = FARRAY(qkv_pieces[1]);
	KaiArray<KFloat> value = FARRAY(qkv_pieces[2]);

	set_aux(pAux, "query", query.get_core());
	set_aux(pAux, "key", key.get_core());
	set_aux(pAux, "value", value.get_core());

	KaiArray<KFloat> att_score = pMath->multi_head_matmul_qk(query, key, head_cnt);
	
	KFloat coef = layerInfo["coef"];
	pMath->mul_on(att_score, coef);

	//set_aux(pAux, "att_score", att_score.get_core());

	KaiArray<KFloat> att_probs = pMath->softmax(att_score);

	if ((KFloat)layerInfo["keep_ratio"] < 1.0f) {
		// 2020버전에서는 계산된 확률값에 대해 드롭아웃 처리를 한다. 처리 위치가 잘못된 듯
		// 확률분포에 대한 드롭아웃 처리는 매우 어색, 내용 확인해볼 필요 있음
		// 현재 드롭아웃 마스크 생성에 hostmath의 랜덤 함수를 사용하므로 속도 크게 저하, 이 부분도 보완 후 드롭아웃 적용이 마땅할 듯
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	set_aux(pAux, "att_probs2", att_probs.get_core());

	KaiArray<KFloat> att_out = pMath->multi_head_matmul_pv(att_probs, value);

	set_aux(pAux, "att_out", att_out.get_core());
	set_aux(pAux, "att_score", att_score.get_core());
	set_aux(pAux, "att_probs", att_probs.get_core());

	KaiArray<KFloat> yarr = KaiOptimizer::forward_affine(att_out, layerInfo, layerParam["O"], pContext, pAux);

	if (debug_trace) {
		query.dump("query");
		key.dump("key");
		value.dump("value");
		att_score.dump("att_score");
		att_probs.dump("att_probs");
		att_out.dump("att_out");
		yarr.dump("yarr");
	}

	return yarr;
}

KaiArray<KFloat> KaiSelfAttentionLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	KaiMath* pMath = pContext->get_math();

	KaiDict pm_O = layerParam["O"];
	KaiArray<KFloat> g_att_out = KaiOptimizer::backprop_affine(gyarr, layerInfo, layerAux, pContext, pm_O);

	KaiArray<KFloat> value = FARRAY(layerAux["value"]);

	KaiArray<KFloat> att_out = FARRAY(layerAux["att_out"]);
	KaiArray<KFloat> att_probs = FARRAY(layerAux["att_probs"]);
	KaiArray<KFloat> att_score = FARRAY(layerAux["att_score"]);

	KInt head_cnt = layerInfo["multi_heads"];		// heads for self attention

	KaiArray<KFloat> g_att_probs = pMath->multi_head_matmul_pv_derv_p(g_att_out, value, head_cnt);
	KaiArray<KFloat> g_value = pMath->multi_head_matmul_pv_derv_v(g_att_out, att_probs);

	if ((KFloat)layerInfo["keep_ratio"] < 1.0f) {
		// 2020버전에서는 계산된 확률값에 대해 드롭아웃 처리를 한다.
		// 확률분포에 대한 드롭아웃 처리는 매우 어색, 내용 확인해볼 필요 있음
		// 현재 드롭아웃 마스크 생성에 hostmath의 랜덤 함수를 사용하므로 속도 크게 저하, 이 부분도 보완 후 드롭아웃 적용이 마땅할 듯
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	//KaiArray<KFloat> att_probs = FARRAY(layerAux["att_probs"]);

	KaiArray<KFloat> g_att_score = pMath->softmax_derv(g_att_probs, att_probs);

	KFloat coef = layerInfo["coef"];
	pMath->mul_on(g_att_score, coef);

	KaiArray<KFloat> query = FARRAY(layerAux["query"]);
	KaiArray<KFloat> key = FARRAY(layerAux["key"]);

	KaiArray<KFloat> g_query = pMath->multi_head_matmul_qk_derv_q(g_att_score, key, head_cnt);
	KaiArray<KFloat> g_key = pMath->multi_head_matmul_qk_derv_k(g_att_score, query, head_cnt);

	KaiArray<KFloat> g_qkv_in_one = pMath->merge_array(KaiList{ g_query.get_core(), g_key.get_core(), g_value.get_core() });

	KaiDict pm_QKV = layerParam["QKV"];
	KaiArray<KFloat> gxarr = KaiOptimizer::backprop_affine(g_qkv_in_one, layerInfo, layerAux, pContext, pm_QKV);

	if (debug_trace) {
		KaiArray<KFloat> qkv_w = ms_extract_weight(layerParam["QKV"]);
		KaiArray<KFloat> qkv_wg = ms_extract_weight_grad(layerParam["QKV"]);
		KaiArray<KFloat> qkv_b = ms_extract_bias(layerParam["QKV"]);
		KaiArray<KFloat> qkv_bg = ms_extract_bias_grad(layerParam["QKV"]);
		KaiArray<KFloat> o_w = ms_extract_weight(layerParam["O"]);
		KaiArray<KFloat> o_wg = ms_extract_weight_grad(layerParam["O"]);
		KaiArray<KFloat> o_b = ms_extract_bias(layerParam["O"]);
		KaiArray<KFloat> o_bg = ms_extract_bias_grad(layerParam["O"]);

		qkv_w.dump("qkv_w");
		qkv_wg.dump("qkv_wg");
		qkv_b.dump("qkv_b");
		qkv_bg.dump("qkv_bg");
		o_w.dump("o_w");
		o_wg.dump("o_wg");
		o_b.dump("o_b");
		o_bg.dump("o_bg");
		g_qkv_in_one.dump("g_qkv_in_one");

		att_score.dump("att_score");
		att_probs.dump("att_probs");
		att_out.dump("att_out");

		g_att_score.dump("g_att_score");
		g_att_probs.dump("g_att_probs");
		g_att_out.dump("g_att_out");

		query.dump("query");
		key.dump("key");
		value.dump("value");

		g_query.dump("g_query");
		g_key.dump("g_key");
		g_value.dump("g_value");

		g_qkv_in_one.dump("g_qkv_in_one");

		gyarr.dump("gyarr");
		gxarr.dump("gxarr");
	}

	return gxarr;
}

KaiExtractLayer::KaiExtractLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiExtractLayer::~KaiExtractLayer() {
}

KaiList KaiExtractLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	info["input_shape"] = shape.copy();

	KaiDict param;

	KInt axis = get_set_info_prop(info, "axis", 0);
	KInt index = get_set_info_prop(info, "index", 0);
	KInt count = get_set_info_prop(info, "count", 1);
	KBool reduce_axis = get_set_info_prop(info, "reduce_axis", true);

	if (axis < -1 || axis >= shape.size()) KaiException(KERR_BAD_AXIS_FOR_EXTRACT_LAYER);
	
	if (axis >= 0) {
		if (index < 0 || index + MAX(count, 1) > shape[axis]) KaiException(KERR_BAD_INDEX_RANGE_FOR_EXTRACT_LAYER);
		if (count == 1 && reduce_axis) shape = shape.remove_nth(axis);
		else shape = shape.replace_nth(axis, (count > 0) ? count : 1);
	}

	info["output_shape"] = shape.copy();

	return KaiList{ info, param };
}

KaiArray<KFloat> KaiExtractLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	KaiShape input_shape = layerInfo["input_shape"];

	set_aux(pAux, "xshape", xarr.shape());

	KInt axis = layerInfo["axis"];
	KInt index = layerInfo["index"];
	KInt count = layerInfo["count"];
	KBool reduce_axis = layerInfo["reduce_axis"];

	axis += xarr.dim() - input_shape.size();

	KaiArray<KFloat> yarr = pContext->get_math()->extract(xarr, axis, index, count, reduce_axis);

	if (debug_trace) {
		xarr.dump("xarr");
		yarr.dump("yarr");
	}

	return yarr;
}

KaiArray<KFloat> KaiExtractLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	KaiShape input_shape = layerInfo["input_shape"];

	KaiShape xshape = layerAux["xshape"];

	KInt axis = layerInfo["axis"];
	KInt index = layerInfo["index"];
	KInt count = layerInfo["count"];
	KBool reduce_axis = layerInfo["reduce_axis"];

	axis += xshape.size() - input_shape.size();

	KaiArray<KFloat> gxarr = pContext->get_math()->extract_derv(gyarr, xshape, axis, index, count, reduce_axis);

	if (debug_trace) {
		gyarr.dump("gyarr");
		gxarr.dump("gxarr");
	}

	return gxarr;
}

KaiEmbedLayer::KaiEmbedLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiEmbedLayer::~KaiEmbedLayer() {
}

KaiList KaiEmbedLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	info["input_shape"] = shape.copy();

	KaiList embed_info = info["embed_info"];
	
	//if (shape.size() != 2) throw KaiException(KERR_BAD_EMBED_INPUT_SHAPE);
	if (embed_info.size() != shape[-1]) throw KaiException(KERR_BAD_EMBED_INFO);

	KInt vec_size = get_set_info_prop(info, "vec_size", 128);

	KaiDict param;

	KString init_weight = kutil.seek_set_dict(info, "init_weight", "gaussian");
	KFloat init_std = kutil.seek_set_dict(info, "init_std", 0.030f);

	param["type"] = "multi_dic";
	param["embed_info"] = embed_info;

	for (auto& it : embed_info) {
		KaiDict term = it;
		param[term["name"]] = pOptimizer->createDicParams(KaiShape{term["size"], vec_size }, true, init_weight, init_std);
	}
	
	shape = shape.replace_end(vec_size);

	info["output_shape"] = shape.copy();

	return KaiList{ info, param };
}

KaiArray<KFloat> KaiEmbedLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	KaiShape input_shape = layerInfo["input_shape"];
	KaiShape output_shape = layerInfo["output_shape"];

	KaiArray<KInt> tokens = NARRAY(pack["#narr"]);

	pack.erase("#narr");

	set_aux(pAux, "tokens", tokens.shape());

	KaiList embed_info = layerInfo["embed_info"];
	KInt vec_size = layerInfo["vec_size"];

	set_aux(pAux, "tokens", tokens.get_core());
	set_aux(pAux, "embed_info", embed_info);

	KaiArray<KFloat> yarr = KaiOptimizer::forward_embed(tokens, embed_info, vec_size, layerParam, pContext, pAux);
	
	if (debug_trace) {
		for (auto& it : embed_info) {
			KaiDict info = it;
			KaiDict pm = layerParam[info["name"]];
			KaiDict w = pm["w"];
			KaiArray<KFloat> weight = FARRAY(w["_pm_"]);
			weight.dump("dic_" + (KString)info["name"]);
		}

		tokens.dump("tokens");
		yarr.dump("yarr");
	}

	return yarr;
}

KaiArray<KFloat> KaiEmbedLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	KaiArray<KInt> tokens = NARRAY(layerAux["tokens"]);

	KaiList embed_info = layerInfo["embed_info"];
	KInt vec_size = layerInfo["vec_size"];

	layerParam["_grad_"] = gyarr.get_core();
	layerParam["_tokens_"] = tokens.get_core();

	if (debug_trace) {
		tokens.dump("tokens");
		gyarr.dump("gyarr");

		for (auto& it : embed_info) {
			KaiDict info = it;
			KaiDict pm = layerParam[info["name"]];
			KaiDict w = pm["w"];
			KaiArray<KFloat> weight = FARRAY(w["_pm_"]);
			weight.dump("pm_" + (KString)info["name"]);
		}
	}

	return KaiArray<KFloat>();
}

KaiSelectLayer::KaiSelectLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiSelectLayer::~KaiSelectLayer() {
}

KaiList KaiSelectLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	info["input_shape"] = shape.copy();
	info["output_shape"] = shape.copy();

	KString selector = get_info_prop(info, "selector", "");
	
	if (selector == "") throw KaiException(KERR_SELCTOR_FOR_SELECT_LAYER_NOT_DEFINED);
	KaiDict param;

	return KaiList{ info, param };
}

KaiArray<KFloat> KaiSelectLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "forward");

	KaiShape vshape = layerInfo["input_shape"];
	KString selector = layerInfo["selector"];

	KaiArray<KInt> selector_arr = NARRAY(pack[selector]);

	set_aux(pAux, "xshape", xarr.shape());
	set_aux(pAux, "selector", selector_arr.get_core());

	KaiArray<KFloat> yarr = pContext->get_math()->select(xarr, selector_arr, vshape);

	if (debug_trace) {
		xarr.dump("xarr");
		yarr.dump("yarr");
		selector_arr.dump("selector1");
	}

	return yarr;
}

KaiArray<KFloat> KaiSelectLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(layerInfo, "backprop");

	KaiShape vshape = layerInfo["input_shape"];
	KaiShape xshape = layerAux["xshape"];

	KaiArray<KInt> selector_arr = NARRAY(layerAux["selector"]);

	KaiArray<KFloat> gxarr = pContext->get_math()->select_derv(gyarr, selector_arr, xshape, vshape);

	if (debug_trace) {
		gyarr.dump("gyarr");
		gxarr.dump("gxarr");
		selector_arr.dump("selector2");
	}

	return gxarr;
}

KaiExpandLayer::KaiExpandLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiExpandLayer::~KaiExpandLayer() {
}

KaiList KaiExpandLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["input_shape"] = shape.copy();

	if (shape.size() < 3) throw KaiException(KERR_TOO_SHORT_INPUT_SHAPE_FOR_CNN_LAYER);

	KaiShape ratio = get_2d_option(info, "ratio");
	info["ratio"] = ratio;

	shape[0] *= ratio[0];
	shape[1] *= ratio[1];

	info["output_shape"] = shape.copy();

	return KaiList{ info, KaiDict() };
}

KaiArray<KFloat> KaiExpandLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KaiShape ratio = layerInfo["ratio"];
	KaiArray<KFloat> yarr = pContext->get_math()->expand(xarr, ratio);
	return yarr;
}

KaiArray<KFloat> KaiExpandLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KaiShape ratio = layerInfo["ratio"];
	KaiArray<KFloat> gxarr = pContext->get_math()->expand_derv(gyarr, ratio);
	return gxarr;
}

KaiPassLayer::KaiPassLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiPassLayer::~KaiPassLayer() {
}

KaiList KaiPassLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	return KaiList{ info, KaiDict() };
}

KaiArray<KFloat> KaiPassLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	return xarr;
}

KaiArray<KFloat> KaiPassLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	return gyarr;
}

KaiStackLayer::KaiStackLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiStackLayer::~KaiStackLayer() {
}

KaiList KaiStackLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KBool ignore_input = kutil.seek_set_dict(info, "ignore_input", false);

	KaiShape tail_shape = info["tail_shape"];
	KaiShape stack_shape = tail_shape.insert_head(0);

	KInt tail_size = tail_shape.total_size();

	if (!ignore_input) {
		m_push_to_stack_shape(stack_shape, shape, tail_size, "(layer-input)");
	}

	KaiList collect = info["collect"];

	for (auto& it : collect) {
		KString sFieldName = it;
		if (pack.find(sFieldName) == pack.end()) throw KaiException(KERR_FIELD_FOR_STACK_LAYER_NOT_FOUND, sFieldName);
		KaiShape field_shape = pack[sFieldName];
		m_push_to_stack_shape(stack_shape, field_shape, tail_size, sFieldName);
	}

	info["output_shape"] = stack_shape;
	shape = stack_shape.copy();

	return { info, KaiDict() };
}

void KaiStackLayer::m_push_to_stack_shape(KaiShape& stack_shape, KaiShape add_shape, KInt tail_size, KString sFieldName) {
	if (add_shape.total_size() % tail_size != 0) throw KaiException(KERR_BAD_SHAPE_FIELD_FOR_STACK_LAYER, sFieldName);
	KInt nRow = add_shape.total_size() / tail_size;
	stack_shape[0] = stack_shape[0] + nRow;
}

KaiArray<KFloat> KaiStackLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KBool ignore_input = layerInfo["ignore_input"];

	KaiShape tail_shape = layerInfo["tail_shape"];
	KaiShape stack_shape = layerInfo["output_shape"];

	KInt mb_size = xarr.shape()[0];
	KInt tail_size = tail_shape.total_size();
	KInt nFrom = 0, nTo = stack_shape[0];

	KaiMath* pMath = pContext->get_math();

	KaiArray<KFloat> yarr = pMath->zeros(stack_shape.insert_head(mb_size));

	if (!ignore_input) {
		set_aux(pAux, "xshape", xarr.shape());
		nFrom = pMath->stack_on(yarr, xarr, tail_size, nFrom, nTo);
	}

	KaiList collect = layerInfo["collect"];

	for (auto& it : collect) {
		KString sFieldName = it;
		if (pack.find(sFieldName) == pack.end()) throw KaiException(KERR_FIELD_DATA_FOR_STACK_LAYER_NOT_FOUND, sFieldName);
		KaiArray<KFloat> farr = FARRAY(pack[sFieldName]);
		set_aux(pAux, sFieldName, farr.shape());
		nFrom = pMath->stack_on(yarr, farr, tail_size, nFrom, nTo);
	}

	if (nFrom != stack_shape[0]) throw KaiException(KERR_BAD_RESULT_FOR_STACK_LAYER);

	return yarr;
}

KaiArray<KFloat> KaiStackLayer::backprop(KaiArray<KFloat> gyarr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KBool ignore_input = layerInfo["ignore_input"];

	KaiShape tail_shape = layerInfo["tail_shape"];
	KaiShape stack_shape = layerInfo["output_shape"];

	KInt mb_size = gyarr.shape()[0];
	KInt tail_size = tail_shape.total_size();

	KInt nFrom = 0, nTo = stack_shape[0];

	KaiMath* pMath = pContext->get_math();

	KaiArray<KFloat> gxarr;

	if (!ignore_input) {
		KaiShape xshape = layerAux["xshape"];
		gxarr = pMath->stack_on_grad(gyarr, xshape, tail_size, nFrom, nTo);
	}

	KaiList collect = layerInfo["collect"];

	for (auto& it : collect) {
		KString sFieldName = it;
		KaiShape fshape = layerAux[sFieldName];
		KaiArray<KFloat> garr = pMath->stack_on_grad(gyarr, fshape, tail_size, nFrom, nTo);
		pack[sFieldName] = garr.get_core();
	}

	if (nFrom != stack_shape[0]) throw KaiException(KERR_BAD_RESULT_FOR_STACK_LAYER);

	return gxarr;
}

KaiSubnetLayer::KaiSubnetLayer(KaiSession* pSession, KaiDict kwArgs) : KaiLayer(pSession, kwArgs) {
}

KaiSubnetLayer::~KaiSubnetLayer() {
}

KaiList KaiSubnetLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["input_shape"] = shape.copy();

	KaiNetwork* pNetwork = (KaiNetwork*)(KHObject)info["branch"];

	KaiList info_pair = pNetwork->prepare_net_exec_info(shape, pOptimizer, info, pack);

	info["output_shape"] = shape.copy();

	info["subnet"] = info_pair[0];
	param["subnet"] = info_pair[1];
	info["netinfo"] = info_pair[2];

	//logger.Print("%s: %s => %s", info["builtin"].desc().c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}

KaiArray<KFloat> KaiSubnetLayer::forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack) {
	KaiList subnetAux;
	KaiList* pSubAux = pAux ? &subnetAux : NULL;

	KaiArray<KFloat> yarr = KaiNetwork::forward(xarr, layerInfo["netinfo"], layerInfo["subnet"], layerParam["subnet"], pContext, pSubAux, pack);
	set_aux(pAux, "subnet", subnetAux);

	return yarr;
}

KaiArray<KFloat> KaiSubnetLayer::backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack) {
	KaiList subnetAux = layerAux["subnet"];
	garr = KaiNetwork::backprop(garr, layerInfo["netinfo"], layerInfo["subnet"], layerParam["subnet"], pContext, subnetAux, pack);

	return garr;
}

KaiCustomLayer::KaiCustomLayer(KaiSession* pSession, KaiDict kwArgs) : KaiSubnetLayer(pSession, kwArgs) {
}

KaiCustomLayer::~KaiCustomLayer() {
}

KaiList KaiCustomLayer::m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) {
	KaiDict param;

	info["input_shape"] = shape.copy();

	KString sMacroName = info["macro_name"];
	KaiNetwork* pNetwork = m_pSession->get_macro(sMacroName);

	//logger.Print("%s: %s START", info["builtin"].desc().c_str(), sMacroName.c_str());

	KaiList info_pair = pNetwork->prepare_net_exec_info(shape, pOptimizer, info, pack);

	info["output_shape"] = shape.copy();

	info["subnet"] = info_pair[0];
	param["subnet"] = info_pair[1];
	info["netinfo"] = info_pair[2];

	//logger.Print("%s : %s END %s => %s", info["builtin"].desc().c_str(), sMacroName.c_str(), info["input_shape"].desc().c_str(), info["output_shape"].desc().c_str());
	return KaiList{ info, param };
}
