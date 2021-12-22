/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "knetwork.h"
#include "klayer.h"
#include "koptimizer.h"
#include "../session/session.h"
#include "../exec/exec_context.h"
#include "../math/kmath.h"
#include "../utils/kutil.h"

//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif


int KaiNetwork::ms_checkCode = 31861992;

KStrList KaiNetwork::ms_builtin = { "mlp", "serial", "parallel", "add", "add_serial", "add_parallel", "pass" };

KaiNetwork::KaiNetwork(KaiSession* pSession, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::network, Ken_object_type::network, kwArgs) {
	m_checkCode = ms_checkCode;
}

KaiNetwork::KaiNetwork(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiComponent(pSession, Ken_component_type::network, Ken_object_type::network, pLib, pComponentInfo) {
	m_checkCode = ms_checkCode;
}

/*
KaiNetwork::KaiNetwork(KaiNetwork* pSrc, KaiDict kwArgs) : KaiComponent(pSrc->m_pSession, Ken_component_type::network, Ken_object_type::network, pSrc->m_propDict) {
	KaiList layers = m_propDict["layers"];
	KaiList clones;

	for (auto& it : layers) {
		KaiLayer* pOrgLayer = (KaiLayer*)(KHObject)it;
		KaiLayer* pCloneLayer = new KaiLayer(pOrgLayer, kwArgs);
		clones.push_back(pCloneLayer);
	}

	m_propDict["layers"] = clones;

	m_checkCode = ms_checkCode;
}
*/

KaiNetwork::~KaiNetwork() {
	m_checkCode = 0;
}

KaiNetwork* KaiNetwork::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Network");

	KaiNetwork* pNetwork = (KaiNetwork*)hObject;

	if (pNetwork->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Network");
	if (pNetwork->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Network");

	return pNetwork;
}

KaiNetwork* KaiNetwork::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Network");

	KaiNetwork* pNetwork = (KaiNetwork*)hObject;

	if (pNetwork->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Network");

	return pNetwork;
}

void KaiNetwork::GetBuiltinNames(KStrList* pslNames) {
	for (auto it = ms_builtin.begin(); it != ms_builtin.end(); it++) {
		pslNames->push_back(*it);
	}
}

KaiNetwork* KaiNetwork::CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs) {
	KaiNetwork* pInstance = NULL;

	if (sBuiltin == "") sBuiltin = "serial";

	if (sBuiltin == "mlp") pInstance = new KaiMlpNetwork(pSession, kwArgs);
	else if (sBuiltin == "serial") pInstance = new KaiSerialNetwork(pSession, kwArgs);
	else if (sBuiltin == "parallel") pInstance = new KaiParallelNetwork(pSession, kwArgs);
	else if (sBuiltin == "add_parallel") pInstance = new KaiAddParallelNetwork(pSession, kwArgs);
	else if (sBuiltin == "add_serial") pInstance = new KaiAddSerialNetwork(pSession, kwArgs);
	else if (sBuiltin == "add") {
		KString subnet = get_info_prop(kwArgs, "subnet", "parallel");
		if (subnet == "parallel") pInstance = new KaiAddParallelNetwork(pSession, kwArgs);
		else if (subnet == "serial") pInstance = new KaiAddSerialNetwork(pSession, kwArgs);
	}
	else if (sBuiltin == "pass") pInstance = new KaiPassNetwork(pSession, kwArgs);

	if (!pInstance) throw KaiException(KERR_UNKNOWN_NETWORK_SUBCLASS_NAME);

	pInstance->m_propDict["builtin"] = sBuiltin;
	pInstance->m_propDict["desc"] = pInstance->desc();

	return pInstance;
}

KaiNetwork* KaiNetwork::copy() {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiNetwork::append_layer(KaiLayer* pLayer) {
	//pLayer->set_property("output", false);
	KaiList layers = get_property("layers", KaiList());
	layers.push_back(pLayer);
	set_property("layers", layers);
	bind(pLayer, "subnet", false, false);
	m_bDirty = true;
}

void KaiNetwork::append_named_layer(KString sLayerName, KaiDict kwArgs) {
	KaiLayer* pLayer = KaiLayer::CreateInstance(m_pSession, sLayerName, kwArgs);
	append_layer(pLayer);
}

void KaiNetwork::append_custom_layer(KString sLayerName, KaiDict kwArgs) {
	KaiLayer* pLayer = KaiLayer::CreateInstance(m_pSession, "custom", kwArgs);
	pLayer->set_property("macro_name", sLayerName);
	append_layer(pLayer);
	/*
	KaiNetwork* pMacro = m_pSession->get_macro(sLayerName);
	KaiNetwork* pClone = new KaiNetwork(pMacro, kwArgs);
	KaiList layers = get_property("layers", KaiList());
	layers.push_back(pClone);
	set_property("layers", layers);
	bind(pClone, "subnet", false, false);
	*/
}

void KaiNetwork::append_subnet(KaiNetwork* pSubnet) {
	KaiLayer* pLayer = KaiLayer::CreateInstance(m_pSession, "subnet", KaiDict());
	pLayer->set_property("branch", pSubnet);
	append_layer(pLayer);
}

KaiList KaiNetwork::prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack) {
	KString sGetName;
	KString sSetName;

	if (call_info.find("get") != call_info.end() && (KString)call_info["get"] != "") {
		sGetName = (KString)call_info["get"];
		if (pack.find(sGetName) == pack.end()) throw KaiException(KERR_FIELD_FOR_SET_ATTRIBUTE_NOT_FOUND, sGetName);
		shape = pack[sGetName];
		call_info.erase("get");
	}

	if (call_info.find("set") != call_info.end() && (KString)call_info["set"] != "") {
		sSetName = (KString)call_info["set"];
		call_info.erase("set");
	}

	KaiList info_list = m_prepare_net_exec_info(shape, pOptimizer, call_info, pack);

	if (sGetName != "" || sSetName != "") {
		KaiDict netInfo = info_list[2];
		if (sGetName != "") netInfo["get"] = sGetName;
		if (sSetName != "") {
			netInfo["set"] = sSetName;
			pack[sSetName] = shape.copy();
		}
	}

	KaiDict dump = info_list[2]; // pack과 netinfo에 set/get 반영여부 확인할 것

	return info_list;
}

KaiList KaiNetwork::m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR);
}

KaiArray<KFloat> KaiNetwork::forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack) {
	if (netinfo.find("get") != netinfo.end()) {
		KString getName = netinfo["get"];
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

	KaiArray<KFloat> yarr;

	KString sBuiltin = netinfo["builtin"];

	if (sBuiltin == "serial") yarr = KaiSerialNetwork::forward(xarr, netinfo, layerInfos, layerParams, pContext, pAux, pack);
	else if (sBuiltin == "parallel") yarr = KaiParallelNetwork::forward(xarr, netinfo, layerInfos, layerParams, pContext, pAux, pack);
	else if (sBuiltin == "add_parallel") yarr = KaiAddParallelNetwork::forward(xarr, netinfo, layerInfos, layerParams, pContext, pAux, pack);
	else if (sBuiltin == "add_serial") yarr = KaiAddSerialNetwork::forward(xarr, netinfo, layerInfos, layerParams, pContext, pAux, pack);
	else if (sBuiltin == "add") {
		throw KaiException(KERR_UNKNOWN_SUBNET_TYPE, sBuiltin);
		//return KaiAddNetwork::forward(xarr, netinfo, layerInfos, layerParams, pContext, pAux, pack);
	}
	else if (sBuiltin == "pass") yarr = KaiPassNetwork::forward(xarr, netinfo, layerInfos, layerParams, pContext, pAux, pack);
	else throw KaiException(KERR_UNKNOWN_SUBNET_TYPE, sBuiltin);

	if (netinfo.find("set") != netinfo.end()) {
		KString setName = netinfo["set"];
		if (setName != "") {
			pack[setName] = yarr.get_core();
			//printf("set(%s)\n", setName.c_str());
		}
	}

	return yarr;
}

KaiArray<KFloat> KaiNetwork::backprop(KaiArray<KFloat> gyarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack) {
	if (netinfo.find("set") != netinfo.end()) {
		KString setName = netinfo["set"];
		if (setName != "" && pack.find(setName) != pack.end()) {
			KaiArray<KFloat> grad = FARRAY(pack[setName]);
			if (gyarr.is_empty()) gyarr = grad;
			else gyarr = pContext->get_math()->add(gyarr, grad);
			//printf("set(%s)\n", setName.c_str());
		}
	}

	KaiArray<KFloat> gxarr;

	KString sBuiltin = netinfo["builtin"];

	if (sBuiltin == "serial") gxarr = KaiSerialNetwork::backprop(gyarr, netinfo, layerInfos, layerParams, pContext, aux, pack);
	else if (sBuiltin == "parallel") gxarr = KaiParallelNetwork::backprop(gyarr, netinfo, layerInfos, layerParams, pContext, aux, pack);
	else if (sBuiltin == "add_parallel") gxarr = KaiAddParallelNetwork::backprop(gyarr, netinfo, layerInfos, layerParams, pContext, aux, pack);
	else if (sBuiltin == "add_serial") gxarr = KaiAddSerialNetwork::backprop(gyarr, netinfo, layerInfos, layerParams, pContext, aux, pack);
	else if (sBuiltin == "add") {
		throw KaiException(KERR_UNKNOWN_SUBNET_TYPE, sBuiltin);
		//return KaiAddNetwork::backprop(garr, netinfo, layerInfos, layerParams, pContext, aux, pack);
	}
	else if (sBuiltin == "pass") gxarr = KaiPassNetwork::backprop(gyarr, netinfo, layerInfos, layerParams, pContext, aux, pack);
	else throw KaiException(KERR_UNKNOWN_SUBNET_TYPE, sBuiltin);

	if (netinfo.find("get") != netinfo.end()) {
		KString getName = netinfo["get"];
		if (getName != "") {
			pack[getName] = gxarr.get_core();
			gxarr = KaiArray<KFloat>();
			//printf("get(%s)\n", getName.c_str());
		}
	}

	return gxarr;
}

KInt KaiNetwork::get_layer_count() {
	KaiList layers = get_property("layers");
	return layers.size();
}

KaiLayer* KaiNetwork::get_nth_layer(KInt nth) {
	KaiList layers = get_property("layers", KaiList());
	if (nth < 0 || nth >= (KInt)layers.size()) throw KaiException(KERR_LAYER_INDEX_OUT_OF_RANGE);
	return (KaiLayer*)(KHObject)layers[nth];
}

KString KaiNetwork::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf, 128, "<KaiComponent Network %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

void KaiNetwork::dump_property(KString sKey, KString sTitle) {
	if (sKey == "structure") {
		logger.Print("Structure of Network-%s", desc().c_str());
		KaiList layers = get_property("layers", KaiList());
		for (auto& it : layers) {
			KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
			/*
		for (auto& it : m_layers) {
			KaiLayer* pLayer = it;
		*/
			KString sBuiltin = pLayer->get_property("builtin");
			KString sInShape = pLayer->get_property("input_shape").desc();
			KString sOutShape = pLayer->get_property("output_shape").desc();
			logger.Print("    %s : %s => %s", sBuiltin.c_str(), sInShape.c_str(), sOutShape.c_str());
		}
	}
	else {
		KaiComponent::dump_property(sKey, sTitle);
	}
}

/*
KaiDict KaiNetwork::copy_properties() {
	KaiDict dict = KaiComponent::copy_properties();

	KaiShape xshape = get_property("input_shape");
	KaiShape yshape = get_property("output_shape");

	KaiShape hshape = xshape;

	KaiList layers = get_property("layers", KaiList());
	KaiList layerInfos;

	for (auto& it : layers) {
		KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
		KaiDict layerInfo = pLayer->set_shape_flow(hshape);
		layerInfos.push_back(layerInfo);
	}

	dict["layerinfos"] = layerInfos;

	return dict;
}
*/

/*
void KaiNetwork::check_shape_flow() {
	KaiShape xshape = get_property("input_shape");
	KaiShape yshape = get_property("output_shape");

	KaiShape hshape = xshape;

	KaiList layers = get_property("layers", KaiList());

	for (auto& it : layers) {
		KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
		hshape = pLayer->set_shape_flow(hshape);
	}
}
*/

KaiMlpNetwork::KaiMlpNetwork(KaiSession* pSession, KaiDict kwArgs) : KaiSerialNetwork(pSession, kwArgs) {
	KaiList widths = kutil.seek_dict(m_propDict, "widths", KaiList());
	KaiList layers;

	KaiShape output_shape = m_propDict["output_shape"];
	KInt ysize = output_shape[0];

	KaiDict layerInfo = m_propDict;

	layerInfo.erase("name");
	layerInfo.erase("widths");

	KBool add_batchnormal = get_property("madd_batchnormal", false);

	KaiLayer* pPrevLayer = NULL;

	for (auto& it : widths) {
		layerInfo["width"] = (KInt)it;
		layerInfo["output"] = false;

		KaiLayer* pLayer = (KaiLayer*) KaiLayer::CreateInstance(m_pSession, "dense", layerInfo);
		layers.push_back(pLayer);
		bind(pLayer, "subnet", false, false);
		pPrevLayer = pLayer;

		if (add_batchnormal) {
			KaiLayer* pBnLayer = (KaiLayer*)KaiLayer::CreateInstance(m_pSession, "batchnormal", layerInfo);
			layers.push_back(pBnLayer);
			bind(pBnLayer, "subnet", false, false);
			pPrevLayer = pBnLayer;
		}
	}

	set_property("layers", layers);
}

KaiMlpNetwork::~KaiMlpNetwork() {
}

KaiSerialNetwork::KaiSerialNetwork(KaiSession* pSession, KaiDict kwArgs) : KaiNetwork(pSession, kwArgs) {
}

KaiSerialNetwork::~KaiSerialNetwork() {
}

KaiList KaiSerialNetwork::m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack) {
	KaiDict info = resolve_copy_properties(call_info);
	KaiDict netInfo;

	KaiList layers = info["layers"]; // get_property(KStrList{ "network", "layers" });

	netInfo["input_shape"] = shape.copy();
	if (info.find("debug_trace") != info.end()) netInfo["debug_trace"] = info["debug_trace"];

	KaiList layerInfos;
	KaiList layerParams;

	for (auto& it : layers) {
		KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
		KaiDict resolved_info = pLayer->resolve_copy_properties(call_info);
		KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);
		if (nRepeat > 1) {
			//KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);
			resolved_info.erase("repeat");
		}
		for (KInt n = 0; n < nRepeat; n++) {
			KaiList info_pair = pLayer->prepare_exec_info(shape, pOptimizer, resolved_info, pack);
			//KaiDict layerParam = pOptimizer->createLayerParams(layerInfo, info);
			layerInfos.push_back(info_pair[0]);
			layerParams.push_back(info_pair[1]);
		}
	}


	if (info.find("set") != info.end() && (KString)info["set"] != "") {
		KString sFieldName = info["set"];
		pack[sFieldName] = shape.copy();
	}

	netInfo["output_shape"] = shape.copy();
	netInfo["builtin"] = "serial";

	return KaiList{ layerInfos, layerParams, netInfo };
}

KBool KaiNetwork::ms_get_debug_trace(KaiDict netinfo, KString phase) {
	if (netinfo.find("debug_trace") == netinfo.end()) return false;
	
	KaiDict debug_trace = netinfo["debug_trace"];

	if (debug_trace.find("phase") != debug_trace.end()) {
		KaiList phases = debug_trace["phase"];
		if (!phases.find_string(phase)) return false;
	}

	if (debug_trace.find("targets") != debug_trace.end()) {
		KaiList target = debug_trace["targets"];
		if (target.find_string((KString)netinfo["builtin"])) return true;
	}

	if (debug_trace.find("checked") != debug_trace.end()) {
		KaiList checked = debug_trace["checked"];
		if (checked.find_string((KString)netinfo["builtin"])) return false;
		return true;
	}

	return false;
}

KaiArray<KFloat> KaiSerialNetwork::forward(KaiArray<KFloat> harr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "forward");

	/*
	static KInt nth_seed = 0;
	KInt nth = nth_seed++;
	netinfo["nth"] = nth;
	KString subName;
	if (netinfo.find("subnet") != netinfo.end()) {
		subName = " " + (KString)((KaiDict)layerInfo["netinfo"])["builtin"];
		if (layerInfo.find("macro_name") != layerInfo.end()) subName += ":" + (KString)layerInfo["macro_name"];
	}

	printf("layer %s%s(%lld): forward input shape %s\n", ((KString)layerInfo["builtin"]).c_str(), subName.c_str(), nth, xarr.shape().desc().c_str());
	*/

	if (netinfo.find("get") != netinfo.end()) {
		KString getName = netinfo["get"];
		if (getName != "") {
			if (pack.find(getName) == pack.end()) throw KaiException(KERR_SET_ATTR_VALUE_NOT_FOUND, getName);
			KaiValue value = pack[getName];
			if (value.is_farray()) {
				harr = FARRAY(value);
			}
			else if (value.is_narray()) {
				pack["#narr"] = value;
				harr = KaiArray<KFloat>();
			}
			else throw KaiException(KERR_BAD_TYPE_DATA_FOR_SET_ATTR_VALUE, getName);
		}
	}

	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux;

		KaiDict* pLayerAux = pAux ? &layerAux : NULL;

		harr = KaiLayer::forward(harr, layerInfos[n], layerParams[n], pContext, pLayerAux, pack);

		if (pAux) pAux->push_back(layerAux);
	}

	return harr;
}

KaiArray<KFloat> KaiSerialNetwork::backprop(KaiArray<KFloat> garr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "backprop");

	for (KInt n = (KInt)layerInfos.size() - 1; n >= 0; n--) {
		garr = KaiLayer::backprop(garr, layerInfos[n], layerParams[n], aux[n], pContext, pack);
	}

	return garr;
}

KaiParallelNetwork::KaiParallelNetwork(KaiSession* pSession, KaiDict kwArgs) : KaiNetwork(pSession, kwArgs) {
}

KaiParallelNetwork::~KaiParallelNetwork() {
}

KaiList KaiParallelNetwork::m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack) {
	KaiDict info = resolve_copy_properties(call_info);
	KaiDict netInfo;

	KaiList layers = info["layers"]; // get_property(KStrList{ "network", "layers" });

	netInfo["input_shape"] = shape.copy();

	if (info.find("debug_trace") != info.end()) netInfo["debug_trace"] = info["debug_trace"];

	KaiList layerInfos;
	KaiList layerParams;

	KaiShape yshape;

	for (auto& it : layers) {
		KaiShape bshape = shape.copy();

		KaiList info_pair;

		KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
		KaiDict resolved_info = pLayer->resolve_copy_properties(call_info);
		KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);

		if (nRepeat <= 0) {
			throw KaiException(KERR_NO_BRANDCH_IN_PARALLEL_NET);
		}
		else if (nRepeat == 1) {
			info_pair = pLayer->prepare_exec_info(bshape, pOptimizer, resolved_info, pack);
		}
		else {
			resolved_info.erase("repeat");
			info_pair = KaiLayer::wrap_repeated_in_parallel_branch(pLayer, nRepeat, bshape, pOptimizer, resolved_info, pack);
		}

		if (yshape.size() == 0) yshape = bshape;
		else {
			if (yshape.total_size() / yshape[-1] != bshape.total_size() / bshape[-1]) throw KaiException(KERR_MISMATCHED_SHAPE_IN_PARALLEL_BRANCH);
			yshape[-1] += bshape[-1];
		}
		//KaiDict layerParam = pOptimizer->createLayerParams(layerInfo, info);
		layerInfos.push_back(info_pair[0]);
		layerParams.push_back(info_pair[1]);
	}

	shape = yshape;

	netInfo["output_shape"] = shape.copy();
	netInfo["builtin"] = "parallel";

	return KaiList{ layerInfos, layerParams, netInfo };
}

KaiArray<KFloat> KaiParallelNetwork::forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KaiShape ishape = netinfo["input_shape"];
	KaiShape oshape = netinfo["output_shape"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_tail(ishape, oshape);

	KaiArray<KFloat> yarr = pContext->get_math()->zeros(yshape);

	KInt nChnPos = 0;

	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux;
		KaiDict* pLayerAux = pAux ? &layerAux : NULL;

		KaiArray<KFloat> barr = KaiLayer::forward(xarr, layerInfos[n], layerParams[n], pContext, pLayerAux, pack);
		//barr.dump("barr");

		pContext->get_math()->CopyIntoSlice(yarr, barr, nChnPos);

		KString sShape = barr.shape().desc();

		if (pAux) {
			layerAux["branch_chn"] = barr.shape()[-1];
			pAux->push_back(layerAux);
		}
	}

	assert(nChnPos == yshape[-1]);
	//yarr.dump("yarr");

	return yarr;
}

KaiArray<KFloat> KaiParallelNetwork::backprop(KaiArray<KFloat> gyarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	//KaiShape ishape = netinfo["input_shape"];
	//KaiShape oshape = netinfo["output_shape"];

	//KaiShape yshape = gyarr.shape();
	//KaiShape xshape = yshape.replace_tail(oshape, ishape);

	KaiArray<KFloat> gxarr;
	//KaiArray<KFloat> gxarr = pContext->get_math()->zeros(xshape);

	KInt nChnPos = 0;

	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux = aux[n];
		KInt nChnCnt = layerAux["branch_chn"];
		KaiArray<KFloat> bgyarr = pContext->get_math()->CopyFromSlice(gyarr, nChnPos, nChnCnt);
		//bgyarr.dump("bgyarr");
		KaiArray<KFloat> bgxarr = KaiLayer::backprop(bgyarr, layerInfos[n], layerParams[n], layerAux, pContext, pack);
		//bgxarr.dump("bgxarr");
		//gxarr.dump("gxarr2");
		if (gxarr.is_empty()) gxarr = bgxarr;
		else pContext->get_math()->add_on(gxarr, bgxarr);
		//gxarr.dump("gxarr3");
	}

	return gxarr;
}

KaiAddParallelNetwork::KaiAddParallelNetwork(KaiSession* pSession, KaiDict kwArgs) : KaiNetwork(pSession, kwArgs) {
}

KaiAddParallelNetwork::~KaiAddParallelNetwork() {
}

KaiList KaiAddParallelNetwork::m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack) {
	KaiDict info = resolve_copy_properties(call_info);
	KaiDict netInfo;

	KaiShape xshape = shape.copy();

	if (info.find("stride") != info.end()) {
		if (shape.size() < 3) throw KaiException(KERR_TOO_SHORT_INPUT_SHAPE_FOR_CNN_LAYER);

		KaiShape stride = KaiLayer::get_2d_option(info, "stride", KaiShape{ 1,1 });

		KInt yh = xshape[-3], yw = xshape[-2];
		KInt sh = stride[0], sw = stride[1];

		shape[-3] = (yh + sh / 2) / sh;
		shape[-2] = (yw + sw / 2) / sw;

		netInfo["stride"] = stride;
	}

	netInfo["input_shape"] = xshape;
	if (info.find("debug_trace") != info.end()) netInfo["debug_trace"] = info["debug_trace"];

	KaiList layers = info["layers"]; // get_property(KStrList{ "network", "layers" });

	KaiList layerInfos;
	KaiList layerParams;

	KaiList bshapes{shape.copy()};
	KInt max_chn = xshape[-1];

	for (auto& it : layers) {
		KaiShape bshape = xshape.copy();

		KaiList info_pair;

		KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
		KaiDict resolved_info = pLayer->resolve_copy_properties(call_info);
		KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);

		if (nRepeat <= 0) {
			throw KaiException(KERR_NO_BRANCH_IN_ADD_NET);
		}
		else if (nRepeat == 1) {
			info_pair = pLayer->prepare_exec_info(bshape, pOptimizer, resolved_info, pack);
		}
		else {
			resolved_info.erase("repeat");
			info_pair = KaiLayer::wrap_repeated_in_parallel_branch(pLayer, nRepeat, bshape, pOptimizer, resolved_info, pack);
		}

		if (bshape[-1] > max_chn) max_chn = bshape[-1];

		bshapes.push_back(bshape);

		layerInfos.push_back(info_pair[0]);
		layerParams.push_back(info_pair[1]);
	}

	shape[-1] = max_chn;

	for (auto& it: bshapes) {
		KaiShape bshape = it;
		if (bshape.replace_end(max_chn) != shape) throw KaiException(KERR_BRANCH_WITH_BAD_SHAPE_IN_ADD_NET);
		if ((max_chn % bshape[-1]) != 0) throw KaiException(KERR_BRANCH_WITH_BAD_CHANNELS_IN_ADD_NET);
	}

	netInfo["output_shape"] = shape.copy();
	netInfo["builtin"] = "add_parallel";

	return KaiList{ layerInfos, layerParams, netInfo };
}

KaiArray<KFloat> KaiAddParallelNetwork::forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "forward");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	//KaiException(KERR_UNIMPEMENTED_YET, "subnet 처리");

	KaiShape ishape = netinfo["input_shape"];
	KaiShape oshape = netinfo["output_shape"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_tail(ishape, oshape);

	KaiArray<KFloat> yarr = pContext->get_math()->zeros(yshape);
	
	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux;
		KaiDict* pLayerAux = pAux ? &layerAux : NULL;

		KaiArray<KFloat> barr = KaiLayer::forward(xarr, layerInfos[n], layerParams[n], pContext, pLayerAux, pack);

		pContext->get_math()->residual_add(yarr, barr);

		if (pAux) {
			layerAux["branch_chn"] = barr.shape()[-1];
			pAux->push_back(layerAux);
		}
	}

	if (netinfo.find("stride") != netinfo.end()) {
		KaiShape stride = netinfo["stride"]; // KaiLayer::get_2d_option(netinfo, "stride", KaiShape{ 1,1 });

		if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
			xarr = pContext->get_math()->stride(xarr, stride);
		}
	}

	if (pAux) {
		pAux->push_back(xarr.shape()[-1]);
	}

	pContext->get_math()->residual_add(yarr, xarr);

	return yarr;
}

KaiArray<KFloat> KaiAddParallelNetwork::backprop(KaiArray<KFloat> gyarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "backprop");

	if (debug_trace) {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	//KaiException(KERR_UNIMPEMENTED_YET, "subnet 처리");

	KaiShape ishape = netinfo["input_shape"];
	KaiShape oshape = netinfo["output_shape"];

	KaiShape yshape = gyarr.shape();
	KaiShape xshape = yshape.replace_tail(oshape, ishape);

	KaiArray<KFloat> gxarr = pContext->get_math()->zeros(xshape);

	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux = aux[n];
		KInt bchn = layerAux["branch_chn"];
		KaiArray<KFloat> gbyarr = pContext->get_math()->residual_add_derv(gyarr, bchn);
		KaiArray<KFloat> gbxarr = KaiLayer::backprop(gbyarr, layerInfos[n], layerParams[n], layerAux, pContext, pack);
		pContext->get_math()->add_on(gxarr, gbxarr);
	}

	KInt xchn = *(aux.end()-1);

	KaiArray<KFloat> grxarr = pContext->get_math()->residual_add_derv(gyarr, xchn);

	if (netinfo.find("stride") != netinfo.end()) {
		KaiShape stride = netinfo["stride"]; // KaiLayer::get_2d_option(netinfo, "stride", KaiShape{ 1,1 });

		if ((KInt)stride[0] != 1 || (KInt)stride[1] != 1) {
			grxarr = pContext->get_math()->stride_derv(gyarr, stride, xshape);
		}
	}

	pContext->get_math()->add_on(gxarr, grxarr);

	return gxarr;
	/*
	throw KaiException(KERR_UNIMPEMENTED_YET);
	KaiShape ishape = netinfo["input_shape"];
	KaiShape oshape = netinfo["output_shape"];


	KaiArray<KFloat> gxarr = pContext->get_math()->zeros(xshape);
	//gxarr.dump("gxarr1");

	KInt nChnPos = 0;

	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux = aux[n];
		KInt nChnCnt = layerAux["branch_chn"];
		KaiArray<KFloat> bgyarr = pContext->get_math()->CopyFromSlice(gyarr, nChnPos, nChnCnt);
		//bgyarr.dump("bgyarr");
		KaiArray<KFloat> bgxarr = KaiLayer::backprop(bgyarr, layerInfos[n], layerParams[n], layerAux, pContext);
		//bgxarr.dump("bgxarr");
		//gxarr.dump("gxarr2");
		pContext->get_math()->add_on(gxarr, bgxarr);
		//gxarr.dump("gxarr3");
	}

	return gxarr;
	*/
}

KaiAddSerialNetwork::KaiAddSerialNetwork(KaiSession* pSession, KaiDict kwArgs) : KaiNetwork(pSession, kwArgs) {
}

KaiAddSerialNetwork::~KaiAddSerialNetwork() {
}

KaiList KaiAddSerialNetwork::m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack) {
	KaiDict info = resolve_copy_properties(call_info);
	KaiDict netInfo;

	KaiShape xshape = shape.copy();

	if (info.find("stride") != info.end()) {
		if (shape.size() < 3) throw KaiException(KERR_TOO_SHORT_INPUT_SHAPE_FOR_CNN_LAYER);

		KaiShape stride = KaiLayer::get_2d_option(info, "stride", KaiShape{ 1,1 });

		KInt yh = xshape[-3], yw = xshape[-2];
		KInt sh = stride[0], sw = stride[1];

		shape[-3] = (yh + sh / 2) / sh;
		shape[-2] = (yw + sw / 2) / sw;

		netInfo["stride"] = stride;
	}

	netInfo["input_shape"] = xshape;

	if (info.find("debug_trace") != info.end()) netInfo["debug_trace"] = info["debug_trace"];

	KaiList layers = info["layers"]; // get_property(KStrList{ "network", "layers" });

	KaiList layerInfos;
	KaiList layerParams;

	KaiShape hshape = xshape.copy();

	for (auto& it : layers) {
		KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
		KaiDict resolved_info = pLayer->resolve_copy_properties(call_info);
		KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);
		if (nRepeat > 1) {
			//KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);
			resolved_info.erase("repeat");
		}
		for (KInt n = 0; n < nRepeat; n++) {
			KaiList info_pair = pLayer->prepare_exec_info(hshape, pOptimizer, resolved_info, pack);
			//KaiDict layerParam = pOptimizer->createLayerParams(layerInfo, info);
			layerInfos.push_back(info_pair[0]);
			layerParams.push_back(info_pair[1]);
		}
	}

	if (hshape != xshape) throw KaiException(KERR_UNMATCHING_SHAPE_ON_ADD_SERIAL);

	netInfo["output_shape"] = shape.copy();
	netInfo["builtin"] = "add_serial";

	return KaiList{ layerInfos, layerParams, netInfo };
}

KaiArray<KFloat> KaiAddSerialNetwork::forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "forward");

	if (debug_trace) {
		xarr.dump("xarr");
	}

	//printf(">> subnet %s: forward input shape %s\n", ((KString)netinfo["builtin"]).c_str(), xarr.shape().desc().c_str());

	KaiShape ishape = netinfo["input_shape"];
	KaiShape oshape = netinfo["output_shape"];

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_tail(ishape, oshape);

	KaiArray<KFloat> harr = xarr;

	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux;

		KaiDict* pLayerAux = pAux ? &layerAux : NULL;

		harr = KaiLayer::forward(harr, layerInfos[n], layerParams[n], pContext, pLayerAux, pack);

		if (pAux) pAux->push_back(layerAux);
	}

	if (debug_trace) {
		harr.dump("harr");
	}

	pContext->get_math()->residual_add(harr, xarr);

	if (debug_trace) {
		harr.dump("yarr");
	}

	//printf("<< subnet %s: forward output shape %s\n", ((KString)netinfo["builtin"]).c_str(), harr.shape().desc().c_str());

	return harr;
}

KaiArray<KFloat> KaiAddSerialNetwork::backprop(KaiArray<KFloat> gyarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "backprop");

	KaiArray<KFloat> gharr = gyarr;

	for (KInt n = (KInt)layerInfos.size() - 1; n >= 0; n--) {
		gharr = KaiLayer::backprop(gharr, layerInfos[n], layerParams[n], aux[n], pContext, pack);
	}

	KaiArray<KFloat> gxarr = pContext->get_math()->add(gyarr, gharr);

	if (debug_trace) {
		gyarr.dump("gyarr");
		gharr.dump("gharr");
		gxarr.dump("gxarr");
	}

	return gxarr;
}

KaiPassNetwork::KaiPassNetwork(KaiSession* pSession, KaiDict kwArgs) : KaiNetwork(pSession, kwArgs) {
}

KaiPassNetwork::~KaiPassNetwork() {
}

KaiList KaiPassNetwork::m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack) {
	KaiDict info = resolve_copy_properties(call_info);
	KaiDict netInfo;

	KaiList layers = info["layers"];

	netInfo["input_shape"] = shape.copy();
	netInfo["output_shape"] = shape.copy();

	if (info.find("debug_trace") != info.end()) netInfo["debug_trace"] = info["debug_trace"];

	KaiList layerInfos;
	KaiList layerParams;

	KaiShape bshape = shape.copy();

	for (auto& it : layers) {
		KaiLayer* pLayer = (KaiLayer*)(KHObject)it;
		KaiDict resolved_info = pLayer->resolve_copy_properties(call_info);
		KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);
		if (nRepeat > 1) {
			KInt nRepeat = KaiLayer::get_info_prop(resolved_info, "repeat", 1);
			resolved_info.erase("repeat");
		}
		for (KInt n = 0; n < nRepeat; n++) {
			KaiList info_pair = pLayer->prepare_exec_info(bshape, pOptimizer, resolved_info, pack);
			//KaiDict layerParam = pOptimizer->createLayerParams(layerInfo, info);
			layerInfos.push_back(info_pair[0]);
			layerParams.push_back(info_pair[1]);
		}
	}

	netInfo["builtin"] = "pass";

	return KaiList{ layerInfos, layerParams, netInfo };
}

KaiArray<KFloat> KaiPassNetwork::forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "forward");

	if (debug_trace) {
		xarr.dump("xarr");
	}

	KaiArray<KFloat> harr = xarr;

	for (KInt n = 0; n < (KInt)layerInfos.size(); n++) {
		KaiDict layerAux;

		KaiDict* pLayerAux = pAux ? &layerAux : NULL;

		harr = KaiLayer::forward(harr, layerInfos[n], layerParams[n], pContext, pLayerAux, pack);

		if (pAux) pAux->push_back(layerAux);
	}

	if (debug_trace) {
		harr.dump("harr_post");
		xarr.dump("xarr_post");
	}

	return xarr;
}

KaiArray<KFloat> KaiPassNetwork::backprop(KaiArray<KFloat> gyarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack) {
	KBool debug_trace = ms_get_debug_trace(netinfo, "backprop");

	KaiArray<KFloat> gharr;

	for (KInt n = (KInt)layerInfos.size() - 1; n >= 0; n--) {
		gharr = KaiLayer::backprop(gharr, layerInfos[n], layerParams[n], aux[n], pContext, pack);
	}

	if (gharr.shape() != gyarr.shape()) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

	KaiArray<KFloat> gxarr = pContext->get_math()->add(gyarr, gharr);

	if (debug_trace) {
		gyarr.dump("gyarr");
		gharr.dump("gharr");
		gxarr.dump("gxarr");
	}

	return gxarr;
}
