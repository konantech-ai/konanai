/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include"component.h"
#include"../math/karray.h"

class KaiDataset;
class KaiLayer;
class KaiOptimizer;

class KaiExecContext;
class KaiParameters;

class KaiNetwork : public KaiComponent {
public:
	KaiNetwork(KaiSession* pSession, KaiDict kwArgs);
	KaiNetwork(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	//KaiNetwork(KaiNetwork* pSrc, KaiDict kwArgs);

	virtual ~KaiNetwork();

	KaiNetwork* copy();

	Ken_object_type get_type() { return Ken_object_type::network; }

	static KaiNetwork* HandleToPointer(KHObject hObject);
	static KaiNetwork* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiNetwork* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	virtual void append_layer(KaiLayer* pLayer);
	virtual void append_named_layer(KString sLayerName, KaiDict kwArgs);

	virtual void append_custom_layer(KString sLayerName, KaiDict kwArgs);
	virtual void append_subnet(KaiNetwork* pSubnet);

	virtual KInt get_layer_count();
	virtual KaiLayer* get_nth_layer(KInt nth);

	virtual void dump_property(KString sKey, KString sTitle);

	KaiList prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack);
	virtual KaiList m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack);

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack);

	//virtual KaiDict copy_properties();
	//virtual void check_shape_flow();

	KString desc();

protected:
	static int ms_checkCode;
	static KStrList ms_builtin;

	static KBool ms_get_debug_trace(KaiDict netinfo, KString phase);
};

class KaiSerialNetwork : public KaiNetwork {
public:
	KaiSerialNetwork(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiSerialNetwork();
	virtual KaiList m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack);

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack);
};

class KaiMlpNetwork : public KaiSerialNetwork {
public:
	KaiMlpNetwork(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiMlpNetwork();
};

class KaiParallelNetwork : public KaiNetwork {
public:
	KaiParallelNetwork(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiParallelNetwork();
	virtual KaiList m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack);

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack);
};

class KaiAddParallelNetwork : public KaiNetwork {
public:
	KaiAddParallelNetwork(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiAddParallelNetwork();
	virtual KaiList m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack);

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack);
};

class KaiAddSerialNetwork : public KaiNetwork {
public:
	KaiAddSerialNetwork(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiAddSerialNetwork();
	virtual KaiList m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack);

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack);
};

class KaiPassNetwork : public KaiNetwork {
public:
	KaiPassNetwork(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiPassNetwork();
	virtual KaiList m_prepare_net_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict call_info, KaiDict& pack);

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict netinfo, KaiList layerInfos, KaiList layerParams, KaiExecContext* pContext, KaiList aux, KaiDict& pack);
};
