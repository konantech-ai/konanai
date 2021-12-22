/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "component.h"
#include "../math/karray.h"

class KaiNetwork;
class KaiMath;
class KaiOptimizer;

enum class rec_cell { rnn, lstm, gru };

class KaiLayer : public KaiComponent {
public:
	KaiLayer(KaiSession* pSession, KaiDict kwArgs);
	KaiLayer(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	//KaiLayer(KaiLayer* pSrc, KaiDict kwArgs);
	virtual ~KaiLayer();

	Ken_object_type get_type() { return Ken_object_type::layer; }

	static KaiLayer* HandleToPointer(KHObject hObject);
	static KaiLayer* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiLayer* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	virtual KaiList prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
	static KaiList wrap_repeated_in_parallel_branch(KaiLayer* pLayer, KInt nRepeat, KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

	KString desc();

	static KaiShape get_2d_option(KaiDict info, KString sKey, KaiShape sDef = KaiShape{ 1,1 });

	static KaiArray<KFloat> get_param(KaiDict layerParam, KString sParamName, KString sSubName = "w");

	static void set_aux(KaiDict* pAux, KString sKey, KaiValue value);


protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack) = 0;

	KInt m_to_actfunc_id(KString funcname);

	static KaiArray<KFloat> ms_extract_weight(KaiDict pm);
	static KaiArray<KFloat> ms_extract_bias(KaiDict pm);
	static KaiArray<KFloat> ms_extract_weight_grad(KaiDict pm);
	static KaiArray<KFloat> ms_extract_bias_grad(KaiDict pm);

	static KBool ms_get_debug_trace(KaiDict layerInfo, KString phase);

protected:
	static int ms_checkCode;
	static KStrList ms_builtin;

	KaiShape m_input_shape;
	KaiShape m_output_shape;
};

class KaiDenseLayer : public KaiLayer {
public:
	KaiDenseLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiDenseLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiConvLayer : public KaiLayer {
public:
	KaiConvLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiConvLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiMaxLayer : public KaiLayer {
public:
	KaiMaxLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiMaxLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiAvgLayer : public KaiLayer {
public:
	KaiAvgLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiAvgLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiGlobalAvgLayer : public KaiLayer {
public:
	KaiGlobalAvgLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiGlobalAvgLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiBatchNormalLayer : public KaiLayer {
public:
	KaiBatchNormalLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiBatchNormalLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiActivateLayer : public KaiLayer {
public:
	KaiActivateLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiActivateLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiDropoutLayer : public KaiLayer {
public:
	KaiDropoutLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiDropoutLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiRecurrentLayer : public KaiLayer {
public:
	KaiRecurrentLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiRecurrentLayer();

protected:
	static KaiArray<KFloat> ms_forward(rec_cell cell, KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux);
	static KaiArray<KFloat> ms_backprop(rec_cell cell, KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext);

	KaiList m_prepare_recurrent_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KInt nBlocks, KInt nExtra=0);
};

class KaiRnnLayer : public KaiRecurrentLayer {
public:
	KaiRnnLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiRnnLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiLstmLayer : public KaiRecurrentLayer {
public:
	KaiLstmLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiLstmLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiGruLayer : public KaiRecurrentLayer {
public:
	KaiGruLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiGruLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiSelfAttentionLayer : public KaiLayer {
public:
	KaiSelfAttentionLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiSelfAttentionLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiExtractLayer : public KaiLayer {
public:
	KaiExtractLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiExtractLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiEmbedLayer : public KaiLayer {
public:
	KaiEmbedLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiEmbedLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiSelectLayer : public KaiLayer {
public:
	KaiSelectLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiSelectLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiExpandLayer : public KaiLayer {
public:
	KaiExpandLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiExpandLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiPassLayer : public KaiLayer {
public:
	KaiPassLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiPassLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiStackLayer : public KaiLayer {
public:
	KaiStackLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiStackLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
	void m_push_to_stack_shape(KaiShape& stack_shape, KaiShape add_shape, KInt tail_size, KString sFieldName);
};

class KaiSubnetLayer : public KaiLayer {
public:
	KaiSubnetLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiSubnetLayer();

	static KaiArray<KFloat> forward(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict layerParam, KaiExecContext* pContext, KaiDict* pAux, KaiDict& pack);
	static KaiArray<KFloat> backprop(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerParam, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pack);

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};

class KaiCustomLayer : public KaiSubnetLayer {
public:
	KaiCustomLayer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiCustomLayer();

protected:
	virtual KaiList m_prepare_exec_info(KaiShape& shape, KaiOptimizer* pOptimizer, KaiDict info, KaiDict& pack);
};
