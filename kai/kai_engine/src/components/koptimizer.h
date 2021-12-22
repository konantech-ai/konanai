/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "component.h"
#include "../include/kai_api.h"
#include "../math/karray.h"

class KaiDataset;
class KaiLayer;
class KaiExecContext;
class KaiMath;

class KaiOptimizer : public KaiComponent {
public:
	KaiOptimizer(KaiSession* pSession, KaiDict kwArgs);
	KaiOptimizer(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	virtual ~KaiOptimizer();

	Ken_object_type get_type() { return Ken_object_type::optimizer; }

	static KaiOptimizer* HandleToPointer(KHObject hObject);
	static KaiOptimizer* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiOptimizer* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	KaiDict createAffineParams(KaiShape wshape, KBool bTrain, KBool use_bias, KString init_weight, KFloat init_std = 0.03f);
	KaiDict createDicParams(KaiShape wshape, KBool bTrain, KString init_weight, KFloat init_std = 0.03f);

	static KaiArray<KFloat> forward_affine(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict pm, KaiExecContext* pContext, KaiDict* pAux);
	static KaiArray<KFloat> backprop_affine(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pm, KBool bAcc=false);

	static KaiArray<KFloat> forward_conv(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict pm, KaiExecContext* pContext, KaiDict* pAux);
	static KaiArray<KFloat> backprop_conv(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pm);

	static KaiArray<KFloat> forward_embed(KaiArray<KInt> xarr, KaiList embedInfo, KInt vec_size, KaiDict params, KaiExecContext* pContext, KaiDict* pAux);

	static KaiArray<KFloat> backprop_rescale(KaiArray<KFloat> garr, KaiArray<KFloat> norm_x, KaiDict pm_scale, KaiDict pm_shift, KaiExecContext* pContext);

	static void update_parameter(KaiList layerParams, KaiExecContext* pContext);

	KFloat eval_grad_norm(KaiList layerParams, KaiMath* pMath);

	void apply_grad_clipping(KaiList layerParams, KFloat ratio, KaiMath* pMath);
	void update_subnet_params(KaiList layerParams, KaiMath* pMath);

protected:
	KaiArray<KFloat> m_init_weight(KaiShape shape, KString init_type, KFloat init_std);

	virtual void m_alloc_weight_extend(KaiDict& pm_w, KaiShape param_shape, KBool bTrain) {}
	virtual void m_alloc_bias_extend(KaiDict& pm_b, KaiShape param_shape, KBool bTrain) {}
	virtual void m_alloc_dic_extend(KaiDict& pm_b, KaiShape param_shape, KBool bTrain) {}
	
	virtual void m_update_weight(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	//virtual void m_update_kernel(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	virtual void m_update_bias(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	virtual void m_update_multi_dic(KaiMath* pMath, KaiDict layerParam);

	KaiArray<KFloat> m_apply_decay(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad);

	static void ms_set_aux(KaiDict* pAux, KString sKey, KaiValue value) {
		if (pAux) {
			KaiDict& dict = *pAux;
			dict[sKey] = value;
		}
	}

	KString desc();

protected:
	static int ms_checkCode;
	static KStrList ms_builtin;
};

class KaiSgdOptimizer : public KaiOptimizer {
public:
	KaiSgdOptimizer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiSgdOptimizer();

protected:
	virtual void m_update_weight(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	//virtual void m_update_kernel(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	virtual void m_update_bias(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	virtual void m_update_multi_dic(KaiMath* pMath, KaiDict layerParam);
};

class KaiAdamOptimizer : public KaiOptimizer {
public:
	KaiAdamOptimizer(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiAdamOptimizer();

	void m_alloc_weight_extend(KaiDict& pm_w, KaiShape param_shape, KBool bTrain);
	void m_alloc_bias_extend(KaiDict& pm_b, KaiShape param_shape, KBool bTrain);
	void m_alloc_dic_extend(KaiDict& pm_w, KaiShape param_shape, KBool bTrain);

	virtual void m_update_weight(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	//virtual void m_update_kernel(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	virtual void m_update_bias(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo);
	virtual void m_update_multi_dic(KaiMath* pMath, KaiDict layerParam);
};
