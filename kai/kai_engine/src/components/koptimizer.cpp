/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "koptimizer.h"
#include "kdataset.h"
#include "klayer.h"
#include "../exec/exec_context.h"
#include "../session/session.h"
#include "../math/kmath.h"
#include "../utils/kutil.h"
#include "../nightly/nightly_utils.h"

//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif


int KaiOptimizer::ms_checkCode = 78126013;

KStrList KaiOptimizer::ms_builtin = { "sgd", "adam" };

KaiOptimizer::KaiOptimizer(KaiSession* pSession, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::optimizer, Ken_object_type::optimizer, kwArgs) {
	m_checkCode = ms_checkCode;
}

KaiOptimizer::KaiOptimizer(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiComponent(pSession, Ken_component_type::optimizer, Ken_object_type::optimizer, pLib, pComponentInfo) {
	m_checkCode = ms_checkCode;
}

KaiOptimizer::~KaiOptimizer() {
	m_checkCode = 0;
}

KaiOptimizer* KaiOptimizer::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Optimizer");

	KaiOptimizer* pOptimizer = (KaiOptimizer*)hObject;

	if (pOptimizer->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Optimizer");
	if (pOptimizer->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Optimizer");

	return pOptimizer;
}

KaiOptimizer* KaiOptimizer::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Optimizer");

	KaiOptimizer* pOptimizer = (KaiOptimizer*)hObject;

	if (pOptimizer->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Optimizer");

	return pOptimizer;
}

KaiOptimizer* KaiOptimizer::CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs) {
	KaiOptimizer* pInstance = NULL;

	if (sBuiltin == "sgd") pInstance = new KaiSgdOptimizer(pSession, kwArgs);
	else if (sBuiltin == "adam") pInstance = new KaiAdamOptimizer(pSession, kwArgs);
	else if (sBuiltin == "") pInstance = new KaiOptimizer(pSession, kwArgs);

	if (!pInstance) throw KaiException(KERR_UNKNOWN_DATALOADER_SUBCLASS_NAME);

	pInstance->m_propDict["builtin"] = sBuiltin;
	pInstance->m_propDict["desc"] = pInstance->desc();

	return pInstance;
}

void KaiOptimizer::GetBuiltinNames(KStrList* pslNames) {
	for (auto it = ms_builtin.begin(); it != ms_builtin.end(); it++) {
		pslNames->push_back(*it);
	}
}


KString KaiOptimizer::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf, 128, "<KaiComponent Optimizer %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

KaiDict KaiOptimizer::createAffineParams(KaiShape wshape, KBool bTrain, KBool use_bias, KString init_weight, KFloat init_std) {
	KaiDict pm;
	KaiDict pm_w;

	pm["type"] = "affine";
	pm["train"] = bTrain;

	if (wshape.total_size() == 0) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

	//KString init_weight = m_seek_property("init_weight", "gaussian", param, networkInfo); // 레이어 정보, 네트워크 정보, 옵티마이저 정보의 순서로 키값을 찾음
	KaiArray<KFloat> arr = m_init_weight(wshape, init_weight, init_std);
	pm_w["_pm_"] = arr.get_core();
	m_alloc_weight_extend(pm_w, wshape, bTrain);
	pm["w"] = pm_w;

	if (use_bias) {
		KaiDict pm_b;
		KaiShape bshape = KaiShape{ wshape[-1] };
		KaiArray<KFloat> arr = m_init_weight(bshape, init_weight, init_std);
		//KaiArray<KFloat> arr = KaiMath::GetHostMath()->zeros(bshape);
		pm_b["_pm_"] = arr.get_core();
		m_alloc_bias_extend(pm_b, bshape, bTrain);
		pm["b"] = pm_b;
	}

	return pm;
}

KaiDict KaiOptimizer::createDicParams(KaiShape wshape, KBool bTrain, KString init_weight, KFloat init_std) {
	KaiDict pm;
	KaiDict pm_w;

	pm["type"] = "dic";
	pm["train"] = bTrain;

	if (wshape.total_size() == 0) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

	KaiArray<KFloat> arr = m_init_weight(wshape, init_weight, init_std);
	pm_w["_pm_"] = arr.get_core();
	m_alloc_dic_extend(pm_w, wshape, bTrain);
	pm["w"] = pm_w;

	return pm;
}

KaiArray<KFloat> KaiOptimizer::m_init_weight(KaiShape shape, KString init_type, KFloat init_std) {
	KaiMath* pMath = KaiMath::GetHostMath();

	if (init_type == "zeros") return pMath->zeros(shape);
	else if (init_type == "ones") return pMath->ones(shape);
	else if (init_type == "uniform") return pMath->random_uniform(shape);
	else if (init_type == "gaussian" || init_type == "gauss" || init_type == "normal") {
		//KFloat init_std = m_seek_property("init_std", 0.030f, layerInfo, networkInfo); // 레이어 정보, 네트워크 정보, 옵티마이저 정보의 순서로 키값을 찾음
		return pMath->random_normal(shape, 0, init_std, false);
	}
	else if (init_type == "adaptive_gaussian" || init_type == "adaptive_gauss" || init_type == "adaptive_normal") {
		return pMath->random_normal(shape, 0, init_std, true);
	}
	else if (init_type == "Xavier") {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}
	else {
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}
}

KaiArray<KFloat> KaiOptimizer::forward_affine(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict pm, KaiExecContext* pContext, KaiDict* pAux) {
	KaiMath* pMath = pContext->get_math();

	KaiDict pm_w = pm["w"];

	KaiArray<KFloat> weight = FARRAY(pm_w["_pm_"]);

	//weight.dump("weight", true);
	KaiArray<KFloat> affine = pMath->matmul(xarr, weight);
	//affine.dump("matmul output", true);

	if (pm.find("b") != pm.end()) {
		KaiDict pm_b = pm["b"];
		KaiArray<KFloat> bias = FARRAY(pm_b["_pm_"]);
		//bias.dump("bias", true);
		affine = pMath->add_bias(affine, bias);
		//affine.dump("bias output", true);
	}

	ms_set_aux(pAux, "x", xarr.get_core());

	return affine;
}

KaiArray<KFloat> KaiOptimizer::forward_conv(KaiArray<KFloat> xarr, KaiDict layerInfo, KaiDict pm, KaiExecContext* pContext, KaiDict* pAux) {
	KaiMath* pMath = pContext->get_math();

	KaiDict pm_k = pm["w"];

	KaiArray<KFloat> kernel = FARRAY(pm_k["_pm_"]);
	//kernel.dump("kernel", true);
	KaiArray<KFloat> conv = pMath->convolution(xarr, kernel);
	//conv.dump("conv output", true);

	if (pm.find("b") != pm.end()) {
		KaiDict pm_b = pm["b"];
		KaiArray<KFloat> bias = FARRAY(pm_b["_pm_"]);
		//bias.dump("bias", true);
		conv = pMath->add_bias(conv, bias);
		//conv.dump("bias output", true);
	}

	ms_set_aux(pAux, "x", xarr.get_core());

	return conv;
}

KaiArray<KFloat> KaiOptimizer::backprop_affine(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pm, KBool bAcc) {
	KaiMath* pMath = pContext->get_math();

	KaiDict pm_w = pm["w"];

	KaiArray<KFloat> w = FARRAY(pm_w["_pm_"]);
	KaiArray<KFloat> x = FARRAY(layerAux["x"]);

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_OPTIMIZER_BACKPROP_AFFINE)
	{
		printf("[TRACE]  %s(%u) {\n", __FUNCTION__, __LINE__);
		printf("    garr.shape() : %s\n", garr.shape().desc().c_str());
		print_kdict(layerInfo, "", 4, "layerInfo");
		print_kdict(layerAux, "", 4, "layerAux");
		print_kdict(pm, "", 4, "pm");

		printf("x.shape()    : %s\n", x.shape().desc().c_str());
		printf("w.shape()    : %s\n", w.shape().desc().c_str());
		
		printf("}\n\n");
	}
#endif

	KaiArray<KFloat> x_transpose = pMath->transpose(x);
	KaiArray<KFloat> w_transpose = pMath->transpose(w);

	KaiArray<KFloat> grad_x = pMath->matmul(garr, w_transpose);
	KaiArray<KFloat> grad_w = pMath->matmul(x_transpose, garr);

	if (bAcc) {
		KaiArray<KFloat> pm_gw = FARRAY(pm_w["_grad_"]);
		pMath->add_on(pm_gw, grad_w);
	}
	else {
		pm_w["_grad_"] = grad_w.get_core();
	}

	if (pm.find("b") != pm.end()) {
		KaiArray<KFloat> grad_b = pMath->sum_on_column(garr);
		KaiDict pm_b = pm["b"];

		if (bAcc) {
			KaiArray<KFloat> pm_gb = FARRAY(pm_b["_grad_"]);
			pMath->add_on(pm_gb, grad_b);
		}
		else {
			pm_b["_grad_"] = grad_b.get_core();
		}
	}

	return grad_x;
}

KaiArray<KFloat> KaiOptimizer::backprop_conv(KaiArray<KFloat> garr, KaiDict layerInfo, KaiDict layerAux, KaiExecContext* pContext, KaiDict& pm) {
	KaiMath* pMath = pContext->get_math();

	KaiDict pm_k = pm["w"];

	KaiArray<KFloat> k= FARRAY(pm_k["_pm_"]);
	KaiArray<KFloat> x = FARRAY(layerAux["x"]);

	KaiArray<KFloat> grad_x = pMath->convolution_derv_x(garr, k);
	KaiArray<KFloat> grad_k = pMath->convolution_derv_k(garr, x, k.shape());

	pm_k["_grad_"] = grad_k.get_core();

	if (pm.find("b") != pm.end()) {
		KaiArray<KFloat> grad_b = pMath->sum_on_column(garr);
		KaiDict pm_b = pm["b"];
		pm_b["_grad_"] = grad_b.get_core();
	}

	return grad_x;
}

KaiArray<KFloat> KaiOptimizer::forward_embed(KaiArray<KInt> tokens, KaiList embedInfo, KInt vec_size, KaiDict params, KaiExecContext* pContext, KaiDict* pAux) {
	KaiShape xshape = tokens.shape();
	KaiShape yshape = xshape.replace_end(vec_size);

	if (xshape[-1] != embedInfo.size()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KaiMath* pMath = pContext->get_math();

	KaiArray<KFloat> yarr = pMath->zeros(yshape);

	KInt axis = 0;

	for (auto& it : embedInfo) {
		KaiDict embedItem = it;
		KaiDict pm = params[(KString)embedItem["name"]];
		KaiDict pm_w = pm["w"];
		KaiArray<KFloat> word_vecs = FARRAY(pm_w["_pm_"]);
		pMath->add_embed_dict(yarr, tokens, word_vecs, axis++);
	}

	return yarr;
}

KaiArray<KFloat> KaiOptimizer::backprop_rescale(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiDict pm_scale, KaiDict pm_shift, KaiExecContext* pContext) {
	KaiMath* pMath = pContext->get_math();

	KaiDict pm_scale_w = pm_scale["w"];
	KaiDict pm_shift_w = pm_shift["w"];

	KaiArray<KFloat> scale = FARRAY(pm_scale_w["_pm_"]);
	KaiArray<KFloat> shift = FARRAY(pm_shift_w["_pm_"]);

	KaiArray<KFloat> grad_scale;
	KaiArray<KFloat> grad_shift;

	pMath->rescale_derv_pm(garr, x, &grad_scale, &grad_shift);

	pm_scale_w["_grad_"] = grad_scale.get_core();
	pm_shift_w["_grad_"] = grad_shift.get_core();

	KaiArray<KFloat> grad_x = pMath->rescale_derv_x(garr, scale);

	return grad_x;

	/*
	KaiArray<KFloat> x = FARRAY(layerAux["x"]);

	KaiArray<KFloat> x_transpose = pMath->transpose(x);
	KaiArray<KFloat> w_transpose = pMath->transpose(w);

	KaiArray<KFloat> grad_x = pMath->matmul(garr, w_transpose);
	KaiArray<KFloat> grad_w = pMath->matmul(x_transpose, garr);
	//grad_x.dump("affine grad_x", true);
	//grad_w.dump("affine grad_w", true);

	pm_w["_grad_"] = grad_w.get_core();

	if (pm.find("b") != pm.end()) {
		KaiArray<KFloat> grad_b = pMath->sum_on_column(garr);
		KaiDict pm_b = pm["b"];
		//grad_b.dump("affine grad_b", true);
		pm_b["_grad_"] = grad_b.get_core();
	}

	return grad_x;
	*/
	throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	if (m_rescale) {
		float* cuda_x = cuda.attach(m_aux["norm_x"], "norm_x:BN(backprop)");

		float* cuda_scale = m_fetch_weight_ptr(m_param["scale"]);

		float* cuda_gscale = cuda.alloc_float_mem(bshape, "gscale::BN(backprop)");
		float* cuda_gshift = cuda.alloc_float_mem(bshape, "shift::BN(backprop)");

		cu_call(ker_bn_rescale_derv_pm, bsize, (bsize, cuda_gscale, cuda_gshift, cuda_gh, cuda_x, hsize));
		cu_call(ker_bn_rescale_derv_x, hsize, (hsize, cuda_gh, cuda_scale, bsize));

		Array<float> G_scale = cuda.detach(cuda_gscale);
		Array<float> G_shift = cuda.detach(cuda_gshift);

		m_update_weight(m_param["scale"], G_scale);
		m_update_weight(m_param["shift"], G_shift);
	}
	*/
}

void KaiOptimizer::update_parameter(KaiList layerParams, KaiExecContext* pContext) {
	KaiMath* pMath = pContext->get_math();
	KFloat clip_grad = pContext->get_property("clip_grad", 0.0f);
	KBool trace_grad_norm = pContext->get_property("trace_grad_norm", false);

	KaiDict kwArgs = pContext->get_component_property("optimizer");
	KString sBuiltin = kwArgs["builtin"];

	KaiOptimizer* pOptimizer = CreateInstance(NULL, sBuiltin, kwArgs); 

	if (clip_grad > 0 || trace_grad_norm) {
		KFloat grad_norm = pOptimizer->eval_grad_norm(layerParams, pMath);
		if (trace_grad_norm) logger.Print("grad_norm = %f", grad_norm);
		if (clip_grad > 0 && grad_norm > clip_grad) {
			KFloat ratio = ::sqrt(clip_grad / grad_norm);
			pOptimizer->apply_grad_clipping(layerParams, ratio, pMath);
		}
	}

	pOptimizer->update_subnet_params(layerParams, pMath);
}

void KaiOptimizer::update_subnet_params(KaiList layerParams, KaiMath* pMath) {
	for (auto& it1 : layerParams) {
		KaiDict layerParam = it1;

		if (layerParam.find("type") != layerParam.end()) {
			if ((KString)layerParam["type"] == "multi_dic") {
				m_update_multi_dic(pMath, layerParam);
				continue;
			}
			else {
				throw KaiException(KERR_INTERNAL_LOGIC_ERROR);
			}
		}

		for (auto& it2 : layerParam) {
			if (it2.second.type() == Ken_value_type::list) {
				update_subnet_params(it2.second, pMath);
				continue;
			}

			KaiDict paramInfo = it2.second;

			if (!(KBool)paramInfo["train"]) continue;

			for (auto& it3 : paramInfo) {
				KString sKey = it3.first;
				if (it3.second.type() != Ken_value_type::dict) continue;

				KaiDict param = it3.second;
				if (param.find("_grad_") == param.end()) continue;

				KaiArray<KFloat> pm = FARRAY(param["_pm_"]);
				KaiArray<KFloat> grad = FARRAY(param["_grad_"]);

				if (sKey == "w") m_update_weight(pMath, pm, grad, param);
				//else if (sKey == "k") pOptimizer->m_update_kernel(pMath, pm, grad, param);
				else if (sKey == "b") m_update_bias(pMath, pm, grad, param);
				else throw KaiException(KERR_UNSUPPORTED_PARAM_TYPE_FOUND);

				param.erase("_grad_");
			}
		}
	}
}

KFloat KaiOptimizer::eval_grad_norm(KaiList layerParams, KaiMath* pMath) {
	KFloat grad_norm = 0;

	for (auto& it1 : layerParams) {
		KaiDict layerParam = it1;

		if (layerParam.find("type") != layerParam.end()) {
			if ((KString)layerParam["type"] == "multi_dic") {
				KaiArray<KFloat> grad = FARRAY(layerParam["_grad_"]);
				grad_norm += pMath->fetch(pMath->sum(pMath->square(grad)));
				continue;
			}
			else {
				throw KaiException(KERR_INTERNAL_LOGIC_ERROR);
			}
		}

		for (auto& it2 : layerParam) {
			if (it2.second.type() == Ken_value_type::list) {
				grad_norm += eval_grad_norm(it2.second, pMath);
				continue;
			}

			KaiDict paramInfo = it2.second;

			if (!(KBool)paramInfo["train"]) continue;

			for (auto& it3 : paramInfo) {
				KString sKey = it3.first;
				if (it3.second.type() != Ken_value_type::dict) continue;

				KaiDict param = it3.second;
				if (param.find("_grad_") != param.end()) {
					KaiArray<KFloat> grad = FARRAY(param["_grad_"]);
					grad_norm += pMath->fetch(pMath->sum(pMath->square(grad)));
				}
			}
		}
	}

	return grad_norm;
}

void KaiOptimizer::apply_grad_clipping(KaiList layerParams, KFloat ratio, KaiMath* pMath) {
	for (auto& it1 : layerParams) {
		KaiDict layerParam = it1;
		for (auto& it2 : layerParam) {
			KaiDict layerParam = it1;

			if (layerParam.find("type") != layerParam.end()) {
				if ((KString)layerParam["type"] == "multi_dic") {
					KaiArray<KFloat> grad = FARRAY(layerParam["_grad_"]);
					pMath->mul_on(grad, ratio);
					continue;
				}
				else {
					throw KaiException(KERR_INTERNAL_LOGIC_ERROR);
				}
			}

			if (it2.second.type() == Ken_value_type::list) {
				apply_grad_clipping(it2.second, ratio, pMath);
				continue;
			}

			KaiDict paramInfo = it2.second;

			if (!(KBool)paramInfo["train"]) continue;

			for (auto& it3 : paramInfo) {
				KString sKey = it3.first;
				if (it3.second.type() != Ken_value_type::dict) continue;

				KaiDict param = it3.second;
				if (param.find("_grad_") != param.end()) {
					KaiArray<KFloat> grad = FARRAY(param["_grad_"]);
					pMath->mul_on(grad, ratio);
				}
			}
		}
	}
}

void KaiOptimizer::m_update_weight(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "unreachable code");
}

void KaiOptimizer::m_update_bias(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "unreachable code");
}

void KaiOptimizer::m_update_multi_dic(KaiMath* pMath, KaiDict layerParam) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "unreachable code");
}

KaiArray<KFloat> KaiOptimizer::m_apply_decay(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad) {
	KFloat l2_decay = get_property("l2_decay", 0.0f);
	KFloat l1_decay = get_property("l1_decay", 0.0f);

	if (l2_decay <= 0 && l1_decay <= 0) return grad;

	return pMath->apply_decay(pm, grad, l2_decay, l1_decay);
}

KaiSgdOptimizer::KaiSgdOptimizer(KaiSession* pSession, KaiDict kwArgs) : KaiOptimizer(pSession, kwArgs) {
}

KaiSgdOptimizer::~KaiSgdOptimizer() {
}

void KaiSgdOptimizer::m_update_weight(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo) {
	KFloat learning_rate = get_property("learning_rate");

	KaiArray<KFloat> decay_grad = m_apply_decay(pMath, pm, grad);

	KaiArray<KFloat> mult_grad = pMath->mul(decay_grad, learning_rate);

	pMath->sub_on(pm, mult_grad);
}

void KaiSgdOptimizer::m_update_bias(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo) {
	KFloat learning_rate = get_property("learning_rate");

	KaiArray<KFloat> mult_grad = pMath->mul(grad, learning_rate);

	pMath->sub_on(pm, mult_grad);
}

void KaiSgdOptimizer::m_update_multi_dic(KaiMath* pMath, KaiDict layerParam) {
	KFloat learning_rate = get_property("learning_rate");

	KaiList embed_info = layerParam["embed_info"];

	KaiArray<KFloat> grad = FARRAY(layerParam["_grad_"]);
	KaiArray<KInt> tokens = NARRAY(layerParam["_tokens_"]);

	KFloat l2_decay = get_property("l2_decay", 0.0f);
	KFloat l1_decay = get_property("l1_decay", 0.0f);

	KInt nth = 0;

	for (auto& it : embed_info) {
		KaiDict dic_info = it;
		KString dic_name = dic_info["name"];
		KaiDict dic_param = layerParam[dic_name];
		if (!(KBool)dic_param["train"]) continue;
		KaiDict pm = dic_param["w"];
		KaiArray<KFloat> weight = FARRAY(pm["_pm_"]);
		//printf("update embeded dic : %s\n", dic_name.c_str());
		pMath->update_dic_weight_sgd(weight, grad, tokens, nth++, learning_rate, l2_decay, l1_decay);
	}
}

KaiAdamOptimizer::KaiAdamOptimizer(KaiSession* pSession, KaiDict kwArgs) : KaiOptimizer(pSession, kwArgs) {
}

KaiAdamOptimizer::~KaiAdamOptimizer() {
}

void KaiAdamOptimizer::m_alloc_weight_extend(KaiDict& pm_w, KaiShape param_shape, KBool bTrain) {
	if (bTrain) {
		pm_w["s"] = KaiArray<KFloat>::zeros(param_shape).get_core();
		pm_w["t"] = KaiArray<KFloat>::zeros(param_shape).get_core();
		pm_w["n"] = 0.0f;
	}
}

void KaiAdamOptimizer::m_alloc_bias_extend(KaiDict& pm_b, KaiShape param_shape, KBool bTrain) {
	if (bTrain) {
		pm_b["s"] = KaiArray<KFloat>::zeros(param_shape).get_core();
		pm_b["t"] = KaiArray<KFloat>::zeros(param_shape).get_core();
		pm_b["n"] = 0.0f;
	}
}

void KaiAdamOptimizer::m_alloc_dic_extend(KaiDict& pm_b, KaiShape param_shape, KBool bTrain) {
	if (bTrain) {
		pm_b["s"] = KaiArray<KFloat>::zeros(param_shape).get_core();
		pm_b["t"] = KaiArray<KFloat>::zeros(param_shape).get_core();
		pm_b["n"] = KaiArray<KFloat>::zeros(KaiShape{ param_shape[0] }).get_core();
	}
}

void KaiAdamOptimizer::m_update_weight(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo) {
	KaiArray<KFloat> pm_s = FARRAY(paramInfo["s"]);
	KaiArray<KFloat> pm_t = FARRAY(paramInfo["t"]);

	KFloat pm_n = (KFloat)paramInfo["n"] + 1;
	paramInfo["n"] = pm_n;

	KFloat ro1 = get_property("ro1", 0.9f);
	KFloat ro2 = get_property("ro2", 0.999f);
	KFloat epsilon = get_property("epsilon", 1.0e-8f);
	KFloat learning_rate = get_property("learning_rate", 0.001f);

	KaiArray<KFloat> adam_grad = pMath->eval_adam_delta(grad, pm_s, pm_t, pm_n, ro1, ro2, epsilon);

	KaiArray<KFloat> decay_grad = m_apply_decay(pMath, pm, adam_grad);

	KaiArray<KFloat> mult_grad = pMath->mul(decay_grad, learning_rate);

	pMath->sub_on(pm, mult_grad);
}

void KaiAdamOptimizer::m_update_bias(KaiMath* pMath, KaiArray<KFloat> pm, KaiArray<KFloat> grad, KaiDict& paramInfo) {
	KaiArray<KFloat> pm_s = FARRAY(paramInfo["s"]);
	KaiArray<KFloat> pm_t = FARRAY(paramInfo["t"]);

	KFloat pm_n = (KFloat)paramInfo["n"] + 1;
	paramInfo["n"] = pm_n;

	KFloat ro1 = get_property("ro1", 0.9f);
	KFloat ro2 = get_property("ro2", 0.999f);
	KFloat epsilon = get_property("epsilon", 1.0e-8f);
	KFloat learning_rate = get_property("learning_rate", 0.001f);

	KaiArray<KFloat> adam_grad = pMath->eval_adam_delta(grad, pm_s, pm_t, pm_n, ro1, ro2, epsilon);

	KaiArray<KFloat> mult_grad = pMath->mul(adam_grad, learning_rate);

	pMath->sub_on(pm, mult_grad);
}

void KaiAdamOptimizer::m_update_multi_dic(KaiMath* pMath, KaiDict layerParam) {
	KFloat ro1 = get_property("ro1", 0.9f);
	KFloat ro2 = get_property("ro2", 0.999f);
	KFloat epsilon = get_property("epsilon", 1.0e-8f);
	KFloat learning_rate = get_property("learning_rate");

	KaiList embed_info = layerParam["embed_info"];

	KaiArray<KFloat> grad = FARRAY(layerParam["_grad_"]);
	KaiArray<KInt> tokens = NARRAY(layerParam["_tokens_"]);

	KFloat l2_decay = get_property("l2_decay", 0.0f);
	KFloat l1_decay = get_property("l1_decay", 0.0f);

	KInt nth = 0;

	for (auto& it : embed_info) {
		KaiDict dic_info = it;
		KString dic_name = dic_info["name"];
		KaiDict dic_param = layerParam[dic_name];
		if (!(KBool)dic_param["train"]) continue;
		KaiDict pm = dic_param["w"];
		KaiArray<KFloat> weight = FARRAY(pm["_pm_"]);
		KaiArray<KFloat> s = FARRAY(pm["s"]);
		KaiArray<KFloat> t = FARRAY(pm["t"]);
		KaiArray<KFloat> n = FARRAY(pm["n"]);
		pMath->update_dic_weight_adam(weight, s, t, n, grad, tokens, nth++, learning_rate, l2_decay, l1_decay, ro1, ro2, epsilon);
	}
}
