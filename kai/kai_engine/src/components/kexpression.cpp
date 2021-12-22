/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kexpression.h"
#include "../math/karray.h"
#include "../math/kmath.h"
#include "../exec/exec_context.h"
#include "../utils/kutil.h"
#include "../nightly/nightly_utils.h"

//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s   snprintf
#endif

int KaiExpression::ms_checkCode = 13298124;

KStrList KaiExpression::ms_builtin = { "hungarian" };

KaiExpression::KaiExpression(KaiSession* pSession, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::expression, Ken_object_type::expression, kwArgs) {
	m_checkCode = ms_checkCode;
}

KaiExpression::KaiExpression(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiComponent(pSession, Ken_component_type::expression, Ken_object_type::expression, pLib, pComponentInfo) {
	m_checkCode = ms_checkCode;
}

KaiExpression::~KaiExpression() {
	m_checkCode = 0;
}

KaiExpression* KaiExpression::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Expression");

	KaiExpression* pExpression = (KaiExpression*)hObject;

	if (pExpression->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Expression");
	if (pExpression->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Expression");

	return pExpression;
}

KaiExpression* KaiExpression::HandleToPointer(KHObject hObject, KBool mayNull) {
	if (hObject == NULL) {
		if (mayNull) return NULL;
		throw KaiException(KERR_NULL_HANDLE_USED, "Expression");
	}

	KaiExpression* pExpression = (KaiExpression*)hObject;

	if (pExpression->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Expression");

	return pExpression;
}

void KaiExpression::GetBuiltinNames(KStrList* pslNames) {
	for (auto it = ms_builtin.begin(); it != ms_builtin.end(); it++) {
		pslNames->push_back(*it);
	}
}

KaiExpression* KaiExpression::CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs) {
	KaiExpression* pInstance = NULL;

	if (sBuiltin == "hungarian") pInstance = new KaiHungarianExpression(pSession, kwArgs);
	else if (sBuiltin == "") pInstance = new KaiExpression(pSession, kwArgs);

	if (!pInstance) throw KaiException(KERR_UNKNOWN_NETWORK_SUBCLASS_NAME);

	pInstance->m_propDict["builtin"] = sBuiltin;
	pInstance->m_propDict["desc"] = pInstance->desc();

	return pInstance;
}

/*
KString KaiExpression::get_op_code() {
	return m_propDict["op_str"];
}

KString KaiExpression::get_op_aux() {
	KaiValue opAux = m_propDict["op_aux"];;

	if (opAux.type() == Ken_value_type::string) return (KString)opAux;
	else if (opAux.type() == Ken_value_type::kfloat) return "float(" + std::to_string((KFloat)opAux) + ")";
	else if (opAux.type() == Ken_value_type::kint) return "int(" + std::to_string((KFloat)opAux) + ")";
	else throw KaiException(KERR_INTERNAL_ERR_BAD_OP_AUX_TYPE);
}

KInt KaiExpression::get_operand_count() {
	return (KInt) m_operands.size();
}

KaiExpression* KaiExpression::get_nth_operand(KInt nth) {
	if (nth < 0 || nth >= (KInt) m_operands.size()) throw KaiException(KERR_INDEX_OUT_OF_RANGE);
	return (KaiExpression*)(KHObject)m_operands[nth];
}
*/

KString KaiExpression::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf, 128, "<KaiComponent Expression %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

KaiDict KaiExpression::evaluate(KString sExpName, KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KBool bScalar, KInt mb_size) {
	if (funcInfo.size() == 0) throw KaiException(KERR_EVALUATE_FUNCTION_NOT_DEFINED);

	KaiMath* pMath = pContext->get_math();
	KaiDict eval_graph;

	if (funcInfo.find(sExpName) == funcInfo.end()) {
		eval_graph = ms_create_exp_graph(funcInfo, ys, outs, false);
		funcInfo[sExpName] = eval_graph;
	}
	else {
		eval_graph = funcInfo[sExpName];
		ms_reset_exp_graph(eval_graph);
	}

	KaiLossExpRoot::set_arg("data_count", mb_size);

	for (auto& it : outs) {
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)eval_graph["@est:" + it.first];
		pExp->set_value(it.second);
	}

	for (auto& it : ys) {
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)eval_graph["@ans:" + it.first];
		pExp->set_value(it.second);
	}

	KaiList dict_terms = funcInfo["dict_terms"];

	KaiDict value_dict;

	for (auto& it : dict_terms) {
		KString sTermName = it;
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)eval_graph[sTermName];
		value_dict[sTermName] = pExp->evaluate_value(pMath);
		//loss_dict[sTermName] = pExp->get_value().get_core();
	}

	return value_dict;
}

KaiDict KaiExpression::postproc(KaiDict funcInfo, KaiExecContext* pContext, KaiDict var_dict) {
	KaiDict xs, ys, outs;
	throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	return evaluate(funcInfo, xs, ys, outs, pContext, true, var_dict);
	exp_op op_code = (exp_op)(KInt)funcInfo["op_code"];
	KaiMath* pMath = pContext->get_math();

	KaiDict acc_dict;


	if (op_code == exp_op::dict) {
		KaiList opnd_infos = funcInfo["operands"];

		for (auto& it : opnd_infos) {
			KaiDict opnd_info = it;
			exp_op term_op_code = (exp_op)(KInt)opnd_info["op_code"];
			if (term_op_code == exp_op::term || term_op_code == exp_op::hidden) {
				KaiList term_info = opnd_info["operands"];
				if (term_info.size() != 2) throw KaiException(KERR_CHILDREN_OF_BAD_ACC_TERM);
				KaiDict key_info = term_info[0];
				KaiDict value_info = term_info[1];
				if ((exp_op)(KInt)key_info["op_code"] != exp_op::string) throw KaiException(KERR_CHILDREN_OF_BAD_ACC_TERM);
				KString sKey = key_info["op_aux"];
				KaiArray<KFloat> acc_arr = evaluate_value(value_info, xs, ys, outs, pContext, var_dict);
				if (acc_arr.total_size() != 1) throw KaiException(KERR_ACC_FUNCTION_WITH_NONSCALAR_OUTPUT);
				KFloat accuracy = pMath->fetch(acc_arr, 0);
				if (term_op_code == exp_op::term) acc_dict[sKey] = accuracy;
				var_dict[sKey] = acc_arr.get_core();
			}
			else throw KaiException(KERR_INVALID_ACCURACY_DICT_CHILD);
		}
	}
	else {
		KaiArray<KFloat> acc_arr = evaluate_value(funcInfo, xs, ys, outs, pContext, var_dict);
		if (acc_arr.total_size() != 1) throw KaiException(KERR_ACC_FUNCTION_WITH_NONSCALAR_OUTPUT);
		KFloat accuracy = pMath->fetch(acc_arr, 0);
		acc_dict["#default"] = accuracy;
	}

	return acc_dict;
	*/
}

KaiValue KaiExpression::evaluate_value(KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KaiDict var_dict) {
	throw KaiException(KERR_DEPRECIATED_FUNCTION_IS_CALLED, "KaiExpression::evaluate_value");
	/*
	exp_op op_code = (exp_op)(KInt)funcInfo["op_code"];
	KaiValue op_aux = funcInfo["op_aux"];

	KaiList opnd_infos = funcInfo["operands"];
	KaiList operands;

	for (auto& it : opnd_infos) {
		KaiDict opnd_info = it;
		KaiValue opnd = KaiExpression::evaluate_value(opnd_info, xs, ys, outs, pContext, var_dict);
		operands.push_back(opnd);
	}

	KBool bFloatResult = true;
	KaiArray<KFloat> fresult;
	KaiArray<KInt> nresult;

	KaiArray<KFloat> arr1, arr2, arr3;
	KaiMath* pMath = pContext->get_math();

	// Fixed by Hyung-jae, Son (2021-09-14)
	//KString sOpAux = op_aux;	// Before (error occurs when constant)
	KString sOpAux;	// Ater

	switch (op_code) {
	case exp_op::feed:
		// Added by Hyung-jae, Son (2021-09-14)
		if (op_aux.type() != Ken_value_type::string)
			throw KaiException(KERR_BAD_TYPE_IN_VALUE_CONVERSION);
		else
			sOpAux = op_aux;

		if (sOpAux == "est") { fresult = FARRAY(outs["#default"]); }
		else if (sOpAux.substr(0, 4) == "est:") {
			KString sFieldName = sOpAux.substr(4);
			fresult = FARRAY(outs[sFieldName]);
		}
		else if (sOpAux.substr(0, 4) == "est:") {
			KString sFieldName = sOpAux.substr(4);
			KaiObject* pObject = outs[sFieldName];
			if (pObject->get_type() == Ken_object_type::farray) {
				fresult = FARRAY(outs[sFieldName]);
			}
			else if (pObject->get_type() == Ken_object_type::narray) {
				nresult = NARRAY(outs[sFieldName]);
				bFloatResult = false;
			}
			else {
				throw KaiException(KERR_ANSWER_TENSOR_NOT_FOUND, sOpAux.substr(4));
			}
		}
		else if (sOpAux == "ans") {
			fresult = FARRAY(ys["#default"]);
		}
		else if (sOpAux.substr(0, 4) == "ans:") {
			KString sFieldName = sOpAux.substr(4);
			KaiObject* pObject = ys[sFieldName];
			if (pObject->get_type() == Ken_object_type::farray) {
				fresult = FARRAY(ys[sFieldName]);
			}
			else if (pObject->get_type() == Ken_object_type::narray) {
				nresult = NARRAY(ys[sFieldName]);
				bFloatResult = false;
			}
			else {
				throw KaiException(KERR_ANSWER_TENSOR_NOT_FOUND, sOpAux.substr(4));
			}
		}
		else throw KaiException(KERR_UNIMPEMENTED_YET);
		break;
	case exp_op::arg:
		if (var_dict.find((KString)op_aux) == var_dict.end()) throw KaiException(KERR_UNDEFINED_VAR_USERD_IN_EXP, (KString)op_aux);
		fresult = FARRAY(var_dict[(KString)op_aux]);
		break;
	case exp_op::constant:
		if (op_aux.type() == Ken_value_type::kfloat) {
			KFloat value = op_aux;
			fresult = pMath->ones(KaiShape{ 1 }, value);
		}
		else if (op_aux.type() == Ken_value_type::kint) {
			KFloat value = (KFloat)(KInt)op_aux;
			fresult = pMath->ones(KaiShape{ 1 }, value);
		}
		else throw KaiException(KERR_BAD_CONSTANT_TYPE_IN_EXPRESSION);
		break;
	case exp_op::add:
	case exp_op::mult:
		if (operands.size() < 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		arr2 = FARRAY(operands[1]);

		fresult = pMath->eval_binary_op(op_code, arr1, arr2);

		for (KInt n = 2; n < operands.size(); n++) {
			arr2 = FARRAY(operands[n]);
			fresult = pMath->eval_binary_op(op_code, fresult, arr2);
		}
		break;
	case exp_op::sub:
	case exp_op::div:
	case exp_op::gt:
	case exp_op::lt:
	case exp_op::ge:
	case exp_op::le:
	case exp_op::equal:
	case exp_op::_and:
	case exp_op::_or:
	case exp_op::sigmoid_cross_entropy_with_logits:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		arr2 = FARRAY(operands[1]);

		fresult = pMath->eval_binary_op(op_code, arr1, arr2);
		break;
	case exp_op::softmax_cross_entropy_with_logits:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		arr2 = FARRAY(operands[1]);

		fresult = pMath->softmax_cross_entropy_with_logits(arr1, arr2);
		break;
	case exp_op::equal_col:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		arr2 = FARRAY(operands[1]);

		fresult = pMath->equal_col(arr1, arr2);
		break;
	case exp_op::argmax:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->argmax(arr1);
		break;
	case exp_op::max:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->max(arr1);
		break;
		// *
	case exp_op::add:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(operands[0]);
		farr2 = FARRAY(operands[1]);
		fresult = pMath->add(farr1, farr2);
		break;
	case exp_op::sub:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(operands[0]);
		farr2 = FARRAY(operands[1]);
		fresult = pMath->sub(farr1, farr2);
		break;
	case exp_op::mult:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(operands[0]);
		farr2 = FARRAY(operands[1]);
		fresult = pMath->mul(farr1, farr2);
		break;
	case exp_op::div:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(operands[0]);
		farr2 = FARRAY(operands[1]);
		fresult = pMath->div(farr1, farr2);
		break;
	case exp_op::sigmoid_cross_entropy:
		throw KaiException(KERR_UNIMPEMENTED_YET);
		break;
	case exp_op::equal:
		throw KaiException(KERR_UNIMPEMENTED_YET);
		break;
	* //
	case exp_op::mean:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->mean(arr1);
		break;
	case exp_op::sum:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->sum(arr1);
		break;
	case exp_op::sqrt:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->sqrt(arr1);
		break;
	case exp_op::square:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->square(arr1);
		break;
	case exp_op::log:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->log(arr1);
		break;
	case exp_op::exp:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->exp(arr1);
		break;
	case exp_op::sigmoid:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->sigmoid(arr1);
		break;
	case exp_op::softmax:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		arr1 = FARRAY(operands[0]);
		fresult = pMath->softmax(arr1);
		break;
	case exp_op::subvector:
	{
		if (operands.size() != 3)
			throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		// Get elements
		arr1 = FARRAY(operands[0]);		// constant begin_index
		arr2 = FARRAY(operands[1]);		// constant n_vectors
		arr3 = FARRAY(operands[2]);		// @est or @ans

		// Copy data from device memory to host memory
		arr1 = arr1.to_host();
		arr2 = arr2.to_host();

		// Get option values
		KInt begin_index = (KInt)arr1.get_at(0);
		KInt copy_count  = (KInt)arr2.get_at(0);
		KInt batch_size  = arr3.shape()[0];
		KInt output_size = arr3.shape()[1];

		// Create the temporary buffer on host memory
		//KaiArray<KFloat> temp_buffer = KaiArray<KFloat>::zeros(KaiShape{ batch_size, copy_count });

		// Create the result data block on device memory
		fresult = pMath->ones(KaiShape{ batch_size, copy_count }, 0.0F);

		// Extract a subvector
		for (KInt batch_index=0; batch_index<batch_size; ++batch_index) {
			KInt nSrcStart = batch_index*output_size + begin_index;
			KInt nDstStart = batch_index*copy_count;
			pMath->to_cuda(arr3, fresult, nSrcStart, nDstStart, copy_count);
		}
	}
		break;
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}

	KaiValue result;

	if (bFloatResult) result = fresult.get_core();
	else result = nresult.get_core();

	return result;
	*/
}

/*
KaiArray<KFloat> KaiExpNode::m_grad1_on_binary_op(KaiMath* pMath, KaiArray<KFloat> grad1, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	switch (m_op_code) {
	case exp_op::add:
		return grad1;
	case exp_op::sub:
		return grad1;
	case exp_op::mult:
		return pMath->mul(grad1, arr2);
	case exp_op::div:
		return pMath->div(grad1, arr2);
	case exp_op::sigmoid_cross_entropy_with_logits:
		return pMath->mul(grad1, pMath->sub(pMath->sigmoid(arr1), arr2));
	case exp_op::softmax_cross_entropy_with_logits:
		return pMath->mul(grad1, pMath->sub(pMath->softmax(arr1), arr2));
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}
}
*/

/*
KaiArray<KFloat> KaiExpNode::m_grad2_on_binary_op(KaiMath* pMath, KaiArray<KFloat> grad2, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	switch (m_op_code) {
	case exp_op::add:
		return grad2;
	case exp_op::sub:
		return pMath->minus(grad2);
	case exp_op::mult:
		return pMath->mul(grad2, arr1);
	case exp_op::div:
		return pMath->mul(grad2, pMath->minus(pMath->div(arr1, pMath->mul(arr2, arr2))));
	case exp_op::sigmoid_cross_entropy_with_logits:
		throw KaiException(KERR_UNIMPEMENTED_YET);
	case exp_op::softmax_cross_entropy_with_logits:
		throw KaiException(KERR_UNIMPEMENTED_YET);
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}
}
*/

/*
KaiValue KaiExpNode::eval_with_grad(KaiMath* pMath, KaiDict xs, KaiDict ys, KaiDict outs, KaiDict& grads, KaiDict var_dict, KaiDict exprs) {
	KaiList operands, opnd_grads;

	for (auto& it : m_operands) {
		KaiExpNode* pChildExp = it;
		KaiDict opnd_grad;
		KaiValue opnd = pChildExp->eval_with_grad(pMath, xs, ys, outs, opnd_grad, var_dict, exprs);
		operands.push_back(opnd);
		opnd_grads.push_back(opnd_grad);
	}

	KaiValue result;

	KaiArray<KFloat> arr1, arr2, grad1, grad2;
	KaiArray<KInt> narr1, narr2;
	KaiDict opnd_grad1, opnd_grad2;
	KInt nArg1, nArg2;
	KFloat fArg1, fArg2;
	KString sOpAux;

	switch (m_op_code) {
	case exp_op::feed:
		sOpAux = m_op_aux;
		if (sOpAux == "est") {
			if (outs.find("#default") == outs.end()) throw KaiException(KERR_ESTIMATE_TENSOR_NOT_FOUND, sOpAux);
			result = outs["#default"];
			arr1 = FARRAY(result);
			grads["#default"] = pMath->ones(arr1.shape()).get_core();
		}
		else if (sOpAux.substr(0, 4) == "est:") {
			KString sFieldName = sOpAux.substr(4);
			if (outs.find("sFieldName") == outs.end()) throw KaiException(KERR_ESTIMATE_TENSOR_NOT_FOUND, sOpAux);
			result = outs[sFieldName];
			KaiObject* pObject = result;
			if (pObject->get_type() == Ken_object_type::farray) {
				arr1 = FARRAY(result);
				grads[sFieldName] = pMath->ones(arr1.shape()).get_core();
			}
		}
		else if (sOpAux == "ans") {
			if (ys.find("#default") == ys.end()) throw KaiException(KERR_ANSWER_TENSOR_NOT_FOUND, sOpAux);
			result = ys["#default"];
		}
		else if (sOpAux.substr(0, 4) == "ans:") {
			KString sFieldName = sOpAux.substr(4);
			if (ys.find(sFieldName) == ys.end()) throw KaiException(KERR_ANSWER_TENSOR_NOT_FOUND, sOpAux);
			result = ys[sFieldName];
		}
		else throw KaiException(KERR_UNIMPEMENTED_YET);
		break;
	case exp_op::subexp:
		sOpAux = m_op_aux;
		if (var_dict.find(sOpAux) != var_dict.end()) {
			result = var_dict[sOpAux];
		}
		else if (exprs.find(sOpAux) != exprs.end()) {
			KaiExpRoot* pExp = (KaiExpRoot*)(KaiObject*)exprs[sOpAux];
			result = pExp->m_pRoot->eval_with_grad(pMath, xs, ys, outs, grads, var_dict, exprs);
			var_dict[sOpAux] = result;
		}
		else {
			throw KaiException(KERR_UNIMPEMENTED_YET);
		}
		break;
	case exp_op::arg:
		throw KaiException(KERR_UNIMPEMENTED_YET);
		break;
	case exp_op::subvector:
		arr1 = FARRAY(operands[0]);
		nArg1 = operands[1];
		nArg2 = operands[2];
		result = pMath->get_subvector(arr1, nArg1, nArg2).get_core();

		opnd_grad1 = opnd_grads[0];
		opnd_grad2 = opnd_grads[1];

		for (auto& it : opnd_grad1) {
			throw KaiException(KERR_UNIMPEMENTED_YET);
		}

		for (auto& it : opnd_grad2) {
			throw KaiException(KERR_UNIMPEMENTED_YET);
		}

		break;
	case exp_op::constant:
		result = m_op_aux;
		//if (m_op_aux.type() == Ken_value_type::kfloat) {
		//	KFloat value = m_op_aux;
		//	result = pMath->ones(KaiShape{ 1 }, value).get_core();
		//}
		//else if (m_op_aux.type() == Ken_value_type::kint) {
		//	KFloat value = (KFloat)(KInt)m_op_aux;
		//	result = pMath->ones(KaiShape{ 1 }, value).get_core();
		//}
		//else throw KaiException(KERR_BAD_CONSTANT_TYPE_IN_EXPRESSION);
		break;
	case exp_op::add:
	case exp_op::sub:
	case exp_op::mult:
	case exp_op::div:
	case exp_op::sigmoid_cross_entropy_with_logits:
	case exp_op::softmax_cross_entropy_with_logits:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		arr2 = FARRAY(operands[1]);

		result = pMath->eval_binary_op(m_op_code, arr1, arr2).get_core();

		opnd_grad1 = opnd_grads[0];
		opnd_grad2 = opnd_grads[1];

		for (auto& it : opnd_grad1) {
			KString sKey = it.first;
			grad1 = FARRAY(it.second);
			grad1 = m_grad1_on_binary_op(pMath, grad1, arr1, arr2);
			if (opnd_grad2.find(sKey) != opnd_grad2.end()) {
				grad2 = FARRAY(opnd_grad2[sKey]);
				grad2 = m_grad2_on_binary_op(pMath, grad2, arr1, arr2);
				grads[sKey] = pMath->add(grad1, grad2).get_core();
			}
			else {
				grads[sKey] = grad1.get_core();
			}
		}

		for (auto& it : opnd_grad2) {
			KString sKey = it.first;
			if (opnd_grad1.find(sKey) == opnd_grad1.end()) {
				grad2 = FARRAY(it.second);
				grad2 = m_grad2_on_binary_op(pMath, grad2, arr1, arr2);
				grads[sKey] = grad2.get_core();

				//dump_grad = FARRAY(grads[sKey]);
				//dump_grad.dump("grad for" + sKey + ":" + std::to_string((int)op_code), true);
			}
		}
		break;
	case exp_op::_and:
	case exp_op::_or:
	case exp_op::gt:
	case exp_op::lt:
	case exp_op::ge:
	case exp_op::le:
	case exp_op::equal:
	case exp_op::equal_col:
		throw KaiException(KERR_NOT_DIRREFENTIABLE_FUNC_IN_LOSS);
		break;
	case exp_op::mean:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		result = pMath->mean(arr1).get_core();

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->mul(grad1, 1.0f / (KFloat)arr1.total_size()).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::sqrt:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		result = pMath->sqrt(arr1).get_core();

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			arr2 = FARRAY(result);
			grads[it.first] = pMath->div(grad1, pMath->mul(arr2, 2.0f)).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::square:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		result = pMath->square(arr1).get_core();

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->mul(grad1, pMath->mul(arr1, 2.0f)).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::log:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		result = pMath->log(arr1).get_core();

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->div(grad1, arr1).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::exp:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		result = pMath->exp(arr1).get_core();

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			arr2 = FARRAY(result);
			grads[it.first] = pMath->mul(grad1, arr2).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::filter:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		narr1 = NARRAY(operands[1]);

		result = pMath->filter(arr1, narr1).get_core();

		opnd_grad1 = opnd_grads[0];

		for (auto& it : opnd_grad1) {
			KString sKey = it.first;
			grad1 = FARRAY(it.second);
			throw KaiException(KERR_UNIMPEMENTED_YET);
			//grad1 = ms_grad1_on_binary_op(op_code, pMath, grad1, arr1, arr2);
			grads[sKey] = grad1.get_core();
		}
		break;
	case exp_op::softmax_cross_entropy_with_logits_idx:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		narr2 = NARRAY(operands[1]);

		result = pMath->softmax_cross_entropy_with_logits_idx(arr1, narr2).get_core();

		opnd_grad1 = opnd_grads[0];

		for (auto& it : opnd_grad1) {
			KString sKey = it.first;
			grad1 = FARRAY(it.second);
			grad1 = pMath->mul(grad1, pMath->softmax_cross_entropy_with_logits_idx_derv(arr1, narr2));
			grads[sKey] = grad1.get_core();
		}
		break;
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}

	return result;
}
*/

KaiDict KaiExpression::ms_create_exp_graph(KaiDict funcInfo, KaiDict ys, KaiDict outs, KBool bLoss) {
	KaiDict graph;
	KaiDict expressions = funcInfo["dict_exprs"];

	for (auto& it : expressions) {
		KaiLossExpRoot* pExp = new KaiLossExpRoot(KBool3::unknown, it.second);
		graph[it.first] = pExp;
	}

	for (auto& it : outs) {
		KaiLossExpRoot* pExp = new KaiLossExpRoot(KBool3::on);
		graph["@est:" + it.first] = pExp;
	}

	for (auto& it : ys) {
		KaiLossExpRoot* pExp = new KaiLossExpRoot(KBool3::off);
		graph["@ans:" + it.first] = pExp;
	}

	for (auto& it : expressions) {
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)graph[it.first];
		pExp->build_graph(graph, bLoss);
	}

	return graph;
}

void KaiExpression::ms_reset_exp_graph(KaiDict graph) {
	for (auto& it : graph) {
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)(it.second);
		pExp->reset();
	}
}

KaiDict KaiExpression::evaluate_with_grad(KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KaiDict& grads, KInt mb_size) {
	if (funcInfo.size() == 0) throw KaiException(KERR_LOSS_FUNCTION_NOT_DEFINED);

	KaiMath* pMath = pContext->get_math();
	KaiDict loss_graph;

	if (funcInfo.find("#loss_graph") == funcInfo.end()) {
		loss_graph = ms_create_exp_graph(funcInfo, ys, outs, true);
		funcInfo["#loss_graph"] = loss_graph;
	}
	else {
		loss_graph = funcInfo["#loss_graph"];
		ms_reset_exp_graph(loss_graph);
	}

	KaiLossExpRoot::set_arg("data_count", mb_size);

	for (auto& it : outs) {
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)loss_graph["@est:" + it.first];
		pExp->set_value(it.second);
	}

	for (auto& it : ys) {
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)loss_graph["@ans:" + it.first];
		pExp->set_value(it.second);
	}

	KaiList dict_terms = funcInfo["dict_terms"];
	KaiArray<KFloat> loss_grad = pMath->ones(KaiShape{ 1 });

	KaiDict loss_dict;

	for (auto& it : dict_terms) {
		KString sTermName = it;
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)loss_graph[sTermName];
		pExp->set_grad(loss_grad);
		loss_dict[sTermName] = pExp->evaluate_value(pMath);
		/*
		KaiArray<KFloat> dump_arr = FARRAY(loss_dict[sTermName]);
		dump_arr.dump(sTermName);
		*/
	}

	/*
	for (auto& it : loss_graph) {
		KString sTermName = it.first;
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)loss_graph[sTermName];
		KaiValue value = pExp->evaluate_value(pMath);
		if (value.is_farray()) {
			KaiArray<KFloat> dump_arr = FARRAY(value);
			//dump_arr.dump(sTermName);
			//KaiArray<KFloat> sum_val = pMath->sum(dump_arr);
			//KFloat sval = pMath->fetch(sum_val);
			printf("%s: farray%s\n", sTermName.c_str(), dump_arr.shape().desc().c_str());
		}
		else {
			//printf("%s: %s\n", sTermName.c_str(), value.desc().c_str());
		}
	}
	*/

	for (auto& it : outs) {
		KString sTermName = it.first;
		KaiLossExpRoot* pExp = (KaiLossExpRoot*)(KaiObject*)loss_graph["@est:" + sTermName];
		KaiArray<KFloat> grad = pExp->evaluate_grad(pMath);
		if (!grad.is_empty()) {
			grads[sTermName] = grad.get_core();
			//KaiArray<KFloat> dump_arr = FARRAY(grads[sTermName]);
			//dump_arr.dump(sTermName);
			//KaiArray<KFloat> sum_val = pMath->sum(dump_arr);
			//KFloat sval = pMath->fetch(sum_val);
			//printf("sum is %f\n", sval);
		}
	}

	return loss_dict;
}

/*
KaiValue KaiExpression::evaluate_value_with_grad(KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KaiDict& grads, KaiDict var_dict) {
	exp_op op_code = (exp_op)(KInt)funcInfo["op_code"];
	KaiValue op_aux = funcInfo["op_aux"];

	KaiList opnd_infos = funcInfo["operands"];
	KaiList operands, opnd_grads;

	for (auto& it : opnd_infos) {
		KaiDict opnd_info = it;
		KaiDict opnd_grad;
		KaiValue opnd = KaiExpression::evaluate_value_with_grad(opnd_info, xs, ys, outs, pContext, opnd_grad, var_dict);
		operands.push_back(opnd);
		opnd_grads.push_back(opnd_grad);
	}

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_EXPRESSION_EVALUATE_WITH_GRAD)
	{
		printf("[TRACE]  %s(%u) {\n", __FUNCTION__, __LINE__);
		print_klist(opnd_infos, "", 4, "opnd_infos");
		print_klist(operands, "", 4, "operands");
		print_klist(opnd_grads, "", 4, "opnd_grads");
		printf("}\n\n");
	}
#endif

	KBool bFloatResult = true;
	KaiArray<KFloat> fresult;
	KaiArray<KInt> nresult;

	KaiArray<KFloat> arr1, arr2, arr3, grad1, grad2;
	KaiArray<KInt> narr1, narr2;

	KaiDict opnd_grad1, opnd_grad2;
	KaiMath* pMath = pContext->get_math();
	KString sOpAux;

	//KaiArray<KFloat> dump_grad;

	switch (op_code) {
	case exp_op::feed:
		sOpAux = op_aux;
		if (sOpAux == "est") {
			fresult = FARRAY(outs["#default"]);
			grads["#default"] = pMath->ones(fresult.shape()).get_core();
		}
		else if (sOpAux.substr(0, 4) == "est:") {
			KString sFieldName = sOpAux.substr(4);
			KaiObject* pObject = outs[sFieldName];
			if (pObject->get_type() == Ken_object_type::farray) {
				fresult = FARRAY(outs[sFieldName]);
			grads[sFieldName] = pMath->ones(fresult.shape()).get_core();
			}
			else if (pObject->get_type() == Ken_object_type::narray) {
				nresult = NARRAY(outs[sFieldName]);
				bFloatResult = false;
			}
			else {
				throw KaiException(KERR_ANSWER_TENSOR_NOT_FOUND, sOpAux.substr(4));
			}
		}
		else if (sOpAux == "ans") {
			fresult = FARRAY(ys["#default"]);
		}
		else if (sOpAux.substr(0, 4) == "ans:") {
			KString sFieldName = sOpAux.substr(4);
			KaiObject* pObject = ys[sFieldName];
			if (pObject->get_type() == Ken_object_type::farray) {
				fresult = FARRAY(ys[sFieldName]);
			}
			else if (pObject->get_type() == Ken_object_type::narray) {
				nresult = NARRAY(ys[sFieldName]);
				bFloatResult = false;
			}
			else {
				throw KaiException(KERR_ANSWER_TENSOR_NOT_FOUND, sOpAux.substr(4));
			}
		}
		else throw KaiException(KERR_UNIMPEMENTED_YET);
		break;
	case exp_op::arg:
		throw KaiException(KERR_UNIMPEMENTED_YET);
	case exp_op::constant:
		if (op_aux.type() == Ken_value_type::kfloat) {
			KFloat value = op_aux;
			fresult = pMath->ones(KaiShape{ 1 }, value);
		}
		else if (op_aux.type() == Ken_value_type::kint) {
			KFloat value = (KFloat)(KInt)op_aux;
			fresult = pMath->ones(KaiShape{ 1 }, value);
		}
		else throw KaiException(KERR_BAD_CONSTANT_TYPE_IN_EXPRESSION);
		break;
	case exp_op::add:
	case exp_op::sub:
	case exp_op::mult:
	case exp_op::div:
	case exp_op::sigmoid_cross_entropy_with_logits:
	case exp_op::softmax_cross_entropy_with_logits:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		arr2 = FARRAY(operands[1]);

		fresult = pMath->eval_binary_op(op_code, arr1, arr2);

		opnd_grad1 = opnd_grads[0];
		opnd_grad2 = opnd_grads[1];

		for (auto& it : opnd_grad1) {
			KString sKey = it.first;
			grad1 = FARRAY(it.second);
			grad1 = ms_grad1_on_binary_op(op_code, pMath, grad1, arr1, arr2);
			if (opnd_grad2.find(sKey) != opnd_grad2.end()) {
				grad2 = FARRAY(opnd_grad2[sKey]);
				grad2 = ms_grad2_on_binary_op(op_code, pMath, grad2, arr1, arr2);
				grads[sKey] = pMath->add(grad1, grad2).get_core();
			}
			else {
				grads[sKey] = grad1.get_core();
			}
		}

		for (auto& it : opnd_grad2) {
			KString sKey = it.first;
			if (opnd_grad1.find(sKey) == opnd_grad1.end()) {
				grad2 = FARRAY(it.second);
				grad2 = ms_grad2_on_binary_op(op_code, pMath, grad2, arr1, arr2);
				grads[sKey] = grad2.get_core();

				//dump_grad = FARRAY(grads[sKey]);
				//dump_grad.dump("grad for" + sKey + ":" + std::to_string((int)op_code), true);
			}
		}
		break;
	case exp_op::_and:
	case exp_op::_or:
	case exp_op::gt:
	case exp_op::lt:
	case exp_op::ge:
	case exp_op::le:
	case exp_op::equal:
	case exp_op::equal_col:
		throw KaiException(KERR_NOT_DIRREFENTIABLE_FUNC_IN_LOSS);
		break;
	case exp_op::mean:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		fresult = pMath->mean(arr1);

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->mul(grad1, 1.0f / (KFloat)arr1.total_size()).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::sqrt:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		fresult = pMath->sqrt(arr1);

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->div(grad1, pMath->mul(fresult, 2.0f)).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::square:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		fresult = pMath->square(arr1);

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->mul(grad1, pMath->mul(arr1, 2.0f)).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::log:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		fresult = pMath->log(arr1);

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->div(grad1, arr1).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::exp:
		if (operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		fresult = pMath->exp(arr1);

		opnd_grad1 = opnd_grads[0];
		for (auto& it : opnd_grad1) {
			grad1 = FARRAY(it.second);
			grads[it.first] = pMath->mul(grad1, fresult).get_core();

			//dump_grad = FARRAY(grads[it.first]);
			//arr1.dump("arr1 for" + it.first + ":" + std::to_string((int)op_code), true);
			//dump_grad.dump("grad for" + it.first + ":" + std::to_string((int)op_code), true);
		}
		break;
	case exp_op::filter:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		narr1 = NARRAY(operands[1]);

		fresult = pMath->filter(arr1, narr1);

		opnd_grad1 = opnd_grads[0];

		for (auto& it : opnd_grad1) {
			KString sKey = it.first;
			grad1 = FARRAY(it.second);
			throw KaiException(KERR_UNIMPEMENTED_YET);
			//grad1 = ms_grad1_on_binary_op(op_code, pMath, grad1, arr1, arr2);
			grads[sKey] = grad1.get_core();
		}
		break;
	case exp_op::softmax_cross_entropy_with_logits_idx:
		if (operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		arr1 = FARRAY(operands[0]);
		narr2 = NARRAY(operands[1]);

		fresult = pMath->softmax_cross_entropy_with_logits_idx(arr1, narr2);

		opnd_grad1 = opnd_grads[0];

		for (auto& it : opnd_grad1) {
			KString sKey = it.first;
			grad1 = FARRAY(it.second);
			grad1 = pMath->mul(grad1, pMath->softmax_cross_entropy_with_logits_idx_derv(arr1, narr2));
			grads[sKey] = grad1.get_core();
		}
		break;
	case exp_op::subvector:
	{
		if (operands.size() != 3)
			throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		// Get elements
		arr1 = FARRAY(operands[0]);		// constant begin_index
		arr2 = FARRAY(operands[1]);		// constant n_vectors
		arr3 = FARRAY(operands[2]);		// @est or @ans

		// Get option values
		KInt begin_index = (KInt)arr1.to_host().get_at(0);
		KInt copy_count  = (KInt)arr2.to_host().get_at(0);
		KInt batch_size  = arr3.shape()[0];
		KInt output_size = arr3.shape()[1];

		// Create the temporary buffer on host memory
		//KaiArray<KFloat> temp_buffer = KaiArray<KFloat>::zeros(KaiShape{ batch_size, copy_count });

		// Create the result data block on device memory
		fresult = pMath->ones(KaiShape{ batch_size, copy_count }, 0.0F);

		// Extract a subvector
		for (KInt batch_index=0; batch_index<batch_size; ++batch_index) {
			KInt nSrcStart = batch_index*output_size + begin_index;
			KInt nDstStart = batch_index*copy_count;
			pMath->to_cuda(arr3, fresult, nSrcStart, nDstStart, copy_count);
		}

		// Get gradient list
		opnd_grad1 = opnd_grads[2];

		for (auto& it : opnd_grad1) {
			// Get an original gradient array
			grad1 = FARRAY(it.second);

			// Create the subvector data block of the gradient on device memory
			KaiArray<KFloat> sub_grad = pMath->ones(KaiShape{ batch_size, copy_count }, 0.0F);

			// Extract a subvector
			for (KInt batch_index=0; batch_index<batch_size; ++batch_index) {
				KInt nSrcStart = batch_index*output_size + begin_index;
				KInt nDstStart = batch_index*copy_count;
				pMath->to_cuda(grad1, sub_grad, nSrcStart, nDstStart, copy_count);
			}

			// Link to output
			grads[it.first] = sub_grad.get_core();
		}
	}
		break;
	case exp_op::sum:
	{
		if (operands.size() != 2)
			throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		// Get elements
		arr1 = FARRAY(operands[0]);		// mean1
		arr2 = FARRAY(operands[1]);		// mean2

		// Calculate the sum
		fresult = pMath->eval_binary_op(exp_op::add, arr1, arr2);

		// Get gradient list
		opnd_grad1 = opnd_grads[0];
		opnd_grad2 = opnd_grads[1];
		
		// Check the validation
		if (opnd_grad1.size() != opnd_grad2.size())
			throw KaiException(KERR_ASSERT);

		for (auto& it : opnd_grad1) {
			// Find data with the keyword
			KaiDictIter it2 = opnd_grad2.find(it.first);

			// Check the validation
			if (it2 == opnd_grad2.end())
				throw KaiException(KERR_ASSERT);

			// Get an original gradient arrays
			grad1 = FARRAY(it.second);
			grad2 = FARRAY(it2->second);

			// Get option values
			KInt batch_size    = grad1.shape()[0];
			KInt input_size[2] = { grad1.shape()[1], grad2.shape()[1] };
			KInt output_size   = input_size[0] + input_size[1];

			// Check the validation
			if (batch_size != grad2.shape()[0])
				throw KaiException(KERR_ASSERT);

			// Create the merged vector data block of the gradients on device memory
			KaiArray<KFloat> merged_grad = pMath->ones(KaiShape{ batch_size, output_size }, 0.0F);

			// Merge two vectors
			for (KInt batch_index=0; batch_index<batch_size; ++batch_index) {
				KInt nSrcStart1 = batch_index*input_size[0];
				KInt nDstStart1 = batch_index*output_size;
				pMath->to_cuda(grad1, merged_grad, nSrcStart1, nDstStart1, input_size[0]);

				KInt nSrcStart2 = batch_index*input_size[1];
				KInt nDstStart2 = batch_index*output_size + input_size[0];
				pMath->to_cuda(grad2, merged_grad, nSrcStart2, nDstStart2, input_size[1]);
			}

			// Link to output
			grads[it.first] = merged_grad.get_core();
		}
	}
		break;
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}

	KString sLoss = fresult.get_core()->desc();
	KString sGrad = grads.desc();

	KaiValue result;

	if (bFloatResult) result = fresult.get_core();
	else result = nresult.get_core();

	return result;
}
*/

KaiHungarianExpression::KaiHungarianExpression(KaiSession* pSession, KaiDict kwArgs) : KaiExpression(pSession, kwArgs) {
	KString sExp = get_property("exp", "");
	KaiDict terms = get_property("terms", KaiDict());
	KaiDict subterms = get_property("subterms", KaiDict());

	KaiDict dict_exprs;
	KaiList dict_terms;

	if (sExp != "") {
		dict_terms.push_back("#default");
		dict_exprs["#default"] = m_parse_exp(sExp);
	}

	for (auto& it : terms) {
		dict_terms.push_back(it.first);
		dict_exprs[it.first] = m_parse_exp(it.second);
	}

	for (auto& it : subterms) {
		dict_exprs[it.first] = m_parse_exp(it.second);
	}

	m_propDict["dict_exprs"] = dict_exprs;
	m_propDict["dict_terms"] = dict_terms;

	if (m_propDict.find("postproc") != m_propDict.end()) {
		m_propDict["postproc_exp"] = m_parse_exp((KString)m_propDict["postproc"]);
	}

	m_bDirty = false;
}

KaiHungarianExpression::~KaiHungarianExpression() {
}

KaiDict KaiHungarianExpression::m_parse_exp(KString sExp) {
	const char* exp = sExp.c_str();
	char* pExp = new char[strlen(exp) + 1];

	KInt nth = 0, m = 0;

	for (size_t n = 0; n < strlen(exp); n++) {
		if (strchr(" \t\r\n", exp[n])) continue;
		pExp[m++] = exp[n]; //pExp[m] = 0;
		if (exp[n] == '\'') {
			while (exp[++n] != '\'') {
				if (exp[n] == 0) break;
				pExp[m++] = exp[n]; //pExp[m] = 0;
			}
			pExp[m++] = exp[n]; //pExp[m] = 0;
		}
	}

	pExp[m] = 0;

	KaiDict exp_info = m_parse_subexp(pExp, nth);

	if (pExp[nth] != 0) {
		KString errString = KString(pExp + nth);
		delete[] pExp;
		throw KaiException(KERR_EXP_WITH_USELESS_TAIL, sExp, errString);
	}

	delete[] pExp;

	return exp_info;
}

KaiDict KaiHungarianExpression::m_parse_subexp(const char* pExp, KInt& nth) {
	KaiDict exp_info = m_split_op_code(pExp, nth);
	exp_info["operands"] = m_seek_operands(pExp, nth);
	return exp_info;
}

KaiDict KaiHungarianExpression::m_split_op_code(const char* pExp, KInt& nth) {
	KInt start = nth;

	enum exp_op op_code;
	KaiValue op_aux;

	while (pExp[nth] && !strchr("(),", pExp[nth])) nth++;

	if (pExp[start] == '@') {
		op_code = exp_op::feed;
		op_aux = KString(pExp + start, nth - start);
	}
	else if (pExp[start] == '#') {
		op_code = exp_op::arg;
		op_aux = KString(pExp + start + 1, nth - start - 1);
	}
	else if (pExp[start] == '%') {
		op_code = exp_op::subexp;
		op_aux = KString(pExp + start + 1, nth - start - 1);
	}
	else if (pExp[start] >= '0' && pExp[start] <= '9') {
		op_code = exp_op::constant;
		op_aux = KString(pExp + start + 1, nth - start - 1);
		KString val = KString(pExp + start, nth - start);
		if ((val.find('.') == KString::npos) && (val.find('f') == KString::npos) && (val.find('e') == KString::npos)) {
			op_aux = std::stoll(val);
		}
		else {
			op_aux = std::stof(val);
		}
	}
	else {
		KString sOpCode = KString(pExp + start, nth - start);

		//if (sOpCode == "dict") op_code = exp_op::dict;
		//else if (sOpCode == "term") op_code = exp_op::term;
		//else if (sOpCode == "hidden") op_code = exp_op::hidden;
		//else 
		if (sOpCode == "add") op_code = exp_op::add;
		else if (sOpCode == "sub") op_code = exp_op::sub;
		else if (sOpCode == "mult") op_code = exp_op::mult;
		else if (sOpCode == "div") op_code = exp_op::div;
		else if (sOpCode == "equal") op_code = exp_op::equal;
		else if (sOpCode == "equal_col") op_code = exp_op::equal_col;
		else if (sOpCode == "and") op_code = exp_op::_and;
		else if (sOpCode == "or") op_code = exp_op::_or;
		else if (sOpCode == "gt") op_code = exp_op::gt;
		else if (sOpCode == "lt") op_code = exp_op::lt;
		else if (sOpCode == "ge") op_code = exp_op::ge;
		else if (sOpCode == "le") op_code = exp_op::le;
		else if (sOpCode == "exp") op_code = exp_op::exp;
		else if (sOpCode == "log") op_code = exp_op::log;
		else if (sOpCode == "mean") op_code = exp_op::mean;
		else if (sOpCode == "sum") op_code = exp_op::sum;
		else if (sOpCode == "sqrt") op_code = exp_op::sqrt;
		else if (sOpCode == "square") op_code = exp_op::square;
		else if (sOpCode == "argmax") op_code = exp_op::argmax;
		else if (sOpCode == "max") op_code = exp_op::max;
		else if (sOpCode == "max_col") op_code = exp_op::max_col;
		else if (sOpCode == "sigmoid") op_code = exp_op::sigmoid;
		else if (sOpCode == "softmax") op_code = exp_op::softmax;
		else if (sOpCode == "subvector") op_code = exp_op::subvector;
		else if (sOpCode == "vstack") op_code = exp_op::vstack;
		else if (sOpCode == "filter") op_code = exp_op::filter;
		else if (sOpCode == "iou") op_code = exp_op::iou;
		else if (sOpCode == "sigmoid_cross_entropy_with_logits") op_code = exp_op::sigmoid_cross_entropy_with_logits;
		else if (sOpCode == "softmax_cross_entropy_with_logits") op_code = exp_op::softmax_cross_entropy_with_logits;
		else if (sOpCode == "softmax_cross_entropy_with_logits_idx") op_code = exp_op::softmax_cross_entropy_with_logits_idx;
		else if (sOpCode[0] == '\'') {
			if (sOpCode[sOpCode.size() - 1] != '\'') throw KaiException(KERR_EXP_UNKNOWN_OPERATOR, sOpCode);
			op_code = exp_op::string;
			op_aux = sOpCode.substr(1, sOpCode.size() - 2);
		}
		else {
			printf("sOpCode = %s, exp_rest = %s\n", sOpCode.c_str(), pExp + nth);
			throw KaiException(KERR_EXP_UNKNOWN_OPERATOR, sOpCode);
		}
	}

	KaiDict exp_info;

	exp_info["op_code"] = (KInt)op_code;
	exp_info["op_aux"] = op_aux;

	return exp_info;
}

KaiList KaiHungarianExpression::m_seek_operands(const char* pExp, KInt& nth) {
	KaiList operands;

	if (pExp[nth] != '(') return operands;

	nth++;

	while (true) {
		KaiDict child_exp = m_parse_subexp(pExp, nth);
		operands.push_back(child_exp);

		const char ch = pExp[nth++];
		if (ch == ')') break;
		else if (ch != ',') {
			KaiException(KERR_EXP_ILLFORMED_OPERANDS, KString(pExp), KString(pExp + nth));
		}
	}

	return operands;
}

/*
KaiExpRoot* KaiHungarianExpression::m_parse_exp(KString sExp) {
	const char* exp = sExp.c_str();
	char* pExp = new char[strlen(exp) + 1];

	printf("Parsing: %s\n", sExp.c_str());

	KInt nth = 0, m = 0;

	for (size_t n = 0; n < strlen(exp); n++) {
		if (strchr(" \t\r\n", exp[n])) continue;
		pExp[m++] = exp[n]; //pExp[m] = 0;
		if (exp[n] == '\'') {
			while (exp[++n] != '\'') {
				if (exp[n] == 0) break;
				pExp[m++] = exp[n]; //pExp[m] = 0;
			}
			pExp[m++] = exp[n]; //pExp[m] = 0;
		}
	}

	pExp[m] = 0;

	KaiExpRoot* pExpRoot = new KaiExpRoot(m_parse_subexp(pExp, nth));

	if (pExp[nth] != 0) {
		KString errString = KString(pExp + nth);
		delete[] pExp;
		throw KaiException(KERR_EXP_WITH_USELESS_TAIL, sExp, errString);
	}

	delete[] pExp;

	return pExpRoot;
}
*/

/*
KaiExpNode* KaiHungarianExpression::m_parse_subexp(const char* exp, KInt& nth) {
	KaiExpNode* pNode = new KaiExpNode();
	m_split_op_code(pNode, exp, nth);
	m_seek_operands(pNode->m_operands, exp, nth);
	return pNode;
}
*/

/*
void KaiHungarianExpression::m_split_op_code(KaiExpNode* pNode, const char* exp, KInt& nth) {
	KInt start = nth;
	while (exp[nth] && !strchr("(),", exp[nth])) nth++;

	if (exp[start] == '@') {
		pNode->m_op_code = exp_op::feed;
		pNode->m_op_aux = KString(exp + start + 1, nth - start - 1);
	}
	else if (exp[start] == '#') {
		pNode->m_op_code = exp_op::arg;
		pNode->m_op_aux = KString(exp + start + 1, nth - start - 1);
	}
	else if (exp[start] == '%') {
		pNode->m_op_code = exp_op::subexp;
		pNode->m_op_aux = KString(exp + start + 1, nth - start - 1);
	}
	else if (exp[start] >= '0' && exp[start] <= '9') {
		pNode->m_op_code = exp_op::constant;
		pNode->m_op_aux = KString(exp + start + 1, nth - start - 1);
		KString val = KString(exp + start, nth - start);
		if ((val.find('.') == KString::npos) && (val.find('f') == KString::npos) && (val.find('e') == KString::npos)) {
			pNode->m_op_aux = std::stoll(val);
		}
		else {
			pNode->m_op_aux = std::stof(val);
		}
	}
	else {
		KString sOpCode = KString(exp + start, nth - start);

		if (sOpCode == "dict") pNode->m_op_code = exp_op::dict;
		else if (sOpCode == "term") pNode->m_op_code = exp_op::term;
		else if (sOpCode == "hidden") pNode->m_op_code = exp_op::hidden;
		else if (sOpCode == "add") pNode->m_op_code = exp_op::add;
		else if (sOpCode == "sub") pNode->m_op_code = exp_op::sub;
		else if (sOpCode == "mult") pNode->m_op_code = exp_op::mult;
		else if (sOpCode == "div") pNode->m_op_code = exp_op::div;
		else if (sOpCode == "equal") pNode->m_op_code = exp_op::equal;
		else if (sOpCode == "equal_col") pNode->m_op_code = exp_op::equal_col;
		else if (sOpCode == "and") pNode->m_op_code = exp_op::_and;
		else if (sOpCode == "or") pNode->m_op_code = exp_op::_or;
		else if (sOpCode == "gt") pNode->m_op_code = exp_op::gt;
		else if (sOpCode == "lt") pNode->m_op_code = exp_op::lt;
		else if (sOpCode == "ge") pNode->m_op_code = exp_op::ge;
		else if (sOpCode == "le") pNode->m_op_code = exp_op::le;
		else if (sOpCode == "exp") pNode->m_op_code = exp_op::exp;
		else if (sOpCode == "log") pNode->m_op_code = exp_op::log;
		else if (sOpCode == "mean") pNode->m_op_code = exp_op::mean;
		else if (sOpCode == "sum") pNode->m_op_code = exp_op::sum;
		else if (sOpCode == "sqrt") pNode->m_op_code = exp_op::sqrt;
		else if (sOpCode == "square") pNode->m_op_code = exp_op::square;
		else if (sOpCode == "argmax") pNode->m_op_code = exp_op::argmax;
		else if (sOpCode == "max") pNode->m_op_code = exp_op::max;
		else if (sOpCode == "sigmoid") pNode->m_op_code = exp_op::sigmoid;
		else if (sOpCode == "softmax") pNode->m_op_code = exp_op::softmax;
		else if (sOpCode == "subvector") pNode->m_op_code = exp_op::subvector;
		else if (sOpCode == "filter") pNode->m_op_code = exp_op::filter;
		else if (sOpCode == "sigmoid_cross_entropy_with_logits") pNode->m_op_code = exp_op::sigmoid_cross_entropy_with_logits;
		else if (sOpCode == "softmax_cross_entropy_with_logits") pNode->m_op_code = exp_op::softmax_cross_entropy_with_logits;
		else if (sOpCode == "softmax_cross_entropy_with_logits_idx") pNode->m_op_code = exp_op::softmax_cross_entropy_with_logits_idx;
		else if (sOpCode[0] == '\'') {
			if (sOpCode[sOpCode.size() - 1] != '\'') throw KaiException(KERR_EXP_UNKNOWN_OPERATOR, sOpCode);
			pNode->m_op_code = exp_op::string;
			pNode->m_op_aux = sOpCode.substr(1, sOpCode.size() - 2);
		}
		else {
			printf("sOpCode = %s, exp_rest = %s\n", sOpCode.c_str(), exp+nth);
			throw KaiException(KERR_EXP_UNKNOWN_OPERATOR, sOpCode);
		}
	}
}
*/

/*
void KaiHungarianExpression::m_seek_operands(vector<KaiExpNode*>& operands, const char* exp, KInt& nth) {
	if (exp[nth] != '(') return;

	nth++;

	while (true) {
		KaiExpNode* pChild = m_parse_subexp(exp, nth);

		operands.push_back(pChild);

		const char ch = exp[nth++];
		if (ch == ')') break;
		else if (ch != ',') {
			KaiException(KERR_EXP_ILLFORMED_OPERANDS, KString(exp), KString(exp + nth));
		}
	}
}

KaiExpNode::KaiExpNode() {
}

KaiExpNode::~KaiExpNode() {
	for (auto& it : m_operands) {
		delete it;
	}
}

*/

/*
void KaiHungarianExpression::m_parse_exp(KaiExpression* pNode, KString sExp) {
	const char* exp = sExp.c_str();
	char* pExp = new char[strlen(exp) + 1];

	KInt nth = 0, m = 0;

	for (size_t n = 0; n < strlen(exp); n++) {
		if (strchr(" \t\r\n", exp[n])) continue;
		pExp[m++] = exp[n]; //pExp[m] = 0;
		if (exp[n] == '\'') {
			while (exp[++n] != '\'') {
				if (exp[n] == 0) break;
				pExp[m++] = exp[n]; //pExp[m] = 0;
			}
			pExp[m++] = exp[n]; //pExp[m] = 0;
		}
	}
	
	pExp[m] = 0;

	m_parse_subexp(pNode, pExp, nth);

	if (pExp[nth] != 0) {
		KString errString = KString(pExp + nth);
		delete[] pExp;
		throw KaiException(KERR_EXP_WITH_USELESS_TAIL, sExp, errString);
	}

	delete[] pExp;
}

void KaiHungarianExpression::m_parse_subexp(KaiExpression* pNode, const char* exp, KInt& nth) {
	m_split_op_code(pNode, exp, nth);
	m_seek_operands(pNode, exp, nth);
}

void KaiHungarianExpression::m_split_op_code(KaiExpression* pNode, const char* exp, KInt& nth) {
	KInt start = nth;
	while (exp[nth] && !strchr("(),", exp[nth])) nth++;

	if (exp[start] == '@') {
		KInt op_code = (KInt)exp_op::feed;
		pNode->m_propDict["op_str"] = "feed";
		pNode->m_propDict["op_code"] = op_code;
		pNode->m_propDict["op_aux"] = KString(exp + start + 1, nth - start - 1);
	}
	else if (exp[start] == '#') {
		KInt op_code = (KInt)exp_op::arg;
		pNode->m_propDict["op_str"] = "arg";
		pNode->m_propDict["op_code"] = op_code;
		pNode->m_propDict["op_aux"] = KString(exp + start + 1, nth - start - 1);
	}
	else if (exp[start] >= '0' && exp[start] <= '9') {
		KInt op_code = (KInt)exp_op::constant;
		pNode->m_propDict["op_str"] = "constant";
		pNode->m_propDict["op_code"] = op_code;
		KString val = KString(exp + start, nth - start);
		if ((val.find('.') == KString::npos) && (val.find('f') == KString::npos) && (val.find('e') == KString::npos)) {
			pNode->m_propDict["op_aux"] = std::stoll(val);
		}
		else {
			pNode->m_propDict["op_aux"] = std::stof(val);
		}
	}
	else {
		KInt op_code;
		KString sOpCode = KString(exp + start, nth - start);
		KString sOpAux;

		if (sOpCode == "dict") op_code = (KInt)exp_op::dict;
		else if (sOpCode == "term") op_code = (KInt)exp_op::term;
		else if (sOpCode == "hidden") op_code = (KInt)exp_op::hidden;
		else if (sOpCode == "add") op_code = (KInt)exp_op::add;
		else if (sOpCode == "sub") op_code = (KInt)exp_op::sub;
		else if (sOpCode == "mult") op_code = (KInt)exp_op::mult;
		else if (sOpCode == "div") op_code = (KInt)exp_op::div;
		else if (sOpCode == "equal") op_code = (KInt)exp_op::equal;
		else if (sOpCode == "equal_col") op_code = (KInt)exp_op::equal_col;
		else if (sOpCode == "and") op_code = (KInt)exp_op::_and;
		else if (sOpCode == "or") op_code = (KInt)exp_op::_or;
		else if (sOpCode == "gt") op_code = (KInt)exp_op::gt;
		else if (sOpCode == "lt") op_code = (KInt)exp_op::lt;
		else if (sOpCode == "ge") op_code = (KInt)exp_op::ge;
		else if (sOpCode == "le") op_code = (KInt)exp_op::le;
		else if (sOpCode == "exp") op_code = (KInt)exp_op::exp;
		else if (sOpCode == "log") op_code = (KInt)exp_op::log;
		else if (sOpCode == "mean") op_code = (KInt)exp_op::mean;
		else if (sOpCode == "sum") op_code = (KInt)exp_op::sum;
		else if (sOpCode == "sqrt") op_code = (KInt)exp_op::sqrt;
		else if (sOpCode == "square") op_code = (KInt)exp_op::square;
		else if (sOpCode == "argmax") op_code = (KInt)exp_op::argmax;
		else if (sOpCode == "max") op_code = (KInt)exp_op::max;
		else if (sOpCode == "sigmoid") op_code = (KInt)exp_op::sigmoid;
		else if (sOpCode == "softmax") op_code = (KInt)exp_op::softmax;
		else if (sOpCode == "subvector") op_code = (KInt)exp_op::subvector;
		else if (sOpCode == "filter") op_code = (KInt)exp_op::filter;
		else if (sOpCode == "sigmoid_cross_entropy_with_logits") op_code = (KInt)exp_op::sigmoid_cross_entropy_with_logits;
		else if (sOpCode == "softmax_cross_entropy_with_logits") op_code = (KInt)exp_op::softmax_cross_entropy_with_logits;
		else if (sOpCode == "softmax_cross_entropy_with_logits_idx") op_code = (KInt)exp_op::softmax_cross_entropy_with_logits_idx;
		else if (sOpCode[0] == '\'') {
			if (sOpCode[sOpCode.size()-1] != '\'') throw KaiException(KERR_EXP_UNKNOWN_OPERATOR, sOpCode);
			op_code = (KInt)exp_op::string;
			sOpAux = sOpCode.substr(1, sOpCode.size() - 2);
		}
		else throw KaiException(KERR_EXP_UNKNOWN_OPERATOR, sOpCode);

		pNode->m_propDict["op_str"] = sOpCode;
		pNode->m_propDict["op_code"] = op_code;
		pNode->m_propDict["op_aux"] = sOpAux;
	}
}

void KaiHungarianExpression::m_seek_operands(KaiExpression* pNode, const char* exp, KInt& nth) {
	if (exp[nth] != '(') return;

	nth++;

	while (true) {
		KaiExpression* pChild = new KaiExpression(m_pSession, KaiDict());

		pChild->m_propDict["builtin"] = "child";
		pChild->m_propDict["desc"] = pChild->desc();

		m_parse_subexp(pChild, exp, nth);

		pNode->m_operands.push_back(pChild);
		pNode->bind(pChild, "child", false, false);

		const char ch = exp[nth++];
		if (ch == ')') break;
		else if (ch != ',') {
			KaiException(KERR_EXP_ILLFORMED_OPERANDS, KString(exp), KString(exp + nth));
		}
	}
}
*/