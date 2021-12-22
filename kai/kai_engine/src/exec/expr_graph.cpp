/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../exec/expr_graph.h"
#include "../exec/exec_context.h"
#include "../math/karray.h"
#include "../math/kmath.h"
#include "../utils/kutil.h"

// 리커젼 방지 위해 어디에서 어떻게 오류로 끊어내야 할지 실험으로 확인할 것
KaiDict KaiLossExpRoot::ms_args;

KaiLossExpRoot::KaiLossExpRoot(KBool3 need_grad, KaiDict exp_info) : KaiObject(Ken_object_type::_exp_node) {
	m_pNode = NULL;
	m_need_grad = need_grad;
	m_exp_info = exp_info;

	if (m_need_grad != KBool3::unknown) m_pNode = new KaiLossExpNode();
}

KaiLossExpRoot::~KaiLossExpRoot() {
	delete m_pNode;
}

void KaiLossExpRoot::build_graph(KaiDict loss_graph, KBool bLoss) {
	if (m_pNode) return;

	m_pNode = new KaiLossExpNode(m_exp_info, loss_graph, bLoss, &m_need_grad);

	if (!bLoss) m_need_grad = KBool3::off;
}

KBool3 KaiLossExpRoot::get_need_grad(KaiDict loss_graph, KBool bLoss) {
	if (m_need_grad == KBool3::unknown) {
		build_graph(loss_graph, bLoss);
	}

	return m_need_grad;
}

void KaiLossExpRoot::set_value(KaiValue value) {
	m_pNode->set_node_value(value);
}

void KaiLossExpRoot::reset() {
	m_pNode->reset();
}

void KaiLossExpRoot::set_grad(KaiArray<KFloat> grad) {
	m_pNode->set_node_grad(grad);
}

KaiValue KaiLossExpRoot::evaluate_value(KaiMath* pMath) {
	return m_pNode->evaluate_node_value(pMath);
}

KaiArray<KFloat> KaiLossExpRoot::evaluate_grad(KaiMath* pMath) {
	return m_pNode->evaluate_node_grad(pMath);
}

void KaiLossExpRoot::set_arg(KString sKey, KaiValue value) {
	ms_args[sKey] = value;
}

KaiValue KaiLossExpRoot::get_arg(KString sKey) {
	return ms_args[sKey];
}

KaiLossExpNode::KaiLossExpNode() {
	m_forLink = NULL;
	m_grad_set = false;
}

KaiLossExpNode::KaiLossExpNode(KaiDict exp_info, KaiDict loss_graph, KBool bLoss, KBool3* p_need_grad) {
	m_forLink = new KaiLossExpForLink(this, exp_info, loss_graph, bLoss, p_need_grad);
	m_grad_set = false;
}

KaiLossExpNode::~KaiLossExpNode() {
	delete m_forLink;

	for (auto& it : m_backLinks) {
		delete it;
	}
}

void KaiLossExpNode::addBackLink(KaiLossExpBackLink* pBackLink) {
	m_backLinks.push_back(pBackLink);
}

void KaiLossExpNode::reset() {
	if (m_value.type() == Ken_value_type::none) return;

	m_grad_set = false;
	m_value = KaiValue();
	m_grad = KaiArray<KFloat>();
	if (m_forLink) m_forLink->reset();
}

void KaiLossExpNode::set_node_value(KaiValue value) {
	m_value = value;
}

void KaiLossExpNode::set_node_grad(KaiArray<KFloat> grad) {
	m_grad = grad;
	m_grad_set = true;
}

KaiValue KaiLossExpNode::evaluate_node_value(KaiMath* pMath) {
	if (m_value.type() == Ken_value_type::none) {
		m_value = m_forLink->evaluate_link_value(pMath);
	}
	
	return m_value;
}

KaiArray<KFloat> KaiLossExpNode::evaluate_node_value_as_farray(KaiMath* pMath) {
	KaiArray<KFloat> farr;
	KaiValue value = evaluate_node_value(pMath);
	if (value.type() == Ken_value_type::kint) farr = pMath->ones(KaiShape{ 1 }, (KFloat)(KInt)value);
	else if (value.type() == Ken_value_type::kfloat) farr = pMath->ones(KaiShape{ 1 }, (KFloat)value);
	else farr = FARRAY(value);

	return farr;
}

KaiArray<KInt> KaiLossExpNode::evaluate_node_value_as_narray(KaiMath* pMath) {
	KaiArray<KInt> narr;
	KaiValue value = evaluate_node_value(pMath);
	if (value.type() == Ken_value_type::kint) throw KaiException(KERR_UNIMPEMENTED_YET);
	else if (value.type() == Ken_value_type::kfloat) throw KaiException(KERR_UNIMPEMENTED_YET);
	else narr = NARRAY(value);

	return narr;
}

KaiArray<KFloat> KaiLossExpNode::evaluate_node_grad(KaiMath* pMath) {
	if (m_backLinks.size() == 0) return m_grad;
	if (m_grad_set) return m_grad;

	for (auto& it : m_backLinks) {
		KaiLossExpBackLink* pLink = it;
		KaiArray<KFloat> value = FARRAY(m_value);
		it->accumulate_link_grad(m_grad, value, pMath);
	}

	m_grad_set = true;

	return m_grad;
}

KaiLossExpForLink::KaiLossExpForLink(KaiLossExpNode* pParent, KaiDict exp_info, KaiDict loss_graph, KBool bLoss, KBool3* p_need_grad) {
	m_op_code = (exp_op)(KInt)exp_info["op_code"];
	m_op_aux = exp_info["op_aux"];

	if (m_op_code == exp_op::feed) {
		KString exp_name = m_op_aux;
		
		if (exp_name == "@est") exp_name = "@est:#default";
		else if (exp_name == "@ans") exp_name = "@ans:#default";

		if (loss_graph.find(exp_name) != loss_graph.end()) {
			KaiLossExpRoot* pDest = (KaiLossExpRoot*)(KaiObject*)loss_graph[exp_name];

			*p_need_grad = pDest->get_need_grad(loss_graph, bLoss);

			m_operands.push_back(pDest->get_root());

			if (*p_need_grad == KBool3::on) {
				KaiLossExpBackLink* pBackLink = new KaiLossExpBackLink(this, pParent);
				pDest->get_root()->addBackLink(pBackLink);
			}
		}
		else {
			throw KaiException(KERR_FEEDING_TERM_NOT_FOUND_IN_EXPRESSION, exp_name);
		}
	}
	else if (m_op_code == exp_op::subexp) {
		KString exp_name = m_op_aux;

		if (loss_graph.find(exp_name) != loss_graph.end()) {
			KaiLossExpRoot* pDest = (KaiLossExpRoot*)(KaiObject*)loss_graph[exp_name];

			*p_need_grad = pDest->get_need_grad(loss_graph, bLoss);

			m_operands.push_back(pDest->get_root());

			if (*p_need_grad == KBool3::on) {
				KaiLossExpBackLink* pBackLink = new KaiLossExpBackLink(this, pParent);
				pDest->get_root()->addBackLink(pBackLink);
			}
		}
		else {
			throw KaiException(KERR_SUBEXP_TERM_NOT_FOUND_IN_EXPRESSION, exp_name);
		}
	}
	else if (exp_info.find("operands") != exp_info.end()) {
		*p_need_grad = KBool3::off;

		KaiList operands = exp_info["operands"];
		KBool3 need_grad;
		for (KInt n = 0; n < operands.size(); n++) {
			KaiDict opnd_info = operands[n];
			KaiLossExpNode* pChild = new KaiLossExpNode(opnd_info, loss_graph, bLoss, &need_grad);
			m_operands.push_back(pChild);

			if (need_grad == KBool3::on) {
				KaiLossExpBackLink* pBackLink = new KaiLossExpBackLink(this, pParent, n);
				pChild->addBackLink(pBackLink);
				*p_need_grad = KBool3::on;
			}
		}
	}
}

KaiLossExpForLink::~KaiLossExpForLink() {
	if (m_op_code == exp_op::feed) return;
	if (m_op_code == exp_op::subexp) return;

	for (auto& it : m_operands) {
		delete it;
	}
}

void KaiLossExpForLink::reset() {
	for (auto& it : m_operands) {
		it->reset();
	}
}

KaiValue KaiLossExpForLink::evaluate_link_value(KaiMath* pMath) {
	KaiValue value;
	KaiArray<KFloat> farr1, farr2;
	KaiArray<KInt> narr1, narr2;
	KInt nFrom, nCount;

	switch (m_op_code) {
	case exp_op::feed:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		value = m_operands[0]->evaluate_node_value(pMath);
		break;
	case exp_op::subexp:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		value = m_operands[0]->evaluate_node_value(pMath);
		break;
	case exp_op::constant:
		value = m_op_aux;
		break;
	case exp_op::arg:
		value = KaiLossExpRoot::get_arg((KString)m_op_aux);
		break;
	case exp_op::add:
	case exp_op::mult:
		if (m_operands.size() < 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		farr1 = m_operands[0]->evaluate_node_value_as_farray(pMath);
		farr2 = m_operands[1]->evaluate_node_value_as_farray(pMath);

		farr1 = pMath->eval_binary_op(m_op_code, farr1, farr2);

		for (KInt n = 2; n < (KInt)m_operands.size(); n++) {
			farr2 = m_operands[n]->evaluate_node_value_as_farray(pMath);
			farr1 = pMath->eval_binary_op(m_op_code, farr1, farr2);
		}

		value = farr1.get_core();
		break;
	case exp_op::vstack:
	{
		if (m_operands.size() < 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		KaiList arrays;

		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		arrays.push_back(farr1.get_core());

		KaiShape sshape = farr1.shape().copy();
		KInt nRows = sshape.total_size() / sshape[-1];

		for (KInt n = 1; n < (KInt)m_operands.size(); n++) {
			farr1 = FARRAY(m_operands[n]->evaluate_node_value(pMath));
			arrays.push_back(farr1.get_core());

			KaiShape oshape = farr1.shape().copy();
			KInt nCols = oshape[-1];
			if (oshape.total_size() / nCols != nRows) throw KaiException(KERR_BAD_SHAPE_OPERANDS_FOR_VSTACK);
			sshape[-1] = sshape[-1] + nCols;
		}

		KInt nFrom = 0;
		farr2 = pMath->zeros(sshape);

		for (KInt n = 0; n < (KInt)m_operands.size(); n++) {
			farr1 = FARRAY(arrays[n]);
			pMath->vstack(farr2, farr1, nFrom);
			nFrom += farr1.shape()[-1];
		}

		value = farr2.get_core();
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
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		farr1 = m_operands[0]->evaluate_node_value_as_farray(pMath);
		farr2 = m_operands[1]->evaluate_node_value_as_farray(pMath);

		value = pMath->eval_binary_op(m_op_code, farr1, farr2).get_core();
		break;
	case exp_op::softmax_cross_entropy_with_logits:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		farr2 = FARRAY(m_operands[1]->evaluate_node_value(pMath));
		value = pMath->softmax_cross_entropy_with_logits(farr1, farr2).get_core();
		break;
	case exp_op::softmax_cross_entropy_with_logits_idx:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		narr2 = NARRAY(m_operands[1]->evaluate_node_value(pMath));
		value = pMath->softmax_cross_entropy_with_logits_idx(farr1, narr2).get_core();
		break;
	case exp_op::equal_col:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		farr2 = FARRAY(m_operands[1]->evaluate_node_value(pMath));
		value = pMath->equal_col(farr1, farr2).get_core();
		break;
	case exp_op::max_col:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->max_col(farr1).get_core();
		break;
	case exp_op::argmax:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->argmax(farr1).get_core();
		break;
	case exp_op::max:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->max(farr1).get_core();
		break;
	case exp_op::mean:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->mean(farr1).get_core();
		break;
	case exp_op::sum:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->sum(farr1).get_core();
		break;
	case exp_op::sqrt:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = m_operands[0]->evaluate_node_value_as_farray(pMath);
		value = pMath->sqrt(farr1).get_core();
		break;
	case exp_op::square:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = m_operands[0]->evaluate_node_value_as_farray(pMath);
		value = pMath->square(farr1).get_core();
		break;
	case exp_op::log:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = m_operands[0]->evaluate_node_value_as_farray(pMath);
		value = pMath->log(farr1).get_core();
		break;
	case exp_op::exp:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = m_operands[0]->evaluate_node_value_as_farray(pMath);
		value = pMath->exp(farr1).get_core();
		break;
	case exp_op::sigmoid:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = m_operands[0]->evaluate_node_value_as_farray(pMath);
		value = pMath->sigmoid(farr1).get_core();
		break;
	case exp_op::softmax:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->softmax(farr1).get_core();
		break;
	case exp_op::subvector:
		if (m_operands.size() != 3) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		nFrom = m_operands[1]->evaluate_node_value(pMath);
		nCount = m_operands[2]->evaluate_node_value(pMath);
		value = pMath->get_subvector(farr1, nFrom, nCount).get_core();
		break;
	case exp_op::iou:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		farr2 = FARRAY(m_operands[1]->evaluate_node_value(pMath));
		value = pMath->iou_yolo(farr1, farr2).get_core();
		break;
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}

	return value;
}

KaiLossExpBackLink::KaiLossExpBackLink(KaiLossExpForLink* linkInfo, KaiLossExpNode* pDest, KInt nth) {
	m_linkInfo = linkInfo;
	m_pDest = pDest;
	m_nth = nth;
}

KaiLossExpBackLink::~KaiLossExpBackLink() {
}

void KaiLossExpBackLink::accumulate_link_grad(KaiArray<KFloat>& grad_acc, KaiArray<KFloat> value, KaiMath* pMath) {
	KaiArray<KFloat> grad, grad_y;
	KaiArray<KFloat> farr1, farr2;
	KaiArray<KInt> narr1, narr2;
	KFloat fn;
	KInt nFrom, nCount;

	switch (m_linkInfo->m_op_code) {
	case exp_op::gt:
	case exp_op::lt:
	case exp_op::ge:
	case exp_op::le:
	case exp_op::equal:
	case exp_op::_and:
	case exp_op::_or:
		// 비교연산자, 논리연산자는 미분불가이며 처리 대상 선택을 위한 조건으로만 사용된다.
		// 선택된 대상의 역전파 처리 과정에서 필요한 처리가 이루어진다고 보고 이들 연산자의 기울기 처리는 하지 않는다.
		return;
	}
	
	// 수식그래프 역추적 과정에서 비교연산자, 논리연산자 등으로 인해 미분불가 처리가 나온 경우 해당 역전파 링크는 무시한다.
	grad_y = m_pDest->evaluate_node_grad(pMath);
	if (grad_y.is_empty()) return;

	switch (m_linkInfo->m_op_code) {
	case exp_op::subvector:
		nFrom = m_linkInfo->m_operands[1]->evaluate_node_value(pMath);
		nCount = m_linkInfo->m_operands[2]->evaluate_node_value(pMath);
		if (grad_acc.is_empty()) grad_acc = pMath->zeros(value.shape());
		pMath->get_subvector_derv_acc(grad_acc, grad_y, nFrom, nCount);
		return;
	case exp_op::feed:
		grad = grad_y;
		break;
	case exp_op::subexp:
		grad = grad_y;
		break;
	case exp_op::sub:
		grad = (m_nth == 0) ? grad_y : pMath->minus(grad_y);
		break;
	case exp_op::square:
		grad = pMath->mul(pMath->mul(grad_y, value), 2.0f);
		break;
	case exp_op::mean:
		assert(grad_y.total_size() == 1);
		fn = pMath->fetch(grad_y) / (float)value.total_size();
		grad = pMath->ones(value.shape(), fn);
		break;
	case exp_op::add:
		grad = grad_y;
		break;
	case exp_op::mult:
		grad = grad_y;
		for (KInt n = 0; n < (KInt)m_linkInfo->m_operands.size(); n++) {
			if (n == m_nth) continue;
			farr1 = m_linkInfo->m_operands[n]->evaluate_node_value_as_farray(pMath);
			grad = pMath->eval_binary_op(exp_op::mult, grad, farr1);
		}
		farr1 = m_linkInfo->m_operands[m_nth]->evaluate_node_value_as_farray(pMath);
		grad = grad.reshape(farr1.shape());
		break;
	case exp_op::div:
	case exp_op::sigmoid_cross_entropy_with_logits:
	case exp_op::softmax_cross_entropy_with_logits:
		farr1 = m_linkInfo->m_operands[0]->evaluate_node_value_as_farray(pMath);
		farr2 = m_linkInfo->m_operands[1]->evaluate_node_value_as_farray(pMath);
		if (m_nth == 0) {
			grad = m_grad1_on_binary_op(m_linkInfo->m_op_code, pMath, grad_y, farr1, farr2);
		}
		else {
			grad = m_grad2_on_binary_op(m_linkInfo->m_op_code, pMath, grad_y, farr1, farr2);
		}
		break;
	case exp_op::softmax_cross_entropy_with_logits_idx:
		farr1 = m_linkInfo->m_operands[0]->evaluate_node_value_as_farray(pMath);
		narr2 = m_linkInfo->m_operands[1]->evaluate_node_value_as_narray(pMath);
		if (m_nth == 0) {
			grad = pMath->mul(grad_y, pMath->softmax_cross_entropy_with_logits_idx_derv(farr1, narr2)); 
		}
		else {
			throw KaiException(KERR_UNIMPEMENTED_YET);
		}
		break;
	case exp_op::sigmoid:
		farr1 = m_pDest->evaluate_node_value_as_farray(pMath);
		grad = pMath->sigmoid_derv_grad(grad_y, farr1);
		break;
	case exp_op::vstack:
	{
		KInt nStart = 0;
		for (KInt n = 0; n < m_nth; n++) {
			farr1 = m_linkInfo->m_operands[n]->evaluate_node_value_as_farray(pMath);
			KaiShape oshape = farr1.shape();
			nStart += oshape[-1];
		}

		farr1 = m_linkInfo->m_operands[m_nth]->evaluate_node_value_as_farray(pMath);
		KaiShape oshape = farr1.shape();
		KInt nCount = oshape[-1];

		grad = pMath->vstack_grad(grad_y, nStart, nCount);
	}
	break;
	case exp_op::max_col:
		farr1 = m_linkInfo->m_operands[0]->evaluate_node_value_as_farray(pMath);
		grad = pMath->max_col_grad(grad_y, farr1);
		break;
	case exp_op::iou:
		throw KaiException(KERR_UNIMPEMENTED_YET);
		/*
		farr1 = m_linkInfo->m_operands[0]->evaluate_node_value_as_farray(pMath);
		farr2 = m_linkInfo->m_operands[1]->evaluate_node_value_as_farray(pMath);
		grad = pMath->iou_yolo_grad(grad_y, farr1, farr2, m_nth);
		*/
		break;
	case exp_op::sum:
		farr1 = m_linkInfo->m_operands[0]->evaluate_node_value_as_farray(pMath);
		grad = pMath->sum_grad(grad_y, farr1);
		break;
		/*
	case exp_op::constant:
		value = m_op_aux;
		break;
	case exp_op::arg:
		value = KaiLossExpRoot::get_arg((KString)m_op_aux);
		break;
	case exp_op::add:
	case exp_op::mult:
		if (m_operands.size() < 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		farr2 = FARRAY(m_operands[1]->evaluate_node_value(pMath));

		farr1 = pMath->eval_binary_op(m_op_code, farr1, farr2);

		for (KInt n = 2; n < (KInt)m_operands.size(); n++) {
			farr2 = FARRAY(m_operands[n]->evaluate_node_value(pMath));
			farr1 = pMath->eval_binary_op(m_op_code, farr1, farr2);
		}

		value = farr1.get_core();
		break;
	case exp_op::vstack:
	{
		if (m_operands.size() < 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		KaiList arrays;

		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		arrays.push_back(farr1.get_core());

		KaiShape sshape = farr1.shape().copy();
		KInt nRows = sshape.total_size() / sshape[-1];

		for (KInt n = 1; n < (KInt)m_operands.size(); n++) {
			farr1 = FARRAY(m_operands[n]->evaluate_node_value(pMath));
			arrays.push_back(farr1.get_core());

			KaiShape oshape = farr1.shape().copy();
			KInt nCols = oshape[-1];
			if (oshape.total_size() / nCols != nRows) throw KaiException(KERR_BAD_SHAPE_OPERANDS_FOR_VSTACK);
			sshape[-1] = sshape[-1] + nCols;
		}

		KInt nFrom = 0;
		farr2 = pMath->zeros(sshape);

		for (KInt n = 0; n < (KInt)m_operands.size(); n++) {
			farr1 = FARRAY(arrays[n]);
			pMath->vstack(farr2, farr1, nFrom);
			nFrom += farr1.shape()[-1];
		}

		value = farr2.get_core();
	}
	break;
	case exp_op::div:
	case exp_op::gt:
	case exp_op::lt:
	case exp_op::ge:
	case exp_op::le:
	case exp_op::equal:
	case exp_op::_and:
	case exp_op::_or:
	case exp_op::sigmoid_cross_entropy_with_logits:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);

		value = m_operands[0]->evaluate_node_value(pMath);
		if (value.type() == Ken_value_type::kint) farr1 = pMath->ones(KaiShape{ 1 }, (KFloat)(KInt)value);
		else if (value.type() == Ken_value_type::kfloat) farr1 = pMath->ones(KaiShape{ 1 }, (KFloat)value);
		else farr1 = FARRAY(value);

		value = m_operands[1]->evaluate_node_value(pMath);
		if (value.type() == Ken_value_type::kint) farr2 = pMath->ones(KaiShape{ 1 }, (KFloat)(KInt)value);
		else if (value.type() == Ken_value_type::kfloat) farr2 = pMath->ones(KaiShape{ 1 }, (KFloat)value);
		else farr2 = FARRAY(value);

		value = pMath->eval_binary_op(m_op_code, farr1, farr2).get_core();
		break;
	case exp_op::softmax_cross_entropy_with_logits:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		farr2 = FARRAY(m_operands[1]->evaluate_node_value(pMath));
		value = pMath->softmax_cross_entropy_with_logits(farr1, farr2).get_core();
		break;
	case exp_op::equal_col:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		farr2 = FARRAY(m_operands[1]->evaluate_node_value(pMath));
		value = pMath->equal_col(farr1, farr2).get_core();
		break;
	case exp_op::max_col:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->max_col(farr1).get_core();
		break;
	case exp_op::argmax:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->argmax(farr1).get_core();
		break;
	case exp_op::max:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->max(farr1).get_core();
		break;
	case exp_op::mean:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->mean(farr1).get_core();
		break;
	case exp_op::sum:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->sum(farr1).get_core();
		break;
	case exp_op::sqrt:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->sqrt(farr1).get_core();
		break;
	case exp_op::square:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->square(farr1).get_core();
		break;
	case exp_op::log:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->log(farr1).get_core();
		break;
	case exp_op::exp:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->exp(farr1).get_core();
		break;
	case exp_op::softmax:
		if (m_operands.size() != 1) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		value = pMath->softmax(farr1).get_core();
		break;
	case exp_op::iou:
		if (m_operands.size() != 2) throw KaiException(KERR_INPROPER_NUMBER_OF_OPERANDS);
		farr1 = FARRAY(m_operands[0]->evaluate_node_value(pMath));
		farr2 = FARRAY(m_operands[1]->evaluate_node_value(pMath));
		value = pMath->iou_yolo(farr1, farr2).get_core();
		*/
		break;
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case" + std::to_string((KInt)m_linkInfo->m_op_code));
	}

	if (!grad.is_empty()) {
		assert(grad.shape() == value.shape());

		if (grad_acc.is_empty()) grad_acc = grad;
		else pMath->add_on(grad_acc, grad);
	}
}

KaiArray<KFloat> KaiLossExpBackLink::m_grad1_on_binary_op(exp_op op_code, KaiMath* pMath, KaiArray<KFloat> grad1, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	switch (op_code) {
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
	case exp_op::le:
	case exp_op::lt:
	case exp_op::ge:
	case exp_op::gt:
		return pMath->zeros(arr1.shape());
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}
}

KaiArray<KFloat> KaiLossExpBackLink::m_grad2_on_binary_op(exp_op op_code, KaiMath* pMath, KaiArray<KFloat> grad2, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	switch (op_code) {
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
	case exp_op::le:
	case exp_op::lt:
	case exp_op::ge:
	case exp_op::gt:
		return pMath->zeros(arr2.shape());
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}
}
