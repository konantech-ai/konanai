/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../components/component.h"
#include "../math/karray.h"

class KaiMath;

class KaiLossExpRoot;
class KaiLossExpNode;
class KaiLossExpForLink;
class KaiLossExpBackLink;
//class KaiExpDefNode;

class KaiLossExpRoot : public KaiObject {
public:
	KaiLossExpRoot(KBool3 need_grad, KaiDict exp_info=KaiDict());
	virtual ~KaiLossExpRoot();

	KString desc() { return "__exp__"; }

	void build_graph(KaiDict loss_graph, KBool bLoss);
	
	KaiLossExpNode* get_root() { return m_pNode; }
	KBool3 get_need_grad(KaiDict loss_graph, KBool bLoss);

	void reset();
	void set_value(KaiValue value);
	void set_grad(KaiArray<KFloat> grad);
	KaiValue evaluate_value(KaiMath* pMath);
	KaiArray<KFloat> evaluate_grad(KaiMath* pMath);

	static void set_arg(KString sKey, KaiValue value);
	static KaiValue get_arg(KString sKey);

public:
	KaiLossExpNode* m_pNode;
	KBool3 m_need_grad;
	KaiDict m_exp_info;

	static KaiDict ms_args;
};

class KaiLossExpNode {
public:
	KaiLossExpNode();
	KaiLossExpNode(KaiDict exp_info, KaiDict loss_graph, KBool bLoss, KBool3* p_need_grad);
	virtual ~KaiLossExpNode();

	void addBackLink(KaiLossExpBackLink* pBackLink);

	void reset();
	void set_node_value(KaiValue value);
	void set_node_grad(KaiArray<KFloat> grad);
	KaiValue evaluate_node_value(KaiMath* pMath);
	KaiArray<KFloat> evaluate_node_value_as_farray(KaiMath* pMath);
	KaiArray<KInt> evaluate_node_value_as_narray(KaiMath* pMath);
	KaiArray<KFloat> evaluate_node_grad(KaiMath* pMath);

protected:
	KaiLossExpForLink* m_forLink;
	vector< KaiLossExpBackLink*> m_backLinks;

	KaiValue m_value;
	KaiArray<KFloat> m_grad;
	KBool m_grad_set;
};

class KaiLossExpForLink {
public:
	KaiLossExpForLink(KaiLossExpNode* pParent, KaiDict exp_info, KaiDict loss_graph, KBool bLoss, KBool3* p_need_grad);
	virtual ~KaiLossExpForLink();

	void reset();
	KaiValue evaluate_link_value(KaiMath* pMath);

protected:
	exp_op m_op_code;
	KaiValue m_op_aux;
	vector< KaiLossExpNode*> m_operands;

	friend class KaiLossExpBackLink;
};

class KaiLossExpBackLink {
public:
	KaiLossExpBackLink(KaiLossExpForLink* linkInfo, KaiLossExpNode* pDest, KInt nth=0);
	virtual ~KaiLossExpBackLink();

	void accumulate_link_grad(KaiArray<KFloat>& grad, KaiArray<KFloat> value, KaiMath* pMath);

protected:
	KaiLossExpForLink* m_linkInfo;
	KaiLossExpNode* m_pDest;
	KInt m_nth;

protected:
	KaiArray<KFloat> m_grad1_on_binary_op(exp_op op_code, KaiMath* pMath, KaiArray<KFloat> grad1, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	KaiArray<KFloat> m_grad2_on_binary_op(exp_op op_code, KaiMath* pMath, KaiArray<KFloat> grad2, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
};
