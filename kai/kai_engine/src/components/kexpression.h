/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "component.h"
#include "../math/karray.h"
#include "../exec/expr_graph.h"

class KaiMath;

class KaiExpression : public KaiComponent {
public:
	KaiExpression(KaiSession* pSession, KaiDict kwArgs);
	KaiExpression(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	virtual ~KaiExpression();

	Ken_object_type get_type() { return Ken_object_type::expression; }

	static KaiExpression* HandleToPointer(KHObject hObject, KBool mayNull = false);
	static KaiExpression* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiExpression* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	//KString get_op_code();
	//KString get_op_aux();
	//KInt get_operand_count();
	//KaiExpression* get_nth_operand(KInt nth);

	static KaiDict evaluate_with_grad(KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KaiDict& grads, KInt mb_size);

	static KaiDict evaluate(KString sExpName, KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KBool bScalar, KInt mb_size);
	static KaiDict postproc(KaiDict funcInfo, KaiExecContext* pContext, KaiDict var_dict);

	//static KaiValue evaluate_value_with_grad(KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KaiDict& grads, KaiDict var_dict = KaiDict());
	static KaiValue evaluate_value(KaiDict funcInfo, KaiDict xs, KaiDict ys, KaiDict outs, KaiExecContext* pContext, KaiDict var_dict = KaiDict());

	KString desc();

protected:
	friend class KaiHungarianExpression;

	static int ms_checkCode;
	static KStrList ms_builtin;

	static KaiDict ms_create_exp_graph(KaiDict funcInfo, KaiDict ys, KaiDict outs, KBool bLoss);
	static void ms_reset_exp_graph(KaiDict graph);
};

class KaiHungarianExpression : public KaiExpression {
public:
	KaiHungarianExpression(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiHungarianExpression();

protected:
	KaiDict m_parse_exp(KString sExp);
	KaiDict m_parse_subexp(const char* pExp, KInt& nth);
	KaiDict m_split_op_code(const char* pExp, KInt& nth);
	KaiList m_seek_operands(const char* pExp, KInt& nth);
};
