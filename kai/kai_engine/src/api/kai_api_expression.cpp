/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"
#include "../components/kexpression.h"

KAI_API KRetCode KAI_Expression_get_builtin_names(KHSession hSession, KStrList* pslNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslNames);

	KaiExpression::GetBuiltinNames(pslNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Expression_create(KHSession hSession, KHExpression* phExpression, KString sBuiltin, KaiDict kwArgs) {
	SESSION_OPEN();
	POINTER_CHECK(phExpression);

	KaiExpression* pExpression = KaiExpression::CreateInstance(pSession, sBuiltin, kwArgs);
	*phExpression = (KHExpression)pExpression;

	SESSION_CLOSE();
}

/*
KAI_API KRetCode KAI_Exp_get_operator(KHExpression hExpression, KString * psOpCode, KString * psOpAux) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiExpression, hExpression, pExpression);
	POINTER_CHECK(psOpCode);

	*psOpCode = pExpression->get_op_code();
	if (psOpAux) *psOpAux = pExpression->get_op_aux();

	NO_SESSION_CLOSE();
}

extern "C" KAI_API KRetCode KAI_Exp_get_operand_count(KHExpression hExpression, KInt * pnOpndCnt) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiExpression, hExpression, pExpression);
	POINTER_CHECK(pnOpndCnt);

	*pnOpndCnt = pExpression->get_operand_count();

	NO_SESSION_CLOSE();
}

extern "C" KAI_API KRetCode KAI_Exp_get_nth_operand(KHExpression hExpression, KInt nth, KHExpression * phOperand) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiExpression, hExpression, pExpression);
	POINTER_CHECK(phOperand);

	*phOperand = (KHExpression) pExpression->get_nth_operand(nth);

	NO_SESSION_CLOSE();
}
*/
