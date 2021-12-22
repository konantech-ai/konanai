/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../components/koptimizer.h"

KAI_API KRetCode KAI_Optimizer_get_builtin_names(KHSession hSession, KStrList* pslNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslNames);

	KaiOptimizer::GetBuiltinNames(pslNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Optimizer_create(KHSession hSession, KHOptimizer* phOptimizer, KString sBuiltin, KaiDict kwArgs) {
	SESSION_OPEN();
	POINTER_CHECK(phOptimizer);

	KaiOptimizer* pOptimizer = KaiOptimizer::CreateInstance(pSession, sBuiltin, kwArgs);
	*phOptimizer = (KHOptimizer)pOptimizer;

	SESSION_CLOSE();
}
