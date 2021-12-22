/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../components/klayer.h"

KAI_API KRetCode KAI_Layer_get_builtin_names(KHSession hSession, KStrList* pslNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslNames);

	KaiLayer::GetBuiltinNames(pslNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Layer_create(KHSession hSession, KHLayer* phLayer, KString sBuiltin, KaiDict kwArgs) {
	SESSION_OPEN();
	POINTER_CHECK(phLayer);

	KaiLayer* pLayer = KaiLayer::CreateInstance(pSession, sBuiltin, kwArgs);
	*phLayer = (KHLayer)pLayer;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Layer_close(KHSession hSession, KHLayer hLayer) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiLayer, hLayer, pLayer);

	pLayer->close();

	SESSION_CLOSE();
}
