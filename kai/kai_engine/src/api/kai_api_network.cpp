/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../components/knetwork.h"
#include "../components/klayer.h"

KAI_API KRetCode KAI_Network_get_builtin_names(KHSession hSession, KStrList* pslNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslNames);

	KaiNetwork::GetBuiltinNames(pslNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_create(KHSession hSession, KHNetwork* phNetwork, KString sBuiltin, KaiDict kwArgs) {
	SESSION_OPEN();
	POINTER_CHECK(phNetwork);

	KaiNetwork* pNetwork = KaiNetwork::CreateInstance(pSession, sBuiltin, kwArgs); // 목록에 저장해 close 때 삭제 처리 빼먹지 않게...
	*phNetwork = (KHNetwork)pNetwork;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_close(KHSession hSession, KHNetwork hNetwork) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(KaiNetwork, hNetwork, pNetwork);

	//pSession->unregist(pNetwork);
	pNetwork->close();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_append_layer(KHSession hSession, KHNetwork hNetwork, KHLayer hLayer) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);
	HANDLE_OPEN(KaiLayer, hLayer, pLayer);

	pNetwork->append_layer(pLayer);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_append_named_layer(KHSession hSession, KHNetwork hNetwork, KString sLayerName, KaiDict kwArgs) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);

	pNetwork->append_named_layer(sLayerName, kwArgs);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_regist_macro(KHSession hSession, KHNetwork hNetwork, KString sMacroName) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);

	pSession->regist_macro(pNetwork, sMacroName);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_append_custom_layer(KHSession hSession, KHNetwork hNetwork, KString sLayerName, KaiDict kwArgs) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);

	pNetwork->append_custom_layer(sLayerName, kwArgs);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_append_subnet(KHSession hSession, KHNetwork hNetwork, KHNetwork hSubnet) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);
	HANDLE_OPEN(KaiNetwork, hSubnet, pSubnet);

	pNetwork->append_subnet(pSubnet);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_get_layer_count(KHSession hSession, KHNetwork hNetwork, KInt* pnLayerCount) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);
	POINTER_CHECK(pnLayerCount);

	*pnLayerCount = pNetwork->get_layer_count();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Network_get_nth_layer(KHSession hSession, KHNetwork hNetwork, KInt nth, KHLayer* phLayer) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiNetwork, hNetwork, pNetwork);
	POINTER_CHECK(phLayer);

	*phLayer = (KHLayer) pNetwork->get_nth_layer(nth);

	SESSION_CLOSE();
}

