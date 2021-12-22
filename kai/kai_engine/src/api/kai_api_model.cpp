/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../components/kmodel.h"

KAI_API KRetCode KAI_Model_get_builtin_names(KHSession hSession, KStrList* pslNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslNames);

	KaiModel::GetBuiltinNames(pslNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_create(KHSession hSession, KHModel* phModel, KString sBuiltin, KaiDict kwArgs) {
	SESSION_OPEN();
	POINTER_CHECK(phModel);

	KaiModel* pModel = KaiModel::CreateInstance(pSession, sBuiltin, kwArgs);
	*phModel = (KHModel)pModel;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_close(KHSession hSession, KHModel hModel) {
	SESSION_OPEN();
	HANDLE_OPEN_OR_NULL(KaiModel, hModel, pModel);

	pModel->close();

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_train(KHModel hModel, KaiDict kwArgs, KBool bAsync) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModel, hModel, pModel);

	pModel->train(kwArgs, bAsync);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_test(KHModel hModel, KaiDict kwArgs, KBool bAsync) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModel, hModel, pModel);

	pModel->test(kwArgs, bAsync);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_visualize(KHModel hModel, KaiDict kwArgs, KBool bAsync) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModel, hModel, pModel);

	pModel->visualize(kwArgs, bAsync);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_predict(KHModel hModel, KaiDict kwArgs, KaiList* pdResult) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModel, hModel, pModel);
	POINTER_CHECK(pdResult);

	*pdResult = pModel->predict(kwArgs);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_get_trained_epoch_count(KHModel hModel, KInt* pnEpochCount) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModel, hModel, pModel);
	POINTER_CHECK(pnEpochCount);

	*pnEpochCount = pModel->get_trained_epoch_count();

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_get_instance(KHModel hModel, KHModelInstance* phModelInst, KaiDict kwArgs) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModel, hModel, pModel);
	POINTER_CHECK(phModelInst);

	*phModelInst = (KHModelInstance)pModel->get_instance(kwArgs);

	NO_SESSION_CLOSE();
}
