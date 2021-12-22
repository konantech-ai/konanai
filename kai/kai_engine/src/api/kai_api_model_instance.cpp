/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../components/kmodel.h"

KAI_API KRetCode KAI_Model_Instance_train(KHModelInstance hModelInst, KaiDict kwArgs, KBool bAsync) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModelInstance, hModelInst, pModelInst);

	pModelInst->train(kwArgs, bAsync);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_Instance_test(KHModelInstance hModelInst, KaiDict kwArgs, KBool bAsync) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModelInstance, hModelInst, pModelInst);

	pModelInst->test(kwArgs, bAsync);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_Instance_visualize(KHModelInstance hModelInst, KaiDict kwArgs, KBool bAsync) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModelInstance, hModelInst, pModelInst);

	pModelInst->visualize(kwArgs, bAsync);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_Instance_predict(KHModelInstance hModelInst, KaiDict kwArgs, KaiList* pdResult) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModelInstance, hModelInst, pModelInst);
	POINTER_CHECK(pdResult);

	*pdResult = pModelInst->predict(kwArgs);

	NO_SESSION_CLOSE();
}

KAI_API KRetCode KAI_Model_Instance_get_trained_epoch_count(KHModelInstance hModelInst, KInt* pnEpochCount) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiModelInstance, hModelInst, pModelInst);
	POINTER_CHECK(pnEpochCount);

	*pnEpochCount = pModelInst->get_trained_epoch_count();

	NO_SESSION_CLOSE();
}
