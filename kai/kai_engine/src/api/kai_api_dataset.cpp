/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../components/kdataset.h"

KAI_API KRetCode KAI_Dataset_get_builtin_names(KHSession hSession, KStrList * pslNames) {
	SESSION_OPEN();
	POINTER_CHECK(pslNames);

	KaiDataset::GetBuiltinNames(pslNames);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Dataset_create(KHSession hSession, KHDataset* phDataset, KString sBuiltin, KaiDict kwArgs) {
	SESSION_OPEN();
	POINTER_CHECK(phDataset);

	KaiDataset* pDataset = KaiDataset::CreateInstance(pSession, sBuiltin, kwArgs); // 목록에 저장해 close 때 삭제 처리 빼먹지 않게...
	*phDataset = (KHDataset)pDataset;

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Dataset_read_file(KHDataset hDataset, KString sDataFilePath) {
	NO_SESSION_OPEN();
	NO_SESSION_HANDLE_OPEN(KaiDataset, hDataset, pDataset);

	pDataset->read_file(sDataFilePath);

	NO_SESSION_CLOSE();
}
