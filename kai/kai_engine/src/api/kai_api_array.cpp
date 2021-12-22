/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_api_common.h"

#include "../math/karray.h"
#include "../utils/kutil.h"

KAI_API KRetCode KAI_Array_get_int_data(KHSession hSession, KHArray hArray, KInt nStart, KInt nCount, KInt* pBuffer) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiArrayCore<KInt>, hArray, pArray);

	pArray->get_data(nStart, nCount, pBuffer);

	SESSION_CLOSE();
}

KAI_API KRetCode KAI_Array_get_float_data(KHSession hSession, KHArray hArray, KInt nStart, KInt nCount, KFloat* pBuffer) {
	SESSION_OPEN();
	HANDLE_OPEN(KaiArrayCore<KFloat>, hArray, pArray);

	pArray->get_data(nStart, nCount, pBuffer);

	SESSION_CLOSE();
}
