/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kmath.h"
#include "khostmath.h"
#include "kcudamath.h"
#include "../gpu_cuda/cuda_manager.h"

class KaiDeviceManager;

KaiMath* KaiMath::GetHostMath() {
	return &hostmath;
}

KaiMath* KaiMath::Allocate(KaiModelInstance* pModelContext) {
	KaiDeviceManager* pDevManager = cuda_man.alloc_device(pModelContext);
	// 추후 device 검사 통해 free memory 가장 많은 device 선정해 해당 devide에서 연산 수행하는 cudamath 우선 배정하는 것으로 수정
	return new KaiCudaMath(pDevManager);
}
