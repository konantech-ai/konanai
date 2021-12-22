/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include <cuda_runtime.h>

#include "../session/kcommon.h"
#include "../include/kai_api.h"

//template <class T> class KaiArrayData;

class KaiDeviceManager;
class KaiModelInstance;

class CudaManager {
public:
	CudaManager();
	virtual ~CudaManager();

	void OpenCuda();

	KaiDeviceManager* alloc_device(KaiModelInstance* pModelContext);

	void lock(int nDevice);
	void unlock(int nDevice);

	//template <class T> void freeMemory(T* cuda_p, KaiArrayData<T>* pArrData);
protected:
	int m_nDeviceCount;
	kmutex_wrap* m_mu_cudacall;
};

extern CudaManager cuda_man;