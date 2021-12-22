/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include <cuda_runtime.h>

#include "../session/kcommon.h"
#include "../include/kai_api.h"

class KaiDeviceManager {
public:
	KaiDeviceManager(int nDevice);
	virtual ~KaiDeviceManager();

	void lock();
	void unlock();

	void errorCheck();
	void* allocMemory(KInt nRequestSize, KInt* npAllocatedSize);

	KFloat* allocFloatMemory(KaiShape shape, KInt* npAllocatedSize);
	KInt* allocIntMemory(KaiShape shape, KInt* npAllocatedSize);

	void freeMemory(void * pData, KInt size);
	//void freeMemory(KFloat* pData);

	KInt getAllocSize() { return m_nTotalAllocSize; }
	
	KInt getMaxBatchSize(KInt nNeedSizePerbatch);

protected:
	int m_nDevice;

	KInt m_nTotalAllocSize;
	std::map<void*, KInt> m_allocMap;
};