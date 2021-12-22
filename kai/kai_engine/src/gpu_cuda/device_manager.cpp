/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "device_manager.h"
#include "cuda_manager.h"
#include "../math/kshape.h"

KaiDeviceManager::KaiDeviceManager(int nDevice) : m_allocMap() {
	m_nDevice = nDevice;
	m_nTotalAllocSize = 0;
}

KaiDeviceManager::~KaiDeviceManager() {
}

void KaiDeviceManager::lock() {
	cuda_man.lock(m_nDevice);
}

void KaiDeviceManager::unlock() {
	cuda_man.unlock(m_nDevice);
}

KInt KaiDeviceManager::getMaxBatchSize(KInt nNeedSizePerbatch) {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free / nNeedSizePerbatch;	// 실제로 fragmentation 때문에 할당이 불가능할 수도 있음
}

void KaiDeviceManager::errorCheck() {
	cudaError_t cuda_ret = cudaGetLastError();

	if (cuda_ret != cudaSuccess) {
		KString sCudaError = cudaGetErrorString(cuda_ret);
		logger.Print("[Temporal Cuda Error] %s", sCudaError.c_str());
		throw KaiException(KERR_CUDA_ERROR, sCudaError);
	}
}

void* KaiDeviceManager::allocMemory(KInt nReqSize, KInt* pnAllocSize) {
	void* cuda_p = NULL;
	KInt nAllocSize = (nReqSize + 15) / 16 * 16;

	cudaMalloc(&cuda_p, nAllocSize);

	if (nAllocSize == 11075584 || nAllocSize == 2768896) {
		int n = 0;
	}

	if (cuda_p == NULL) {
		cudaError_t cuda_ret = cudaGetLastError();

		if (cuda_ret != cudaSuccess) {
			KString sCudaError = cudaGetErrorString(cuda_ret);
			logger.Print("[Temporal Cuda Error] %s", sCudaError.c_str());
			throw KaiException(KERR_CUDA_ERROR, sCudaError);
		}

		throw KaiException(KERR_CUDA_MALLOC_FAILURE);
	}
	if (pnAllocSize) *pnAllocSize = nAllocSize;
	
	cudaMemset(cuda_p, 0, nReqSize);

	m_nTotalAllocSize += nAllocSize;
	m_allocMap[cuda_p] = nAllocSize;

	return cuda_p;
}

KFloat* KaiDeviceManager::allocFloatMemory(KaiShape shape, KInt* npAllocatedSize) {
	KInt nReqSize = shape.total_size() * sizeof(KFloat);
	return (KFloat*)allocMemory(nReqSize, npAllocatedSize);
}

KInt * KaiDeviceManager::allocIntMemory(KaiShape shape, KInt* npAllocatedSize) {
	KInt nReqSize = shape.total_size() * sizeof(KInt);
	return (KInt*)allocMemory(nReqSize, npAllocatedSize);
}

void KaiDeviceManager::freeMemory(void* pData, KInt size) {
	auto it = m_allocMap.find(pData);
	if (it == m_allocMap.end()) throw KaiException(KERR_FREE_REQUEST_FOR_UNALLOCATED_DEV_MEM);
	if (it->second != size) throw KaiException(KERR_FREE_REQUEST_SIZE_MISMATCH);
	m_allocMap.erase(it);

	m_nTotalAllocSize -= size;

	cudaFree(pData);
}

/*
void KaiDeviceManager::freeMemory(KFloat* pData) {
    throw KaiException(KERR_UNIMPEMENTED_YET);
}
*/