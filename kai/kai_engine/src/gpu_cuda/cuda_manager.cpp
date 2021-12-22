/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "cuda_manager.h"
#include "device_manager.h"

CudaManager cuda_man;

static kmutex_wrap* ms_mu_cudacall;

CudaManager::CudaManager() {
    m_mu_cudacall = NULL;
}

CudaManager::~CudaManager() {
    delete[] m_mu_cudacall;
}

void CudaManager::OpenCuda() {
    cudaGetDeviceCount(&m_nDeviceCount);
    m_mu_cudacall = new kmutex_wrap[m_nDeviceCount];
}

void CudaManager::lock(int nDevice) {
    m_mu_cudacall[nDevice].lock();
}

void CudaManager::unlock(int nDevice) {
    m_mu_cudacall[nDevice].unlock();
}

KaiDeviceManager* CudaManager::alloc_device(KaiModelInstance* pModelContext) {
    int nDevice = -1;
    KInt max_free = 0;
    for (int n = 0; n < m_nDeviceCount; n++) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        if ((KInt)free > max_free) {
            max_free = (KInt)free;
            nDevice = n;
        }
    }

    if (nDevice < 0) throw KaiException(KERR_NO_CUDA_DEVICE_IS_AVAILABLE);

    cudaError_t cuda_ret = cudaSetDevice(nDevice);
    if (cuda_ret != cudaSuccess) throw KaiException(KERR_FAILURE_ON_SETTING_CUDA_DEVICE);

    return new KaiDeviceManager(nDevice);
}

#ifdef xxx
void CudaManager::OpenCuda() {
    // CudaConn::OpenCuda() 같은 일이 각 device마다 일어나도록 처리, 즉 CudaDevManager를 CudaManager 부하로 둔다.
    logger.Print("CudaManager::OpenCuda() is not implemented yet");
    /*
    if (nDevice >= 0) ms_nDevice = nDevice;
    float* cuda_s;
    cudaSetDevice(ms_nDevice);
    cudaMalloc(&cuda_s, 1024 * sizeof(float));
    cudaFree(cuda_s);
    cudaError_t cuda_ret = cudaGetLastError();

    if (cuda_ret == cudaSuccess) {
        ms_using_cuda = true;
        kmath = &cmath;

        curand_call(curandCreateGenerator(&ms_rand_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curand_call(curandSetPseudoRandomGeneratorSeed(ms_rand_gen, 1234ULL));
    }
    else {
        ms_using_cuda = false;
        kmath = &hmath;
    }
    */
}

template <class T> void CudaManager::freeMemory(T* cuda_p, KaiArrayData<T>* pArrData) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

template void CudaManager::freeMemory(KInt* cuda_p, KaiArrayData<KInt>* pArrData);
template void CudaManager::freeMemory(KFloat* cuda_p, KaiArrayData<KFloat>* pArrData);
#endif