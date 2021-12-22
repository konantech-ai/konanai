/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "cuda_conn.cuh"
#include "cuda_note.h"
#include "cuda_kernels.h"
#include "cuda_math.h"

#include "../core/common.h"
#include "../core/log.h"
#include "../core/array.h"
#include "../core/engine.h"
#include "../core/value.h"

#include "../int_plugin/layer.cuh"
#include "../int_plugin/optimizer.cuh"

#ifdef KAI2021_WINDOWS
int block_size = 512; // 256;
#else
int block_size = 1024; // GTE-960 장착된 desktop에서 실행하니 'too many resources requested for launch' 오류 발생
#endif

bool CudaConn::ms_using_cuda = false;

curandGenerator_t CudaConn::ms_rand_gen;

int CudaConn::ms_nDevice = 0;

mutex CudaConn::ms_mu_cudacall;

extern HostMath* kmath = NULL;

bool CudaConn::IsCudaAvailable() {
    float* cuda_s;
    cudaMalloc(&cuda_s, 1024 * sizeof(float));
    cudaFree(cuda_s);
    cudaError_t cuda_ret = cudaGetLastError();
    return cuda_ret == cudaSuccess;
}

void CudaConn::OpenCuda(int nDevice) {
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
}

void CudaConn::CloseCuda() {
    ms_using_cuda = false;
    kmath = &hmath;
}

bool CudaConn::SetDevice(int nDevice) {
    if (nDevice < 0) nDevice = ms_nDevice;
    cudaError_t cuda_ret = cudaSetDevice(nDevice);
    if (cuda_ret != cudaSuccess) return false;
    ms_nDevice = nDevice;
    return true;
}

int CudaConn::GetDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

int64 CudaConn::getUsingMemSize() {
    if (!ms_using_cuda) return 0;
    return cudanote.getUsingMemSize();
}

int64 CudaConn::getAvailMemSize() {
    if (!ms_using_cuda) return 0;
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (int64)free;
}

CudaConn::CudaConn(string name, Layer* pLayer) {
    m_name = name;
    m_pLayer = pLayer;
}

CudaConn::~CudaConn() {
    //static int nth = 0;
    for (vector<void*>::iterator it = m_host_mem_blocks.begin(); it != m_host_mem_blocks.end(); it++) {
        void* host_p = *it;
        free(host_p);
    }

    if (LockOpt()) {
        cudanote.free_all_local(this);
        cuda_check("destructor()");
        Unlock();
    }
}

bool CudaConn::LockOpt() {
    if (!ms_using_cuda) return false;

    ms_mu_cudacall.lock();
    return true;
}

void CudaConn::LockObl() {
    if (!ms_using_cuda) throw KaiException(KERR_ASSERT);
    ms_mu_cudacall.lock();
}

void CudaConn::Unlock() {
    ms_mu_cudacall.unlock();
}

void CudaConn::cuda_check(string pos) {
    cudaError_t cuda_ret = cudaGetLastError();

    if (cuda_ret != cudaSuccess) {
        cudanote.dump_usage();
        logger.Print("cudaError: %s in %s(%s)::%s", cudaGetErrorString(cuda_ret), m_name.c_str(), m_pLayer->desc().c_str(), pos.c_str());
        throw KaiException(KERR_ASSERT);
    }
}

string CudaConn::desc() {
    if (this == NULL) return "NULL";
    else return m_name;
}

void CudaConn::random_uniform(float* cuda_p, int64 size) {
    // curand 랜덤 발생기는 홀수 크기의 발생을 요구받으면 105(CURAND_STATUS_LENGTH_NOT_MULTIPLE) 에러를 발생시킴
    size = (size + 3) / 4 * 4;
    LockObl();
    curand_call(curandGenerateUniform(ms_rand_gen, cuda_p, size));
    Unlock();
}

void CudaConn::random_bernoulli(float* cuda_p, float prob_threshod, int64 size) {
    // curand 랜덤 발생기는 홀수 크기의 발생을 요구받으면 105(CURAND_STATUS_LENGTH_NOT_MULTIPLE) 에러를 발생시킴
    size = (size + 3) / 4 * 4;
    LockObl();
    curand_call(curandGenerateUniform(ms_rand_gen, cuda_p, size));
    cu_call(ker_binomial, size, (size, cuda_p, prob_threshod));
    Unlock();
}

void CudaConn::random_normal(float* cuda_p, int64 size, float mean, float std) {
    // curand 랜덤 발생기는 홀수 크기의 발생을 요구받으면 105(CURAND_STATUS_LENGTH_NOT_MULTIPLE) 에러를 발생시킴
    size = (size + 3) / 4 * 4;
    LockObl();
    curand_call(curandGenerateNormal(ms_rand_gen, cuda_p, size, mean, std));
    Unlock();
}

float CudaConn::get_nth_element(float* cuda_p, int nth) {
    float element;
    LockObl();
    cudaMemcpy(&element, cuda_p + nth, sizeof(float), cudaMemcpyDeviceToHost);
    Unlock();
    return element;
}

void CudaConn::get_nth_row(float* host_dst, float* cuda_src, int64 nth_row, int64 ncol_size) {
    LockObl();
    cudaMemcpy(host_dst, cuda_src + nth_row * ncol_size, ncol_size * sizeof(float), cudaMemcpyDeviceToHost);
    Unlock();
}

int64 CudaConn::get_nth_element(int64* cuda_p, int64 nth) {
    int64 element;
    LockObl();
    cudaMemcpy(&element, cuda_p + nth, sizeof(int64), cudaMemcpyDeviceToHost);
    Unlock();
    return element;
}

void CudaConn::Copy_host_to_cuda(void* p_cuda, void* p_host, int64 data_size) {
    LockObl();
    cudaMemcpy(p_cuda, p_host, data_size, cudaMemcpyHostToDevice);
    Unlock();
}

void CudaConn::Free(void* cuda_p, void* pCore) {
    if (cuda_p == NULL) return;

    LockObl();
    cudanote.free_global(cuda_p, pCore);
    Unlock();
}

bool CudaConn::HasCudaMem(Array<float> arr) {
    LockObl();
    if (arr.is_cuda()) {
        cudanote.existing_global_check(arr.data_ptr());
        Unlock();
        return true;
    }
    else {
        cudanote.not_existing_global_check(arr.data_ptr());
        Unlock();
        return false;
    }
}

float* CudaConn::GetCudaFloatMem(Array<float> arr, string desc) {
    return GetCudaMem(arr, desc);
}

float* CudaConn::GetCudaMem(Array<float> arr, string desc) {
    LockObl();
    assert(arr.total_size() > 0);  // shape
    if (arr.m_core->m_mdata->m_isCuda) {
        cudanote.existing_global_check(arr.data_ptr());
    }
    else {
        cudanote.not_existing_global_check(arr.data_ptr());
        cudanote.allocate_global(arr, desc);
    }
    Unlock();

    return arr.m_core->m_mdata->m_data;
}

int* CudaConn::GetCudaMem(Array<int> arr, string desc) {
    LockObl();
    assert(arr.total_size() > 0);  // shape
    if (arr.m_core->m_mdata->m_isCuda) {
        cudanote.existing_global_check(arr.data_ptr());
    }
    else {
        cudanote.not_existing_global_check(arr.data_ptr());
        cudanote.allocate_global(arr, desc);
    }
    Unlock();

    return arr.m_core->m_mdata->m_data;
}

int64* CudaConn::GetCudaMem(Array<int64> arr, string desc) {
    LockObl();
    assert(arr.total_size() > 0);  // shape
    if (arr.m_core->m_mdata->m_isCuda) {
        cudanote.existing_global_check(arr.data_ptr());
    }
    else {
        cudanote.not_existing_global_check(arr.data_ptr());
        cudanote.allocate_global(arr, desc);
    }
    Unlock();

    return arr.m_core->m_mdata->m_data;
}

float* CudaConn::GetHostMem(Array<float> arr, string desc) {
    LockObl();
    if (arr.m_core->m_mdata->m_isCuda) {
        cudanote.existing_global_check(arr.data_ptr());
        cudanote.restore_global(arr.data_ptr(), arr.m_core->m_mdata);
    }
    else {
        cudanote.not_existing_global_check(arr.m_core->m_mdata->m_data);
    }
    Unlock();

    return arr.m_core->m_mdata->m_data;
}

Array<float> CudaConn::ToHostArray(Array<float> arr, string desc) {
    if (LockOpt()) {
        if (arr.m_core->m_mdata->m_isCuda) {
            Array<float> harr(arr.shape());
            cudaMemcpy(harr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyDeviceToHost);
            arr = harr;
        }
        Unlock();
    }
    return arr;
}

Array<float> CudaConn::ToCudaArray(Array<float> arr, string desc) {
    if (LockOpt()) {
        if (!arr.m_core->m_mdata->m_isCuda) {
            Array<float> carr = cudanote.create_farray(NULL, arr.shape(), desc);
            cudaMemcpy(carr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyHostToDevice);
            arr = carr;
        }
        Unlock();
    }
    return arr;
}

Array<int> CudaConn::ToHostArray(Array<int> arr, string desc) {
    if (LockOpt()) {
        if (arr.m_core->m_mdata->m_isCuda) {
            Array<int> harr(arr.shape());
            cudaMemcpy(harr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyDeviceToHost);
            arr = harr;
        }
        Unlock();
    }
    return arr;
}

Array<int> CudaConn::ToCudaArray(Array<int> arr, string desc) {
    if (LockOpt()) {
        if (!arr.m_core->m_mdata->m_isCuda) {
            Array<int> carr = cudanote.create_narray(NULL, arr.shape(), desc);
            cudaMemcpy(carr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyHostToDevice);
            arr = carr;
        }
        Unlock();
    }
    return arr;
}

Array<int64> CudaConn::ToHostArray(Array<int64> arr, string desc) {
    if (LockOpt()) {
        if (arr.m_core->m_mdata->m_isCuda) {
            Array<int64> harr(arr.shape());
            cudaMemcpy(harr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyDeviceToHost);
            arr = harr;
        }
        Unlock();
    }
    return arr;
}

Array<int64> CudaConn::ToCudaArray(Array<int64> arr, string desc) {
    if (LockOpt()) {
        if (!arr.m_core->m_mdata->m_isCuda) {
            Array<int64> carr = cudanote.create_n64array(NULL, arr.shape(), desc);
            cudaMemcpy(carr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyHostToDevice);
            arr = carr;
        }
        Unlock();
    }
    return arr;
}

Array<bool> CudaConn::ToHostArray(Array<bool> arr, string desc) {
    if (LockOpt()) {
        if (arr.m_core->m_mdata->m_isCuda) {
            Array<bool> harr(arr.shape());
            cudaMemcpy(harr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyDeviceToHost);
            arr = harr;
        }
        Unlock();
    }
    return arr;
}

Array<bool> CudaConn::ToCudaArray(Array<bool> arr, string desc) {
    if (LockOpt()) {
        if (!arr.m_core->m_mdata->m_isCuda) {
            Array<bool> carr = cudanote.create_barray(NULL, arr.shape(), desc);
            cudaMemcpy(carr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyHostToDevice);
            arr = carr;
        }
        Unlock();
    }
    return arr;
}

Array<short> CudaConn::ToHostArray(Array<short> arr, string desc) {
    if (LockOpt()) {
        if (arr.m_core->m_mdata->m_isCuda) {
            Array<short> harr(arr.shape());
            cudaMemcpy(harr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyDeviceToHost);
            arr = harr;
        }
        Unlock();
    }
    return arr;
}

Array<short> CudaConn::ToCudaArray(Array<short> arr, string desc) {
    if (LockOpt()) {
        if (!arr.m_core->m_mdata->m_isCuda) {
            Array<short> carr = cudanote.create_sarray(NULL, arr.shape(), desc);
            cudaMemcpy(carr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyHostToDevice);
            arr = carr;
        }
        Unlock();
    }
    return arr;
}

Array<float> CudaConn::to_cuda_array(Array<float> arr, string desc) {
    if (LockOpt()) {
        if (!arr.m_core->m_mdata->m_isCuda) {
            Array<float> carr = cudanote.create_farray(this, arr.shape(), desc);
            cudaMemcpy(carr.data_ptr(), arr.data_ptr(), arr.m_core->m_mdata->m_size, cudaMemcpyHostToDevice);
            arr = carr;
        }
        Unlock();
    }
    return arr;
}

void CudaConn::DumpUsage() {
    LockObl();
    cudanote.dump_usage();
    Unlock();
}

void CudaConn::GarbageCheck() {
    if (LockOpt()) {
        cudanote.garbage_check();
        Unlock();
    }
}

float* CudaConn::Alloc_float_mem(Shape shape, string desc) {
    LockObl();
    float* result = cudanote.alloc_float_mem(NULL, shape, desc);
    Unlock();
    return result;
}

int* CudaConn::Alloc_int_mem(Shape shape, string desc) {
    LockObl();
    int* result = cudanote.alloc_int_mem(NULL, shape, desc);
    Unlock();
    return result;
}

float* CudaConn::alloc_float_mem(Shape shape, string desc) {
    LockObl();
    float* result = cudanote.alloc_float_mem(this, shape, desc);
    Unlock();
    return result;
}

int* CudaConn::alloc_int_mem(Shape shape, string desc) {
    LockObl();
    int* result = cudanote.alloc_int_mem(this, shape, desc);
    Unlock();
    return result;
}

int64* CudaConn::alloc_int64_mem(Shape shape, string desc) {
    LockObl();
    int64* result = cudanote.alloc_int64_mem(this, shape, desc);
    Unlock();
    return result;
}

int64* CudaConn::alloc_int64_mem(int64* p_host, int64 size, string desc) {
    LockObl();
    int64* result = cudanote.alloc_int64_mem(this, p_host, size, desc);
    Unlock();
    return result;
}

unsigned char* CudaConn::alloc_byte_mem(Shape shape, string desc) {
    LockObl();
    unsigned char* result = cudanote.alloc_byte_mem(this, shape, desc);
    Unlock();
    return result;
}

float* CudaConn::copy_to_buffer(Array<float> arr, string desc) {
    LockObl();
    float* cuda_p = cudanote.alloc_float_mem(this, arr.shape(), desc);
    if (arr.m_core->m_mdata->m_isCuda)
        cudaMemcpy(cuda_p, arr.data_ptr(), arr.total_size() * sizeof(float), cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy(cuda_p, arr.data_ptr(), arr.total_size() * sizeof(float), cudaMemcpyHostToDevice);
    Unlock();
    return cuda_p;
}

int* CudaConn::copy_to_buffer(Array<int> arr, string desc) {
    LockObl();
    int* cuda_p = cudanote.alloc_int_mem(this, arr.shape(), desc);
    if (arr.m_core->m_mdata->m_isCuda)
        cudaMemcpy(cuda_p, arr.data_ptr(), arr.total_size() * sizeof(int), cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy(cuda_p, arr.data_ptr(), arr.total_size() * sizeof(int), cudaMemcpyHostToDevice);
    Unlock();
    return cuda_p;
}

int64* CudaConn::copy_to_buffer(Array<int64> arr, string desc) {
    LockObl();
    int64* cuda_p = cudanote.alloc_int64_mem(this, arr.shape(), desc);
    if (arr.m_core->m_mdata->m_isCuda)
        cudaMemcpy(cuda_p, arr.data_ptr(), arr.total_size() * sizeof(int64), cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy(cuda_p, arr.data_ptr(), arr.total_size() * sizeof(int64), cudaMemcpyHostToDevice);
    Unlock();
    return cuda_p;
}

float* CudaConn::copy(Array<float> arr, string desc) {
    LockObl();
    float* result = cudanote.copy(this, arr, desc);
    Unlock();
    return result;
}

float* CudaConn::Copy(Array<float> arr, string desc) {
    LockObl();
    float* result = cudanote.copy(NULL, arr, desc);
    Unlock();
    return result;
}

float* CudaConn::attach(Array<float> arr, string desc) {
    LockObl();
    float* result = cudanote.attach(this, arr, desc);
    Unlock();
    return result;
}

int* CudaConn::attach_int(Array<int> arr, string desc) {
    LockObl();
    int* result = cudanote.attach_int(this, arr, desc);
    Unlock();
    return result;
}

int64* CudaConn::attach_int64(Array<int64> arr, string desc) {
    LockObl();
    int64* result = cudanote.attach_int64(this, arr, desc);
    Unlock();
    return result;
}

short* CudaConn::attach_short(Array<short> arr, string desc) {
    LockObl();
    short* result = cudanote.attach_short(this, arr, desc);
    Unlock();
    return result;
}

Array<float> CudaConn::detach(float* cuda_p, string desc) {
    LockObl();
    Array<float> result = cudanote.detach(this, cuda_p, desc);
    Unlock();
    return result;
}

Array<int> CudaConn::detach(int* cuda_p, string desc) {
    LockObl();
    Array<int> result = cudanote.detach(this, cuda_p, desc);
    Unlock();
    return result;
}

Array<int64> CudaConn::detach(int64* cuda_p, string desc) {
    LockObl();
    Array<int64> result = cudanote.detach(this, cuda_p, desc);
    Unlock();
    return result;
}

Array<float> CudaConn::copy(float* cuda_p, string desc) {
    LockObl();
    Array<float> result = cudanote.copy(this, cuda_p, desc);
    Unlock();
    return result;
}

template <class T> Array<T> CudaConn::Copy(T* cuda_p, string desc) {
    LockObl();
    Array<T> result = cudanote.copy(NULL, cuda_p, desc);
    Unlock();
    return result;
}

/*
template <class T> T* CudaConn::CopyAsPtr2Ptr(T* cuda_p, string desc) {
    LockObl();
    T* result = cudanote.copy_as_p2p(NULL, cuda_p, desc);
    Unlock();
    return result;
}
*/

Array<float> CudaConn::create_farray(Shape shape, string desc) {
    LockObl();
    Array<float> result = cudanote.create_farray(this, shape, desc);
    Unlock();
    return result;
}

Array<float> CudaConn::CreateFloatArray(Shape shape, string desc) {
    LockObl();
    Array<float> result = cudanote.create_farray(NULL, shape, desc);
    Unlock();
    return result;
}

Array<int> CudaConn::create_narray(Shape shape, string desc) {
    LockObl();
    Array<int> result = cudanote.create_narray(this, shape, desc);
    Unlock();
    return result;
}

Array<int64> CudaConn::create_n64array(Shape shape, string desc) {
    LockObl();
    Array<int64> result = cudanote.create_n64array(this, shape, desc);
    Unlock();
    return result;
}

/*
void CudaConn::optimize_weight(Dict& param, Engine& engine, float* cuda_gpm) {
    Optimizer* optimizer = engine.get_optimizer();
    if (optimizer != NULL) {
        optimizer->update_weight_cuda(param, cuda_gpm);
    }
    else {
        optimize_old(param, "w", engine, cuda_gpm);
    }
}

void CudaConn::optimize_bias(Dict& param, Engine& engine, float* cuda_gpm) {
    Optimizer* optimizer = engine.get_optimizer();
    if (optimizer != NULL) {
        optimizer->update_bias_cuda(param, cuda_gpm);
    }
    else {
        optimize_old(param, "b", engine, cuda_gpm);
    }
}

void CudaConn::optimize(Dict& param, string key, Engine& engine, float* cuda_gpm) {
    throw KaiException(KERR_ASSERT);
}

void CudaConn::optimize_old(Dict& param, string key, Engine& engine, float* cuda_gpm) {
    Array<float> pm = param[key];
    Shape pshape = pm.shape();
    int64 psize = pshape.total_size();

    float* cuda_pm = pm.data_ptr();

    float learning_rate = engine.learning_rate;
    float l2_decay = (key != "b") ? engine.l2_decay : 0;
    float l1_decay = (key != "b") ? engine.l1_decay : 0;

    if (!engine.use_adam) {
        cu_call_no_lock(ker_update_param_sgd, psize, (psize, cuda_pm, cuda_gpm, learning_rate, l2_decay, l1_decay));
    }
    else {
        float ro1 = 0.900f, ro2 = 0.999f, epsilon = 1.0e-8f;

        Array<float> s = param["s"+key], t = param["t" + key];

        float* cuda_s = s.data_ptr();
        float* cuda_t = t.data_ptr();

        int nstep = param["n" + key];

        param["n" + key] = ++nstep;  // param must be passed by call-by-reference (not call-by-value)

        cu_call_no_lock(ker_update_param_adam, psize, (psize, cuda_pm, cuda_s, cuda_t, cuda_gpm, ro1, ro2, nstep, epsilon, learning_rate, l2_decay, l1_decay));
    }
    Unlock();
}

void CudaConn::optimize_select(Dict& param, string key, Engine& engine, int64 voc_cnt, int64 word_cnt, int* cuda_wid, float* cuda_gpm) {
    LockObl();
    Array<float> pm = param[key];

    int64 vec_size = pm.axis_size(-1);
    int64 psize = voc_cnt * vec_size;

    //logger.Print("psize = %lld, voc_cnt = %lld, word_cnt = %lld, vec_size = %lld", psize, voc_cnt, word_cnt, vec_size);

    float* cuda_pm = pm.data_ptr();

    float learning_rate = engine.learning_rate;
    float l2_decay = (key != "b") ? engine.l2_decay : 0;
    float l1_decay = (key != "b") ? engine.l1_decay : 0;

    if (!engine.use_adam) {
        throw KaiException(KERR_ASSERT); // optimize_select()처럼 누적식 처리로 수정
        cu_call_no_lock(ker_update_param_sgd_select, psize, (psize, cuda_pm, cuda_wid, cuda_gpm, word_cnt, vec_size, learning_rate, l2_decay, l1_decay));
    }
    else {
        float ro1 = 0.900f, ro2 = 0.999f, epsilon = 1.0e-8f;

        Array<float> s = param["s"+key], t = param["t" + key], n = param["n" + key];

        float* cuda_s = s.data_ptr();
        float* cuda_t = t.data_ptr();
        float* cuda_n = n.data_ptr();

        cu_call_no_lock(ker_update_param_adam_select, psize, (psize, cuda_pm, cuda_s, cuda_t, cuda_n, cuda_wid, cuda_gpm, word_cnt, vec_size, ro1, ro2, epsilon, learning_rate, l2_decay, l1_decay));
    }
    Unlock();
}

void CudaConn::optimize_select_multi_dic(Dict& param, string key, Engine& engine, int64 word_cnt, int* cuda_wid, float* cuda_gpm, int64 dic_count, int64* voc_counts) {
    //throw KaiException(KERR_ASSERT); // optimize_select()처럼 누적식 처리로 수정
    LockObl();
    Array<float> pm = param[key];

    int64 vec_size = pm.axis_size(-1);
    int64 psize = word_cnt * dic_count * vec_size;

    float* cuda_pm = pm.data_ptr();

    float learning_rate = engine.learning_rate;
    float l2_decay = (key != "b") ? engine.l2_decay : 0;
    float l1_decay = (key != "b") ? engine.l1_decay : 0;

    Shape wshape = cudanote.get_shape(cuda_wid);
    Shape dshape = wshape.append(vec_size);

    float* cuda_delta = cudanote.alloc_float_mem(this, dshape, "delta_sum");
    float* cuda_count = cudanote.alloc_float_mem(this, wshape, "dup_count"); 

    cu_call_no_lock(ker_update_param_dup_count, psize, (psize, cuda_delta, cuda_count, cuda_wid, cuda_gpm, dic_count, vec_size));

    //DumpArr(cuda_delta, Shape(), "delta");
    //DumpArr(cuda_count, Shape(), "count");

    if (!engine.use_adam) {
        throw KaiException(KERR_ASSERT);
        cu_call_no_lock(ker_update_param_sgd_select_multi_dic, psize, (psize, cuda_pm, cuda_wid, cuda_gpm, dic_count, voc_counts, vec_size, learning_rate, l2_decay, l1_decay));
    }
    else {
        float ro1 = 0.900f, ro2 = 0.999f, epsilon = 1.0e-8f;

        Array<float> s = param["s" + key], t = param["t" + key], n = param["n" + key];

        float* cuda_s = s.data_ptr();
        float* cuda_t = t.data_ptr();
        float* cuda_n = n.data_ptr();

        cu_call_no_lock(ker_update_param_adam_select_multi_dic, psize, (psize, cuda_pm, cuda_s, cuda_t, cuda_n, cuda_wid, cuda_delta, cuda_count, dic_count, voc_counts, vec_size, ro1, ro2, epsilon, learning_rate, l2_decay, l1_decay));
    }
    Unlock();
}
*/

float* CudaConn::get_host_data(Array<float> arr) {
    if (!arr.m_core->m_mdata->m_isCuda) {
        return arr.m_core->m_mdata->m_data;
    }

    LockObl();

    void* host_p = malloc(arr.m_core->m_mdata->m_size);
    void* cuda_p = arr.m_core->m_mdata->m_data;

    cudanote.existing_global_check(cuda_p);

    int64 size = arr.m_core->m_mdata->m_size;

    //logger.Print("get_host_data: cuda_p = %lld, size = %lld", (int64) cuda_p, size);
    //cudanote.dump("get_host_data");

    cudaMemcpy(host_p, cuda_p, size, cudaMemcpyDeviceToHost);

    m_host_mem_blocks.push_back(host_p);
    Unlock();

    return (float*) host_p;
}

Array<float> CudaConn::to_host_array(float* cuda_p) {
    LockObl();
    Shape shape = cudanote.get_shape(cuda_p);

    Array<float> arr(shape);

    cudaMemcpy(arr.data_ptr(), cuda_p, shape.total_size() * sizeof(float), cudaMemcpyDeviceToHost);

    Unlock();
    return arr;
}

void CudaConn::dump_note(string desc) {
    LockObl();
    cudanote.dump(desc);
    Unlock();
}

string CudaConn::ms_to_string(void* host_p, arr_type atype, Shape shape, Idx& idx, int64 depth, bool full) {
    string buffer;
    string infix = ",";
    string delimeter = "[";

    int64 axis_size = shape[depth];

    if (depth < shape.size() - 1) {
        infix = ",\n     ";
        for (int n = 0; n < depth; n++) infix += " ";
    }

    for (int64 n = 0; n < axis_size; n++) {
        idx[depth] = n;
        if (!full && axis_size >= 10 && n >= 3 && n <= axis_size - 3) {
            if (n == 3) buffer += delimeter + "...";
            continue;
        }
        if (depth == shape.size() - 1) {
            int64 nth = idx[0];
            for (int k = 1; k < shape.size(); k++) {
                nth = nth * shape[k] + idx[k];
            }

            char fbuf[2048];

            //if (atype == arr_type::arr_float) sprintf(fbuf, "%16.9e", ((float*)host_p)[nth]);
            if (atype == arr_type::arr_float) sprintf(fbuf, "%16.9e", ((float*)host_p)[nth]);
            else if (atype == arr_type::arr_int) sprintf(fbuf, "%d", ((int*)host_p)[nth]);
            else if (atype == arr_type::arr_int64) sprintf(fbuf, "%lld", ((int64*)host_p)[nth]);
            else if (atype == arr_type::arr_uchar) sprintf(fbuf, "%02x ", ((unsigned char*)host_p)[nth]);
            else throw KaiException(KERR_ASSERT);

            buffer += delimeter + (string)fbuf; // +"(" + idx.desc() + ")";
        }
        else {
            buffer += delimeter + ms_to_string(host_p, atype, shape, idx, depth + 1, full);
        }
        delimeter = infix;
    }

    return buffer + "]";
}

string CudaConn::ms_rows_to_string(void* host_p, arr_type atype, Shape shape, int64 nfrom, int64 nto, int64 col) {
    if (col <= 0) col = shape.total_size() / shape[0];

    assert(shape.total_size() % col == 0);

    int64 nrows = shape.total_size() / col;

    if (nto < 0) nto = nrows;

    string buffer;

    for (int64 n = nfrom; n < nto; n++) {
        buffer += (n == nfrom) ? "[" : "]\n     ";

        for (int64 m = 0; m < col; m++) {
            buffer += (m == 0) ? "[" : ",";

            char fbuf[2048];

            int64 nth = n * col + m;

            if (atype == arr_type::arr_float) sprintf(fbuf, "%16.9e", ((float*)host_p)[nth]);
            else if (atype == arr_type::arr_int) sprintf(fbuf, "%d", ((int*)host_p)[nth]);
            else if (atype == arr_type::arr_int64) sprintf(fbuf, "%lld", ((int64*)host_p)[nth]);
            else if (atype == arr_type::arr_uchar) sprintf(fbuf, "%02x ", ((unsigned char*)host_p)[nth]);
            else throw KaiException(KERR_ASSERT);

            buffer += fbuf;
        }
    }

    return buffer + "]]";
}

string CudaConn::ms_selected_rows_to_string(void* host_p, arr_type atype, Shape shape, int64* pidxs, int64 max_cnt, int64 col) {
    if (col <= 0) col = shape.total_size() / shape[0];

    assert(shape.total_size() % col == 0);

    string buffer;

    for (int n = 0; n < max_cnt; n++) {
        buffer += ((n == 0) ? "[" : "]\n     ");
        buffer += to_string(pidxs[n]) + " ";

        for (int64 m = 0; m < col; m++) {
            buffer += (m == 0) ? "[" : ",";

            char fbuf[2048];

            int64 nth = pidxs[n] * col + m;
            //logger.Print("BP1: n = %d, pidxs[n] = %lld, m = %lld, col = %lld, nth = %lld", n, pidxs[n], m, col, nth);

            if (atype == arr_type::arr_float) sprintf(fbuf, "%16.9e", ((float*)host_p)[nth]);
            else if (atype == arr_type::arr_int) sprintf(fbuf, "%d", ((int*)host_p)[nth]);
            else if (atype == arr_type::arr_int64) sprintf(fbuf, "%lld", ((int64*)host_p)[nth]);
            else if (atype == arr_type::arr_uchar) sprintf(fbuf, "%02x ", ((unsigned char*)host_p)[nth]);
            else throw KaiException(KERR_ASSERT);

            buffer += fbuf;
        }
    }

    return buffer + "]]";
}

void CudaConn::DumpShape(void* cuda_p, string desc) {
    LockObl();
    Shape shape = cudanote.get_shape(cuda_p);
    Unlock();

    logger.Print("[%s] %s cuda", desc.c_str(), shape.desc().c_str());
}

int64 CudaConn::DumpSparse(void* cuda_p, string desc, int64* pidxs, int64 max_cnt, int64 max_get_cnt, int64 ncols) {
    LockObl();
    Shape shape = cudanote.get_shape(cuda_p);
    int64 reg_size = cudanote.get_msize(cuda_p);
    arr_type atype = cudanote.get_type(cuda_p);
    Unlock();

    if (max_cnt > 0) {
        logger.Print("[%s] %s cuda", desc.c_str(), shape.desc().c_str());
    }

    int64 size = reg_size;

    int tsize = 0;
    if (atype == arr_type::arr_float) tsize = sizeof(float);
    else if (atype == arr_type::arr_int) tsize = sizeof(int);
    else if (atype == arr_type::arr_int64) tsize = sizeof(int64);
    else if (atype == arr_type::arr_uchar) tsize = sizeof(unsigned char);
    else if (atype == arr_type::arr_bool) tsize = sizeof(bool);
    else throw KaiException(KERR_ASSERT);

    char* host_p = (char*)malloc(size * tsize);
    cudaMemcpy(host_p, cuda_p, size * tsize, cudaMemcpyDeviceToHost);

    if (ncols < 0) ncols = shape[-1];
    assert(shape.total_size() % ncols == 0);
    int64 nrows = shape.total_size() / ncols;

    char* zeros = (char*)malloc(ncols * tsize);
    assert(zeros != NULL);
    memset(zeros, 0, ncols * tsize);

    int64 count = 0;

    for (int64 n = 0; n < nrows; n++) {
        void* dp = host_p + (n * tsize * ncols);
        if (memcmp(dp, zeros, tsize * ncols) != 0) {
            if (count < max_cnt) {
                logger.PrintWait("    Line %lld(%d-th): ", n, count);
                for (int64 m = 0; m < ncols; m++) {
                    logger.PrintWait("%c", m ? ',' : '[');
                    if (atype == arr_type::arr_float) logger.PrintWait("%16.9e", ((float*)dp)[m]);
                    else if (atype == arr_type::arr_int) logger.PrintWait("%d", ((int*)dp)[m]);
                    else if (atype == arr_type::arr_int64) logger.PrintWait("%lld", ((int64*)dp)[m]);
                    else if (atype == arr_type::arr_uchar) logger.PrintWait("%02x ", ((unsigned char*)dp)[m]);
                    else throw KaiException(KERR_ASSERT);
                }
                logger.Print("]");
            }
            if (pidxs && count < max_get_cnt) {
                //logger.Print("setting pidxs[%lld] to %lld", count, n);
                pidxs[count] = n;
            }
            count++;
        }
    }

    free(host_p);
    free(zeros);

    return count;
}

int64 CudaConn::DumpZeros(void* cuda_p, string desc, int64* pidxs, int64 max_cnt, int64 max_get_cnt, int64 ncols) {
    LockObl();
    Shape shape = cudanote.get_shape(cuda_p);
    int64 reg_size = cudanote.get_msize(cuda_p);
    arr_type atype = cudanote.get_type(cuda_p);
    Unlock();

    logger.Print("[%s] %s cuda", desc.c_str(), shape.desc().c_str());

    int64 size = reg_size;

    int tsize = 0;
    if (atype == arr_type::arr_float) tsize = sizeof(float);
    else if (atype == arr_type::arr_int) tsize = sizeof(int);
    else if (atype == arr_type::arr_int64) tsize = sizeof(int64);
    else if (atype == arr_type::arr_uchar) tsize = sizeof(unsigned char);
    else if (atype == arr_type::arr_bool) tsize = sizeof(bool);
    else throw KaiException(KERR_ASSERT);

    char* host_p = (char*)malloc(size * tsize);
    cudaMemcpy(host_p, cuda_p, size * tsize, cudaMemcpyDeviceToHost);

    if (ncols < 0) ncols = shape[-1];
    assert(shape.total_size() % ncols == 0);
    int64 nrows = shape.total_size() / ncols;

    char* zeros = (char*)malloc(ncols * tsize);
    assert(zeros != NULL);
    memset(zeros, 0, ncols * tsize);

    int64 count = 0;

    for (int64 n = 0; n < nrows; n++) {
        void* dp = host_p + (n * tsize * ncols);
        if (memcmp(dp, zeros, tsize * ncols) == 0) {
            if (count < max_cnt) {
                logger.Print("    Line %lld(%d-th): zero", n, count);
            }
            if (pidxs && count < max_get_cnt) pidxs[count] = n;
            count++;
        }
    }

    logger.Print("");

    free(host_p);
    free(zeros);

    return count;
}

void CudaConn::DumpArr(void* cuda_p, string desc, Shape shape, bool full) {
    LockObl();
    int64 reg_size = cudanote.get_msize(cuda_p);
    if (shape.size() == 0) shape = cudanote.get_shape(cuda_p);
    arr_type atype = cudanote.get_type(cuda_p);
    Unlock();

    int64 size = reg_size;

    int tsize = 0;
    if (atype == arr_type::arr_float) tsize = sizeof(float);
    else if (atype == arr_type::arr_int) tsize = sizeof(int);
    else if (atype == arr_type::arr_int64) tsize = sizeof(int64);
    else if (atype == arr_type::arr_uchar) tsize = sizeof(unsigned char);
    else if (atype == arr_type::arr_bool) tsize = sizeof(bool);
    else throw KaiException(KERR_ASSERT);

    void* host_p = malloc(size * tsize);

    cudaMemcpy(host_p, cuda_p, size * tsize, cudaMemcpyDeviceToHost);

    Idx idx;
    idx.set_size(shape.size());
    string contents = ms_to_string(host_p, atype, shape, idx, 0, full);

    logger.Print("[%s] %s cuda", desc.c_str(), shape.desc().c_str());
    logger.Print("    %s", contents.c_str());

    free(host_p);
}

Shape CudaConn::get_shape(void* cuda_p) {
    return cudanote.get_shape(cuda_p);
}

Shape CudaConn::GetShape(void* cuda_p) {
    return cudanote.get_shape(cuda_p);
}

void CudaConn::Print_rows(void* cuda_p, string desc, int64 nfrom, int64 nto, int64 col) {
    LockObl();
    int64 reg_size = cudanote.get_msize(cuda_p);
    Shape shape = cudanote.get_shape(cuda_p);
    arr_type atype = cudanote.get_type(cuda_p);
    Unlock();

    int64 size = reg_size;

    int tsize = 0;
    if (atype == arr_type::arr_float) tsize = sizeof(float);
    else if (atype == arr_type::arr_int) tsize = sizeof(int);
    else if (atype == arr_type::arr_int64) tsize = sizeof(int64);
    else if (atype == arr_type::arr_uchar) tsize = sizeof(unsigned char);
    else if (atype == arr_type::arr_bool) tsize = sizeof(bool);
    else throw KaiException(KERR_ASSERT);

    void* host_p = malloc(size * tsize);

    cudaMemcpy(host_p, cuda_p, size * tsize, cudaMemcpyDeviceToHost);

    Idx idx;
    idx.set_size(shape.size());
    string contents = ms_rows_to_string(host_p, atype, shape, nfrom, nto, col);

    logger.Print("[%s] %s cuda", desc.c_str(), shape.desc().c_str());
    logger.Print("    %s", contents.c_str());

    free(host_p);
}

void CudaConn::Print_selected_rows(void* cuda_p, string desc, int64* pidxs, int64 max_cnt, int64 col) {
    LockObl();
    int64 reg_size = cudanote.get_msize(cuda_p);
    Shape shape = cudanote.get_shape(cuda_p);
    arr_type atype = cudanote.get_type(cuda_p);
    Unlock();

    int64 size = reg_size;

    int tsize = 0;
    if (atype == arr_type::arr_float) tsize = sizeof(float);
    else if (atype == arr_type::arr_int) tsize = sizeof(int);
    else if (atype == arr_type::arr_int64) tsize = sizeof(int64);
    else if (atype == arr_type::arr_uchar) tsize = sizeof(unsigned char);
    else if (atype == arr_type::arr_bool) tsize = sizeof(bool);
    else throw KaiException(KERR_ASSERT);

    void* host_p = malloc(size * tsize);

    cudaMemcpy(host_p, cuda_p, size * tsize, cudaMemcpyDeviceToHost);

    Idx idx;
    idx.set_size(shape.size());
    string contents = ms_selected_rows_to_string(host_p, atype, shape, pidxs, max_cnt, col);

    logger.Print("[%s selected] %s cuda", desc.c_str(), shape.desc().c_str());
    logger.Print("    %s", contents.c_str());

    free(host_p);
}

template Array<float> CudaConn::Copy(float* cuda_p, string desc);
template Array<int> CudaConn::Copy(int* cuda_p, string desc);
template Array<int64> CudaConn::Copy(int64* cuda_p, string desc);
template Array<bool> CudaConn::Copy(bool* cuda_p, string desc);
template Array<unsigned char> CudaConn::Copy(unsigned char* cuda_p, string desc);
template Array<short> CudaConn::Copy(short* cuda_p, string desc);

/*
template float* CudaConn::CopyAsPtr2Ptr(float* cuda_p, string desc);
template int* CudaConn::CopyAsPtr2Ptr(int* cuda_p, string desc);
template int64* CudaConn::CopyAsPtr2Ptr(int64* cuda_p, string desc);
template bool* CudaConn::CopyAsPtr2Ptr(bool* cuda_p, string desc);
template unsigned char* CudaConn::CopyAsPtr2Ptr(unsigned char* cuda_p, string desc);
template short* CudaConn::CopyAsPtr2Ptr(short* cuda_p, string desc);
*/