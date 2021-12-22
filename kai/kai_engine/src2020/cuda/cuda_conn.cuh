/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/common.h"
#include "../core/shape.h"

class Engine;
class Layer;

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

extern int block_size;

#define cu_call(funcname, size, args) CudaConn::LockObl(); funcname << <(unsigned int)((size + block_size-1) / block_size), block_size>> > args; CudaConn::Unlock()
#define cu_call_no_lock(funcname, size, args) funcname << <(unsigned int)((size + block_size-1) / block_size), block_size>> > args
#define curand_call(x) do { curandStatus_t res=(x); if(res != CURAND_STATUS_SUCCESS) { logger.Print("Error at %s:%d, err:%d",__FILE__,__LINE__, res); throw KaiException(KERR_ASSERT);}} while(0)

template <class T> class ArrayDataCore;

class HostMath;

extern HostMath* kmath;

class CudaConn {
public:
	CudaConn(string name, Layer* pLayer);
	virtual ~CudaConn();

	static bool LockOpt(); // optional lock: 쿠다 사용중이면 lock 걸고 true 반환, 아니면 false 반환
	static void LockObl(); // obligation lock: 쿠다 사용중이면 lock 걸고, 아니면 throw KaiException(KERR_ASSERT) 처리
	static void Unlock();

	static void OpenCuda(int nDevice=-1);
	static void CloseCuda();
	static bool IsCudaAvailable();
	static bool UsingCuda() { return ms_using_cuda; }

	static int GetDeviceCount();
	static int GetCurrDevice() { return ms_using_cuda ? ms_nDevice : -1; }
	static int GetBlockSize() { return ::block_size; }

	static int64 getUsingMemSize();
	static int64 getAvailMemSize();

	static bool SetDevice(int nDevice = -1);
	static void SetBlockSize(int size) { ::block_size = size; }

	void cuda_check(string pos);

	static void random_uniform(float* cuda_p, int64 size);
	static void random_bernoulli(float* cuda_p, float prob_threshod, int64 size);
	static void random_normal(float* cuda_p, int64 size, float mean, float std);

	//template <class T> static void Free(T* cuda_p, ArrayDataCore<T>* core);
	static void Free(void* cuda_p, void* pCore);

	static bool HasCudaMem(Array<float> arr);

	static float* GetCudaFloatMem(Array<float> arr, string desc = "");

	static float* GetCudaMem(Array<float> arr, string desc = "");
	static int* GetCudaMem(Array<int> arr, string desc = "");
	static int64* GetCudaMem(Array<int64> arr, string desc = "");
	//static float* GetCudaMem(Shape shape, string desc="");
	//static float* GetCudaMem(Array<float> arr, string desc="");

	static float* GetHostMem(Array<float> arr, string desc="");

	static Array<float> ToHostArray(Array<float> arr, string desc=""); // 쿠다 배열의 경우 새로운 일반 배열 생성, 아니면 그대로 반환
	static Array<float> ToCudaArray(Array<float> arr, string desc=""); // 일반 배열의 경우 새로운 쿠다 배열 생성, 아니면 그대로 반환

	static Array<int> ToHostArray(Array<int> arr, string desc = ""); // 쿠다 배열의 경우 새로운 일반 배열 생성, 아니면 그대로 반환
	static Array<int> ToCudaArray(Array<int> arr, string desc = ""); // 일반 배열의 경우 새로운 쿠다 배열 생성, 아니면 그대로 반환

	static Array<int64> ToHostArray(Array<int64> arr, string desc = ""); // 쿠다 배열의 경우 새로운 일반 배열 생성, 아니면 그대로 반환
	static Array<int64> ToCudaArray(Array<int64> arr, string desc = ""); // 일반 배열의 경우 새로운 쿠다 배열 생성, 아니면 그대로 반환

	static Array<bool> ToHostArray(Array<bool> arr, string desc = ""); // 쿠다 배열의 경우 새로운 일반 배열 생성, 아니면 그대로 반환
	static Array<bool> ToCudaArray(Array<bool> arr, string desc = ""); // 일반 배열의 경우 새로운 쿠다 배열 생성, 아니면 그대로 반환

	static Array<short> ToHostArray(Array<short> arr, string desc = ""); // 쿠다 배열의 경우 새로운 일반 배열 생성, 아니면 그대로 반환
	static Array<short> ToCudaArray(Array<short> arr, string desc = ""); // 일반 배열의 경우 새로운 쿠다 배열 생성, 아니면 그대로 반환

	Array<float> to_cuda_array(Array<float> arr, string desc=""); // 일반 배열의 경우 새로운 쿠다 배열 생성, 아니면 그대로 반환
	Array<float> to_host_array(float* cuda_p);

	static void DumpUsage();
	static void GarbageCheck();

	float* copy(Array<float> arr, string desc="");
	float* copy_to_buffer(Array<float> arr, string desc="");
	int* copy_to_buffer(Array<int> arr, string desc = "");
	int64* copy_to_buffer(Array<int64> arr, string desc = "");
	float* attach(Array<float> arr, string desc="");

	float* alloc_float_mem(Shape shape, string desc="");
	int* alloc_int_mem(Shape shape, string desc="");
	int64* alloc_int64_mem(Shape shape, string desc="");
	unsigned char* alloc_byte_mem(Shape shape, string desc="");

	int64* alloc_int64_mem(int64* p_host, int64 size, string desc="");

	static float* Alloc_float_mem(Shape shape, string desc="");
	static int* Alloc_int_mem(Shape shape, string desc="");

	int* attach_int(Array<int> arr, string desc = "");
	int64* attach_int64(Array<int64> arr, string desc = "");
	short* attach_short(Array<short> arr, string desc = "");

	Array<float> detach(float* cuda_p, string desc="");
	Array<int> detach(int* cuda_p, string desc = "");
	Array<int64> detach(int64* cuda_p, string desc = "");

	Array<float> copy(float* cuda_p, string desc="");
	Array<float> create_farray(Shape shape, string desc="");
	Array<int> create_narray(Shape shape, string desc = "");
	Array<int64> create_n64array(Shape shape, string desc = "");

	static Array<float> CreateFloatArray(Shape shape, string desc = "");


	static float* Copy(Array<float> arr, string desc="");
	
	template <class T> static Array<T> Copy(T* cuda_p, string desc = "");

	//template <class T> static T* CopyAsPtr2Ptr(T* cuda_p, string desc = "");

	float get_nth_element(float* cuda_p, int nth64 = 0);
	void get_nth_row(float* host_dst, float* cuda_src, int64 nth_row, int64 ncol_size);
	int64 get_nth_element(int64* cuda_p, int64 nth = 0);

	string desc();

	Shape get_shape(void* cuda_p);
	static Shape GetShape(void* cuda_p);

	static void Copy_host_to_cuda(void* p_cuda, void* p_host, int64 data_size);

	float* get_host_data(Array<float> arr);

	void dump_note(string desc="");

	static void DumpShape(void* cuda_p, string desc="");

	static int64 DumpSparse(void* cuda_p, string desc, int64* pidxs = NULL, int64 max_dump_cnt = 10, int64 max_get_cnt=10, int64 ncols = -1);
	static int64 DumpZeros(void* cuda_p, string desc, int64* pidxs = NULL, int64 max_cnt = 10, int64 max_get_cnt = 10, int64 ncols = -1);

	static void DumpArr(void* cuda_p, string desc, Shape shape = Shape(), bool full = false);

	static void Print_rows(void* cuda_p, string desc, int64 nfrom = 0, int64 nto = -1, int64 col = 0);
	static void Print_selected_rows(void* cuda_p, string desc, int64* pidxs, int64 max_cnt, int64 col = 0);

protected:
	static int ms_nDevice;
	string m_name;
	Layer* m_pLayer;

	static mutex ms_mu_cudacall;

	vector<void*> m_host_mem_blocks;

	static bool ms_using_cuda;
	static curandGenerator_t ms_rand_gen;

	static string ms_to_string(void* host_p, arr_type atype, Shape shape, Idx& idx, int64 depth, bool full);
	static string ms_rows_to_string(void* host_p, arr_type atype, Shape shape, int64 nfrom = 0, int64 nto = -1, int64 col=0);
	static string ms_selected_rows_to_string(void* host_p, arr_type atype, Shape shape, int64* pidxs, int64 max_cnt, int64 col=0);
};
