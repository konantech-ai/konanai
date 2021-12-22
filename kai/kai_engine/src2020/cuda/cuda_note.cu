/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "cuda_note.h"
#include "../core/array.h"
#include "../core/log.h"

CudaNote cudanote;

CudaNote::CudaNote() {
	m_bDump = false;
}

CudaNote::~CudaNote() {
}

void CudaNote::free_all_local(CudaConn* pConn) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.begin();
	
	while (it != m_note.end()) {
		if (it->second->m_pConn == pConn) {
			map<void*, CudaNoteComponent*>::iterator curr = it++;

			CudaNoteComponent* pComponent = curr->second;
			
			m_total_size -= pComponent->m_csize;
			cudaFree(pComponent->m_cuda_p);
			m_note.erase(curr);
		}
		else {
			it++;
		}
	}
}

void CudaNote::free_global(void* cuda_p, void* pCore) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) {
		logger.Print("free_global(%lld, %lld) failure", (int64)cuda_p, (int64)pCore);
		dump("free_global failure");
		throw KaiException(KERR_ASSERT);
	}
	CudaNoteComponent* pComponent = it->second;
	if (pComponent->m_pConn != NULL) throw KaiException(KERR_ASSERT);
	if (pComponent->m_pDataCore != pCore) throw KaiException(KERR_ASSERT);

	cudaFree(cuda_p);

	m_total_size -= pComponent->m_csize;
	string desc = pComponent->m_desc;
	m_note.erase(it);
	m_dump("free_global(" + desc + ") is called");
}

void CudaNote::restore_global(float* cuda_p, void* pCore) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);
	CudaNoteComponent* pComponent = it->second;
	if (pComponent->m_pConn != NULL) throw KaiException(KERR_ASSERT);
	if (pComponent->m_pDataCore != pCore) throw KaiException(KERR_ASSERT);

	ArrayDataCore<float>* core = (ArrayDataCore<float> *) pComponent->m_pDataCore;

#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
#else
	core->m_data = (float*) malloc(pComponent->m_msize);
#endif
	core->m_isCuda = false;
	
	cudaMemcpy(core->m_data, cuda_p, pComponent->m_msize, cudaMemcpyDeviceToHost);
	cudaFree(cuda_p);

	m_total_size -= pComponent->m_csize;
	string desc = pComponent->m_desc;
	m_note.erase(it);
	
	m_dump("restore_global(" + desc + ") is restored");
}

void CudaNote::existing_global_check(void* cuda_p) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it != m_note.end()) return;
	throw KaiException(KERR_ASSERT);
}

void CudaNote::not_existing_global_check(void* cuda_p) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) return;
	logger.Print("cuda_p = %lld", (int64)cuda_p);
	m_dump("not_existing_global_check() failed", true);
	logger.Print("cuda_p = %lld", (int64)cuda_p);
	throw KaiException(KERR_ASSERT);
}

void CudaNote::allocate_global(Array<float> arr, string desc) {
	float* cuda_p;

	int64 msize = arr.total_size() * sizeof(float);
	int64 csize = (msize + 15) / 16 * 16;
	cudaMalloc(&cuda_p, csize);
	assert(cuda_p != NULL);
	cudaMemcpy(cuda_p, arr.m_core->m_mdata->m_data, msize, cudaMemcpyHostToDevice);

	free(arr.m_core->m_mdata->m_data);
	arr.m_core->m_mdata->m_data = cuda_p;
	arr.m_core->m_mdata->m_isCuda = true;

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = arr.shape();
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_float;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = NULL;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("allocate_global(" + desc + ") is called");
}

void CudaNote::allocate_global(Array<int> arr, string desc) {
	int* cuda_p;

	int64 msize = arr.total_size() * sizeof(int);
	int64 csize = (msize + 15) / 16 * 16;
	cudaMalloc(&cuda_p, csize);
	assert(cuda_p != NULL);
	cudaMemcpy(cuda_p, arr.m_core->m_mdata->m_data, msize, cudaMemcpyHostToDevice);

	free(arr.m_core->m_mdata->m_data);
	arr.m_core->m_mdata->m_data = cuda_p;
	arr.m_core->m_mdata->m_isCuda = true;

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = arr.shape();
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_int;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = NULL;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("allocate_global(" + desc + ") is called");
}

void CudaNote::allocate_global(Array<int64> arr, string desc) {
	int64* cuda_p;

	int64 msize = arr.total_size() * sizeof(int64);
	int64 csize = (msize + 15) / 16 * 16;
	cudaMalloc(&cuda_p, csize);
	assert(cuda_p != NULL);
	cudaMemcpy(cuda_p, arr.m_core->m_mdata->m_data, msize, cudaMemcpyHostToDevice);

	free(arr.m_core->m_mdata->m_data);
	arr.m_core->m_mdata->m_data = cuda_p;
	arr.m_core->m_mdata->m_isCuda = true;

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = arr.shape();
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_int64;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = NULL;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("allocate_global(" + desc + ") is called");
}

float* CudaNote::alloc_float_mem(CudaConn* pConn, Shape shape, string desc) {
	float* cuda_p;

	int64 msize = shape.total_size() * sizeof(float);
	int64 csize = (msize + 15) / 16 * 16;

	if (csize <= 0) {
		logger.Print("device memory allocation of zero or negative size(pConn = %s, shape = %s, desc = %s) was requested", pConn->desc().c_str(), shape.desc().c_str(), desc.c_str());
		throw KaiException(KERR_ASSERT);
		return NULL;
	}

	cudaMalloc(&cuda_p, csize);
	cudaMemset(cuda_p, 0, csize);
	if (cuda_p == NULL) {
		//dump("alloc_float_mem failure");
		logger.Print("*** alloc_float_mem failure ***");
		logger.Print("alloc_float_mem(pConn = %s, shape = %s, desc = %s) was called", pConn->desc().c_str(), shape.desc().c_str(), desc.c_str());
		cudaError_t cuda_ret = cudaGetLastError();
		if (cuda_ret != cudaSuccess) {
			logger.Print("cudaError: %s", cudaGetErrorString(cuda_ret));
		}
		throw KaiException(KERR_ASSERT);
	}

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = NULL;
	pComponent->m_type = arr_type::arr_float;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("alloc_float_mem(" + desc + ") is called");

	return cuda_p;
}

int* CudaNote::alloc_int_mem(CudaConn* pConn, Shape shape, string desc) {
	int* cuda_p;

	int64 msize = shape.total_size() * sizeof(int);
	int64 csize = (msize + 15) / 16 * 16;
	cudaMalloc(&cuda_p, csize);
	cudaMemset(cuda_p, 0, csize);
	if (cuda_p == NULL) {
		logger.Print("alloc_int_mem() failure on shape %s", shape.desc().c_str());
		throw KaiException(KERR_ASSERT);
	}

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = NULL;
	pComponent->m_type = arr_type::arr_int;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("alloc_int_mem(" + desc + ") is called");

	return cuda_p;
}

int64* CudaNote::alloc_int64_mem(CudaConn* pConn, Shape shape, string desc) {
	int64* cuda_p;

	int64 msize = shape.total_size() * sizeof(int64);
	int64 csize = (msize + 15) / 16 * 16;
	cudaMalloc(&cuda_p, csize);
	cudaMemset(cuda_p, 0, csize);
	if (cuda_p == NULL) {
		logger.Print("alloc_int_mem() failure on shape %s", shape.desc().c_str());
		throw KaiException(KERR_ASSERT);
	}

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = NULL;
	pComponent->m_type = arr_type::arr_int64;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("alloc_int_mem(" + desc + ") is called");

	return cuda_p;
}

int64* CudaNote::alloc_int64_mem(CudaConn* pConn, int64* host_p, int64 size, string desc) {
	int64* cuda_p;

	int64 msize = size * sizeof(int64);
	int64 csize = (msize + 15) / 16 * 16;
	cudaMalloc(&cuda_p, csize);
	if (cuda_p == NULL) {
		logger.Print("alloc_int_mem() failure on size %lld", size);
		pConn->cuda_check("alloc_int_mem");
		//dump("alloc_int_mem() failure");
		throw KaiException(KERR_ASSERT);
	}

	cudaMemcpy(cuda_p, host_p, size * sizeof(int64), cudaMemcpyHostToDevice);

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = Shape(size);
	pComponent->m_pDataCore = NULL;
	pComponent->m_type = arr_type::arr_int64;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("alloc_int_mem(" + desc + ") is called");

	return cuda_p;
}

unsigned char* CudaNote::alloc_byte_mem(CudaConn* pConn, Shape shape, string desc) {
	unsigned char* cuda_p;

	int64 msize = shape.total_size() * sizeof(unsigned char);
	int64 csize = (msize + 15) / 16 * 16;
	cudaMalloc(&cuda_p, csize);
	cudaMemset(cuda_p, 0, csize);
	if (cuda_p == NULL) {
		logger.Print("alloc_int_mem() failure on shape %s", shape.desc().c_str());
		throw KaiException(KERR_ASSERT);
	}

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = NULL;
	pComponent->m_type = arr_type::arr_uchar;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("alloc_byte_mem(" + desc + ") is called");

	return cuda_p;
}

float* CudaNote::copy(CudaConn* pConn, Array<float> arr, string desc) {
	float* cuda_p = arr.m_core->m_mdata->m_data;

	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	float* clone_p;
	cudaMalloc(&clone_p, pComponent->m_csize);
	assert(clone_p != NULL);

	cudaMemcpy(clone_p, cuda_p, pComponent->m_csize, cudaMemcpyDeviceToDevice);

	CudaNoteComponent* pClone = new CudaNoteComponent;

	pClone->m_cuda_p = clone_p;
	pClone->m_shape = arr.shape();
	pClone->m_pDataCore = arr.m_core->m_mdata;
	pClone->m_type = arr_type::arr_float;
	pClone->m_msize = pComponent->m_msize;
	pClone->m_csize = pComponent->m_csize;
	pClone->m_desc = desc + ".copied." + pComponent->m_desc;
	pClone->m_pConn = pConn;

	m_total_size += pClone->m_csize;
	m_note[clone_p] = pClone;

	m_dump("copy to cuda_ptr is called");

	return clone_p;
}

float* CudaNote::attach(CudaConn* pConn, Array<float> arr, string desc) {
	if (!arr.m_core->m_mdata->m_isCuda) {
		logger.Print("attach(%s, %s) is called for non-cuda array", pConn->desc().c_str(), desc.c_str());
		throw KaiException(KERR_ASSERT);
	}

	float* cuda_p = arr.m_core->m_mdata->m_data;

	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) {
		logger.Print("Bad attach(%s, %s) is called for unregistered cuda array", pConn->desc().c_str(), desc.c_str());
		throw KaiException(KERR_ASSERT);
	}

	CudaNoteComponent* pComponent = it->second;

	arr.m_core->m_mdata->m_data = NULL;
	arr.m_core->m_mdata->m_isCuda = false;

	pComponent->m_pDataCore = NULL;
	pComponent->m_desc += "." + desc;
	pComponent->m_pConn = pConn;

	m_dump("attach(" + desc + ") is called");

	return cuda_p;
}

int* CudaNote::attach_int(CudaConn* pConn, Array<int> arr, string desc) {
	if (!arr.m_core->m_mdata->m_isCuda) {
		logger.Print("attach(%s, %s) is called for non-cuda array", pConn->desc().c_str(), desc.c_str());
		throw KaiException(KERR_ASSERT);
	}

	int* cuda_p = arr.m_core->m_mdata->m_data;

	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	arr.m_core->m_mdata->m_data = NULL;
	arr.m_core->m_mdata->m_isCuda = false;

	pComponent->m_pDataCore = NULL;
	pComponent->m_desc += "." + desc;
	pComponent->m_pConn = pConn;

	m_dump("attach(" + desc + ") is called");

	return cuda_p;
}

int64* CudaNote::attach_int64(CudaConn* pConn, Array<int64> arr, string desc) {
	if (!arr.m_core->m_mdata->m_isCuda) {
		logger.Print("attach(%s, %s) is called for non-cuda array", pConn->desc().c_str(), desc.c_str());
		throw KaiException(KERR_ASSERT);
	}

	int64* cuda_p = arr.m_core->m_mdata->m_data;

	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	arr.m_core->m_mdata->m_data = NULL;
	arr.m_core->m_mdata->m_isCuda = false;

	pComponent->m_pDataCore = NULL;
	pComponent->m_desc += "." + desc;
	pComponent->m_pConn = pConn;

	m_dump("attach(" + desc + ") is called");

	return cuda_p;
}

short* CudaNote::attach_short(CudaConn* pConn, Array<short> arr, string desc) {
	if (!arr.m_core->m_mdata->m_isCuda) {
		logger.Print("attach(%s, %s) is called for non-cuda array", pConn->desc().c_str(), desc.c_str());
		throw KaiException(KERR_ASSERT);
	}

	short* cuda_p = arr.m_core->m_mdata->m_data;

	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	arr.m_core->m_mdata->m_data = NULL;
	arr.m_core->m_mdata->m_isCuda = false;

	pComponent->m_pDataCore = NULL;
	pComponent->m_desc += "." + desc;
	pComponent->m_pConn = pConn;

	m_dump("attach(" + desc + ") is called");

	return cuda_p;
}

Array<float> CudaNote::create_farray(CudaConn* pConn, Shape shape, string desc) {
	float* cuda_p;

	int64 msize = shape.total_size() * sizeof(float);
	int64 csize = (msize + 15) / 16 * 16;

	cudaMalloc(&cuda_p, csize);
	
	if (cuda_p == NULL) {
		logger.Print("create_farray(%s, %s) failed: csize = %lld", desc.c_str(), Value::description(shape).c_str(), csize);
		m_dump("create_farray is called", true);
		cudaError_t cuda_ret = cudaGetLastError();
		logger.Print("cudaError: %s", cudaGetErrorString(cuda_ret));
		throw KaiException(KERR_ASSERT);
	}

	cudaMemset(cuda_p, 0, csize);

	Array<float> arr;
	arr.m_core->m_dimension = Dim(shape);
	arr.m_core->m_mdata = new ArrayDataCore<float>(shape.total_size(), cuda_p);

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_float;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("create_farray is called");

	return arr;
}

Array<int> CudaNote::create_narray(CudaConn* pConn, Shape shape, string desc) {
	int* cuda_p;

	int64 msize = shape.total_size() * sizeof(int);
	int64 csize = (msize + 15) / 16 * 16;

	cudaMalloc(&cuda_p, csize);
	assert(cuda_p != NULL);
	cudaMemset(cuda_p, 0, csize);

	Array<int> arr;
	arr.m_core->m_dimension = Dim(shape);
	arr.m_core->m_mdata = new ArrayDataCore<int>(shape.total_size(), cuda_p);

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_int;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("create_farray is called");

	return arr;
}

Array<int64> CudaNote::create_n64array(CudaConn* pConn, Shape shape, string desc) {
	int64* cuda_p;

	int64 msize = shape.total_size() * sizeof(int64);
	int64 csize = (msize + 15) / 16 * 16;

	cudaMalloc(&cuda_p, csize);
	assert(cuda_p != NULL);
	cudaMemset(cuda_p, 0, csize);

	Array<int64> arr;
	arr.m_core->m_dimension = Dim(shape);
	arr.m_core->m_mdata = new ArrayDataCore<int64>(shape.total_size(), cuda_p);

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_int64;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("create_farray is called");

	return arr;
}

Array<bool> CudaNote::create_barray(CudaConn* pConn, Shape shape, string desc) {
	bool* cuda_p;

	int64 msize = shape.total_size() * sizeof(bool);
	int64 csize = (msize + 15) / 16 * 16;

	cudaMalloc(&cuda_p, csize);
	assert(cuda_p != NULL);
	cudaMemset(cuda_p, 0, csize);

	Array<bool> arr;
	arr.m_core->m_dimension = Dim(shape);
	arr.m_core->m_mdata = new ArrayDataCore<bool>(shape.total_size(), cuda_p);

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_bool;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("create_barray is called");

	return arr;
}

Array<short> CudaNote::create_sarray(CudaConn* pConn, Shape shape, string desc) {
	short* cuda_p;

	int64 msize = shape.total_size() * sizeof(short);
	int64 csize = (msize + 15) / 16 * 16;

	cudaMalloc(&cuda_p, csize);
	assert(cuda_p != NULL);
	cudaMemset(cuda_p, 0, csize);

	Array<short> arr;
	arr.m_core->m_dimension = Dim(shape);
	arr.m_core->m_mdata = new ArrayDataCore<short>(shape.total_size(), cuda_p);

	CudaNoteComponent* pComponent = new CudaNoteComponent;

	pComponent->m_cuda_p = cuda_p;
	pComponent->m_shape = shape;
	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_type = arr_type::arr_short;
	pComponent->m_msize = msize;
	pComponent->m_csize = csize;
	pComponent->m_desc = desc;
	pComponent->m_pConn = pConn;

	m_total_size += csize;
	m_note[cuda_p] = pComponent;

	m_dump("create_barray is called");

	return arr;
}

Array<float> CudaNote::detach(CudaConn* pConn, float* cuda_p, string desc) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	if (pComponent->m_pConn != pConn) {
		logger.Print("Bad detach(%s) cuda_p(%s) not belong to CudaConn(%s): belong to %s", desc.c_str(), pComponent->m_desc.c_str(), pConn->desc().c_str(), pComponent->m_pConn->desc().c_str());
		throw KaiException(KERR_ASSERT);
	}
	if (pComponent->m_type != arr_type::arr_float) throw KaiException(KERR_ASSERT);

	int64 size = pComponent->m_msize / sizeof(float);

	Array<float> arr;
	arr.m_core->m_dimension = Dim(pComponent->m_shape);
	arr.m_core->m_mdata = new ArrayDataCore<float>(size, cuda_p);

	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_desc = desc;
	pComponent->m_pConn = NULL;

	m_dump("detach is called");

	return arr;
}

Array<int> CudaNote::detach(CudaConn* pConn, int* cuda_p, string desc) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	if (pComponent->m_pConn != pConn) throw KaiException(KERR_ASSERT);
	if (pComponent->m_type != arr_type::arr_int) throw KaiException(KERR_ASSERT);

	int64 size = pComponent->m_msize / sizeof(int);

	Array<int> arr;
	arr.m_core->m_dimension = Dim(pComponent->m_shape);
	arr.m_core->m_mdata = new ArrayDataCore<int>(size, cuda_p);

	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_desc = desc;
	pComponent->m_pConn = NULL;

	m_dump("detach for int is called");

	return arr;
}

Array<int64> CudaNote::detach(CudaConn* pConn, int64* cuda_p, string desc) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	if (pComponent->m_pConn != pConn) throw KaiException(KERR_ASSERT);
	if (pComponent->m_type != arr_type::arr_int64) throw KaiException(KERR_ASSERT);

	int64 size = pComponent->m_msize / sizeof(int64);

	Array<int64> arr;
	arr.m_core->m_dimension = Dim(pComponent->m_shape);
	arr.m_core->m_mdata = new ArrayDataCore<int64>(size, cuda_p);

	pComponent->m_pDataCore = arr.m_core->m_mdata;
	pComponent->m_desc = desc;
	pComponent->m_pConn = NULL;

	m_dump("detach for int64 is called");

	return arr;
}

template <class T> Array<T> CudaNote::copy(CudaConn* pConn, T* cuda_p, string desc) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	T* clone_p;
	cudaMalloc(&clone_p, pComponent->m_csize);
	if (clone_p == NULL) {
		logger.Print("Cuda memory allocation failure in copy(%s, %s, %lld): allocated size = %lld", pConn->desc().c_str(), desc.c_str(), pComponent->m_csize, m_total_size);
	}

	cudaMemcpy(clone_p, cuda_p, pComponent->m_csize, cudaMemcpyDeviceToDevice);

	int64 size = pComponent->m_msize / sizeof(T);

	Array<T> arr;
	arr.m_core->m_dimension = Dim(pComponent->m_shape);
	arr.m_core->m_mdata = new ArrayDataCore<T>(size, clone_p);

	CudaNoteComponent* pClone = new CudaNoteComponent;

	pClone->m_cuda_p = clone_p;
	pClone->m_shape = pComponent->m_shape;
	pClone->m_pDataCore = arr.m_core->m_mdata;
	pClone->m_type = ms_getArrTyoe(cuda_p);
	pClone->m_msize = pComponent->m_msize;
	pClone->m_csize = pComponent->m_csize;
	pClone->m_desc = desc+".copied."+pComponent->m_desc;
	pClone->m_pConn = pConn;

	m_total_size += pClone->m_csize;
	m_note[clone_p] = pClone;

	m_dump("copy to array is called");

	return arr;
}

/*
template <class T> T* CudaNote::copy_as_p2p(CudaConn* pConn, T* cuda_p, string desc) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	T* clone_p;
	cudaMalloc(&clone_p, pComponent->m_csize);
	if (clone_p == NULL) {
		logger.Print("Cuda memory allocation failure in copy(%s, %s, %lld): allocated size = %lld", pConn->desc().c_str(), desc.c_str(), pComponent->m_csize, m_total_size);
	}

	cudaMemcpy(clone_p, cuda_p, pComponent->m_csize, cudaMemcpyDeviceToDevice);

	CudaNoteComponent* pClone = new CudaNoteComponent;

	pClone->m_cuda_p = clone_p;
	pClone->m_shape = pComponent->m_shape;
	pClone->m_pDataCore = NULL;
	pClone->m_type = ms_getArrTyoe(cuda_p);
	pClone->m_msize = pComponent->m_msize;
	pClone->m_csize = pComponent->m_csize;
	pClone->m_desc = desc + ".copied." + pComponent->m_desc;
	pClone->m_pConn = pConn;

	m_total_size += pClone->m_csize;
	m_note[clone_p] = pClone;

	m_dump("copy to array is called");

	return clone_p;
}
*/

int64 CudaNote::get_msize(void* cuda_p) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	int64 msize = 0;

	if (pComponent->m_type == arr_type::arr_float) msize = pComponent->m_msize / sizeof(float);
	else if (pComponent->m_type == arr_type::arr_int) msize = pComponent->m_msize / sizeof(int);
	else if (pComponent->m_type == arr_type::arr_int64) msize = pComponent->m_msize / sizeof(int64);
	else if (pComponent->m_type == arr_type::arr_bool) msize = pComponent->m_msize / sizeof(bool);
	else if (pComponent->m_type == arr_type::arr_uchar) msize = pComponent->m_msize / sizeof(unsigned char);
	else if (pComponent->m_type == arr_type::arr_short) msize = pComponent->m_msize / sizeof(short);
	else throw KaiException(KERR_ASSERT);

	return msize;
}

Shape CudaNote::get_shape(void* cuda_p) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	return pComponent->m_shape;
}

arr_type CudaNote::get_type(void* cuda_p) {
	map<void*, CudaNoteComponent*>::iterator it = m_note.find(cuda_p);
	if (it == m_note.end()) throw KaiException(KERR_ASSERT);

	CudaNoteComponent* pComponent = it->second;

	return pComponent->m_type;
}

void CudaNote::dump_usage() {
	logger.Print("Cuda Memory allocated: %lld bytes", m_total_size);
}

void CudaNote::garbage_check() {
	if (m_note.size() > 0) {
		m_dump("garbage_check failure", true);
		//throw KaiException(KERR_ASSERT);
	}
}

void CudaNote::m_dump(string event, bool always) {
	if (!m_bDump && !always) return;

	logger.Print("********************************************************************************");
	logger.Print("   CudaNote: %s (total size: %lld)", event.c_str(), m_total_size);
	logger.Print("********************************************************************************");

	for (map<void*, CudaNoteComponent*>::iterator it = m_note.begin(); it != m_note.end(); it++) {
		void* cuda_p = it->first;
		CudaNoteComponent* pComponent = it->second;

		logger.Print("%lld: %lld %s=%lld/%lld, %s, %s", (int64) cuda_p, (int64) pComponent->m_cuda_p,
			pComponent->m_shape.desc().c_str(), pComponent->m_msize, pComponent->m_csize, pComponent->m_desc.c_str(), pComponent->m_pConn->desc().c_str());
	}
}

template Array<float> CudaNote::copy(CudaConn* pConn, float* cuda_p, string desc);
template Array<int> CudaNote::copy(CudaConn* pConn, int* cuda_p, string desc);
template Array<int64> CudaNote::copy(CudaConn* pConn, int64* cuda_p, string desc);
template Array<bool> CudaNote::copy(CudaConn* pConn, bool* cuda_p, string desc);
template Array<unsigned char> CudaNote::copy(CudaConn* pConn, unsigned char* cuda_p, string desc);
template Array<short> CudaNote::copy(CudaConn* pConn, short* cuda_p, string desc);

/*
template float* CudaNote::copy_as_p2p(CudaConn* pConn, float* cuda_p, string desc);
template int* CudaNote::copy_as_p2p(CudaConn* pConn, int* cuda_p, string desc);
template int64* CudaNote::copy_as_p2p(CudaConn* pConn, int64* cuda_p, string desc);
template bool* CudaNote::copy_as_p2p(CudaConn* pConn, bool* cuda_p, string desc);
template unsigned char* CudaNote::copy_as_p2p(CudaConn* pConn, unsigned char* cuda_p, string desc);
template short* CudaNote::copy_as_p2p(CudaConn* pConn, short* cuda_p, string desc);
*/
