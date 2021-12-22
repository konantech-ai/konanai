/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/common.h"
#include "../core/array.h"

class CudaConn;

class CudaNoteComponent {
public:
	CudaNoteComponent() {}
	virtual ~CudaNoteComponent() {}

protected:
	friend class CudaNote;

	void* m_cuda_p;
	Shape m_shape;
	int64 m_msize;	// 실제 필요한 메모리 크기
	int64 m_csize;	// 할당한 메모리 크기 (16의 배수)
	string m_desc;
	arr_type m_type;
	void* m_pDataCore;
	CudaConn* m_pConn; // NULL is global
};

class CudaNote {
public:
	CudaNote();
	virtual ~CudaNote();

	void free_all_local(CudaConn* pConn);
	void free_global(void* cuda_p, void* pCore);
	void restore_global(float* cuda_p, void* pCore);

	void existing_global_check(void* cuda_p);
	void not_existing_global_check(void* cuda_p);

	void allocate_global(Array<float> arr, string desc);
	void allocate_global(Array<int> arr, string desc);
	void allocate_global(Array<int64> arr, string desc);

	float* alloc_float_mem(CudaConn* pConn, Shape shape, string desc);

	float* attach(CudaConn* pConn, Array<float> arr, string desc);

	float* copy(CudaConn* pConn, Array<float> arr, string desc);

	template <class T> Array<T> copy(CudaConn* pConn, T* cuda_p, string desc);

	//template <class T> T* copy_as_p2p(CudaConn* pConn, T* cuda_p, string desc);

	int* alloc_int_mem(CudaConn* pConn, Shape shape, string desc);
	int64* alloc_int64_mem(CudaConn* pConn, Shape shape, string desc);
	int64* alloc_int64_mem(CudaConn* pConn, int64* p_host, int64 size, string desc);
	unsigned char* alloc_byte_mem(CudaConn* pConn, Shape shape, string desc);
	int* attach_int(CudaConn* pConn, Array<int> arr, string desc);
	int64* attach_int64(CudaConn* pConn, Array<int64> arr, string desc);
	short* attach_short(CudaConn* pConn, Array<short> arr, string desc);

	Array<float> create_farray(CudaConn* pConn, Shape shape, string desc);
	Array<int> create_narray(CudaConn* pConn, Shape shape, string desc);
	Array<int64> create_n64array(CudaConn* pConn, Shape shape, string desc);
	Array<bool> create_barray(CudaConn* pConn, Shape shape, string desc);
	Array<short> create_sarray(CudaConn* pConn, Shape shape, string desc);

	Array<float> detach(CudaConn* pConn, float* cuda_p, string desc);
	Array<int> detach(CudaConn* pConn, int* cuda_p, string desc);
	Array<int64> detach(CudaConn* pConn, int64* cuda_p, string desc);

	void dump_usage();
	void garbage_check();

	void dump(string event) { m_dump("Extern:" + event, true); }

	int64 get_msize(void* cuda_p);
	Shape get_shape(void* cuda_p);
	arr_type get_type(void* cuda_p);

	int64 getUsingMemSize() { return m_total_size; }

protected:
	int64 m_total_size;
	map<void*, CudaNoteComponent*> m_note;

	bool m_bDump;

	void m_dump(string event, bool always = false);
	
	static arr_type ms_getArrTyoe(float* cuda_p) { return arr_type::arr_float; }
	static arr_type ms_getArrTyoe(int* cuda_p) { return arr_type::arr_int; }
	static arr_type ms_getArrTyoe(int64* cuda_p) { return arr_type::arr_int64; }
	static arr_type ms_getArrTyoe(bool* cuda_p) { return arr_type::arr_bool; }
	static arr_type ms_getArrTyoe(unsigned char* cuda_p) { return arr_type::arr_uchar; }
	static arr_type ms_getArrTyoe(short* cuda_p) { return arr_type::arr_short; }
};

extern CudaNote cudanote;