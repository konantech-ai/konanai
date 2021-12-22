/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "kshape.h"
#include "../library/object.h"
#include "../gpu_cuda/device_manager.h"

class KaiSession;

extern kmutex_wrap ms_mu_karr_data;
extern kmutex_wrap ms_mu_karr_core;

template <class T> class KaiArray;
template <class T> class KaiArrayCore;

template <class T>
class KaiArrayData {
public:
	KaiArrayData(KInt size) {
		ms_mu_karr_data.lock();
		m_nRefCount = 1;
		m_size = size * sizeof(T);
		m_data = (T*)malloc(m_size);
		m_pDevMan = NULL;
		ms_mu_karr_data.unlock();
	}

	KaiArrayData(T* cuda_p, KInt size, KaiDeviceManager* pDevMan) {
		ms_mu_karr_data.lock();
		m_nRefCount = 1;
		m_size = size; 
		m_data = cuda_p;
		m_pDevMan = pDevMan;
		ms_mu_karr_data.unlock();
	}

	virtual ~KaiArrayData() {
		if (m_pDevMan) m_pDevMan->freeMemory(m_data, m_size);
		else free(m_data);
	}

	void destroy() {
		ms_mu_karr_data.lock();
		bool del = (this && --m_nRefCount <= 0);
		ms_mu_karr_data.unlock();
		if (del) delete this;
	}

protected:
	friend class KaiArrayCore<T>;
	friend class KaiArray<T>;
	//friend class CudaConn;
	//friend class CudaNote;

	KInt m_nRefCount;
	KInt m_size;
	T* m_data;
	KaiDeviceManager* m_pDevMan;
};

template <class T>
class KaiArrayCore : public KaiObject {
public:
	KaiArrayCore();
	virtual ~KaiArrayCore() { if (m_mdata != 0) m_mdata->destroy(); }
	/*
	void destroy() {
		ms_mu_karr_core.lock();
		bool del = (this && --m_nRefCount <= 0);
		ms_mu_karr_core.unlock();
		if (del) delete this;
	}
	*/

	Ken_object_type get_type();

	KString desc();

	static KaiArrayCore* HandleToPointer(KHObject hObject);
	static KaiArrayCore* HandleToPointer(KHObject hObject, KaiSession* pSession);

	void get_data(KInt nStart, KInt nCount, T* pBuffer);

protected:
	T* data() { return m_mdata->m_data; }
	KInt data_size() { return m_mdata->m_size; }
	KaiArrayData<T>* share_data() {
		ms_mu_karr_data.lock();
		m_mdata->m_nRefCount++;
		ms_mu_karr_data.unlock();
		return m_mdata;
	}

protected:
	friend class KaiArray<T>;
	//friend class ArrayLoopAxisInfo<T>;
	//friend class HostMath;
	//friend class Value;
	//friend class CudaConn;
	//friend class CudaNote;

	//KInt m_nRefCount;
	KaiShape m_shape;
	//Dim m_dimension;
	KaiArrayData<T>* m_mdata;
	int m_checkCode;
	static int ms_checkCode;
};

template <class T>
class KaiArray {
protected:
	KaiArrayCore<T>* m_core;

public:
	KaiArray() {
		ms_mu_karr_core.lock();
		m_core = new KaiArrayCore<T>();
		ms_mu_karr_core.unlock();
	}
	KaiArray(const KaiArray& src) {
		ms_mu_karr_core.lock();
		m_core = src.m_core;
		m_core->m_nRefCount++;
		ms_mu_karr_core.unlock();
	}
	KaiArray(KaiShape shape) {
		KInt size = shape.total_size();
		ms_mu_karr_core.lock();
		m_core = new KaiArrayCore<T>();
		m_core->m_shape = shape;
		m_core->m_mdata = new KaiArrayData<T>(size);
		ms_mu_karr_core.unlock();
	}
	KaiArray(KaiArrayCore<T>* core) {
		ms_mu_karr_core.lock();
		m_core = core;
		core->m_nRefCount++;
		ms_mu_karr_core.unlock();
	}
	KaiArray(T* cuda_p, KaiShape shape, KInt size, KaiDeviceManager* pDevMan) {
		ms_mu_karr_core.lock();
		m_core = new KaiArrayCore<T>();
		m_core->m_shape = shape;
		m_core->m_mdata = new KaiArrayData<T>(cuda_p, size, pDevMan);
		ms_mu_karr_core.unlock();
	}

	virtual ~KaiArray() {
		m_core->destroy();
		m_core = NULL;
	}

	KaiArray(KaiShape shape, T* values) {
		KInt size = shape.total_size();
		ms_mu_karr_core.lock();
		m_core = new KaiArrayCore<T>();
		m_core->m_shape = shape;
		m_core->m_mdata = new KaiArrayData<T>(size);
		memcpy(m_core->m_mdata->m_data, values, sizeof(T) * size);
		ms_mu_karr_core.unlock();
	}

	//KaiArray(List list);

	//KaiArray  data_share_clone();

	// 핸들 획득해 저장해야 하는 경우를 위한 것이나 릭 발생 가능성이 큼, 개선된 메카니즘 개발 혹은 사용처 추적 및 확인 필요
	KaiArrayCore<T>* get_core() {
		//m_core->m_nRefCount++;
		return m_core;
	}

	KaiArray& operator = (const KaiArray& src) {
		if (this == &src) return *this;
		m_core->destroy();
		ms_mu_karr_core.lock();
		m_core = src.m_core;
		m_core->m_nRefCount++;
		ms_mu_karr_core.unlock();
		return *this;
	}

	static KString element_type_name(); // return "KInt" or "KFloat"
	static const char* elem_format();
	static const char* py_typename(); // return "int" or float"

	static KaiArray  zeros(KaiShape shape);
	static KaiArray  ones(KaiShape shape, T coef = 1);

	T get_at(KInt nth1, KInt nth2 = -1, KInt nth3 = -1, KInt nth4 = -1, KInt nth5 = -1, KInt nth6 = -1, KInt nth7 = -1, KInt nth8 = -1) const;
	T& set_at(KInt nth1, KInt nth2 = -1, KInt nth3 = -1, KInt nth4 = -1, KInt nth5 = -1, KInt nth6 = -1, KInt nth7 = -1, KInt nth8 = -1);

	bool is_empty() { return m_core->m_mdata == NULL; }
	bool is_cuda() { return m_core->m_mdata->m_pDevMan != NULL; }

	KaiShape shape() const { return m_core->m_shape; }

	KInt dim() const { return m_core->m_shape.size(); }
	KInt axis_size(KInt axis) const { return m_core->m_shape[axis]; }
	KInt total_size() const { return m_core->m_shape.total_size(); }
	KInt mem_size() { return m_core->m_mdata ? m_core->m_mdata->m_size : 0; }

	T* data_ptr() { return m_core->data(); }

	KaiArray to_host();

	KaiArray reshape(KaiShape shape) {
		assert(m_core->m_shape.total_size() == shape.total_size());
		KaiArray clone;
		clone.m_core->m_shape = shape;
		clone.m_core->m_mdata = m_core->m_mdata;
		clone.m_core->m_mdata->m_nRefCount++;
		return clone;
	}

	void dump(KString sTitlt, KBool bFull = false);
	/*
	void reset(); // set to zero(false) all array elements

	KaiArray transpose();
	KaiArray transpose(Idx idx);
	KaiArray reshape(Shape shape);
	KaiArray init_grid(Shape shape);
	KaiArray tile(int64 num);
	KaiArray untile(int64 num);
	KaiArray flatten();
	KaiArray maxarg(int64 axis, KaiArray<int64>& arg);
	KaiArray sum(int64 axis);
	KaiArray avg(int64 axis);
	KaiArray var(int64 axis, KaiArray* pavg = NULL);
	KaiArray square();
	KaiArray abs();
	KaiArray<int64> round();
	KaiArray<int64> binary_row_to_int();
	KaiArray dotsum(KaiArray other);
	//KaiArray sort(sortdir dir = sortdir::sd_asc, KaiArray<int64>& sort_idx = NULL);
	//KaiArray sort(sortdir dir, KaiArray<int64>& sort_idx);
	KaiArray<float> to_float();
	KaiArray logical_or(KaiArray other);
	KaiArray extract_selected(KaiArray<bool> selector);
	KaiArray fill_selected(KaiArray other, KaiArray<bool> selector);
	KaiArray fetch_rows(vector<int64> rows);
	KaiArray wvec_select(KaiArray<int64> other);
	KaiArray wvec_select_idx(KaiArray<int64> other, int64 dic_count, int64* voc_counts);
	KaiArray select_col(KaiArray<int64> selector);
	KaiArray minus_1_on_idx(KaiArray<int64> selector);
	KaiArray minus_1_on_1st();
	KaiArray get_row(int64 nth);
	KaiArray get_col(int64 nth);
	KaiArray filter(int64 axis, int64 index);
	KaiArray mult_dim(int64 axis = -1);
	KaiArray unfilter(KaiArray filtered, int64 axis, int64 index);
	vector<KaiArray> hsplit();

	KaiArray expand_images(Shape stride);
	KaiArray expand_undo_images(Shape stride);


	void set_row(int64 nth, KaiArray other);
	void set_row(int64 nth, T other);
	void sub_row(int64 nth, KaiArray other);

	KaiArray<int> to_int();
	KaiArray<int64> to_int64();

	int64 true_count();

	void split_axis_size(int64 axis, int64* psize1, int64* psize2, int64* psize3);

	KaiArray merge_time_axis();
	KaiArray split_time_axis(int64 mb_size);

	KaiArray add_axis(int64 pos);

	int64 count_in_range(T min_val, T max_val);

	T sum();

	void setarg(KaiArray<int64>& arg, KaiArray values);

	bool is_empty() const { return dim() == 0; }

	KaiArray deepcopy() const;

	KaiArray to_host();

	Dim dimension() { return m_core->m_dimension; }



	void copy(Idx dest_idx, KaiArray src, Idx src_idx);
	void copy(KaiArray src);

	ArrayCopyInfo<T> operator [](Axis axis) { return ArrayCopyInfo<T>(*this, axis); }
	ArrayCopyInfo<T> operator [](Axis axis) const { return ArrayCopyInfo<T>((KaiArray&)*this, axis); }

	void get(KaiArray src, Axis axis);

	static int64 element_size() { return  sizeof(T); }


	ArrayLoopAxisInfo<T> operator [](int64 nth);

	T& operator [](Idx idx);
	T operator [](Idx idx) const;

	KaiArray operator +=(const KaiArray other);
	KaiArray operator -=(const KaiArray other);
	KaiArray operator *=(const KaiArray other);
	KaiArray operator /=(const KaiArray other);

	KaiArray operator +=(T other);
	KaiArray operator -=(T other);
	KaiArray operator *=(T other);
	KaiArray operator /=(T other);

	KaiArray operator +(const KaiArray other) const;
	KaiArray operator -(const KaiArray other) const;
	KaiArray operator *(const KaiArray other) const;
	KaiArray operator /(const KaiArray other) const;

	KaiArray operator +(T other) const;
	KaiArray operator -(T other) const;
	KaiArray operator *(T other) const;
	KaiArray operator /(T other) const;

	KaiArray<bool> operator == (const KaiArray other) const;

	KaiArray<bool> operator >(T other) const;
	KaiArray<bool> operator <(T other) const;
	KaiArray<bool> operator == (T other) const;

	KaiArray& operator =(const ArrayCopyInfo<T>& src);

public:  // debugging methods
	void print(string title, bool full = false);
	void print_rows(string title, int64 nfrom, int64 nto, int64 col = 0);
	void print_shape(string title);
	void prod_check() { m_core->m_dimension.prod_check(); }
	void dump_sparse(string title, int64 max_lines = 100);

protected:
	friend class ArrayLoopAxisInfo<T>;
	friend class HostMath;
	friend class Value;

	KaiArray(KaiArrayCore<T>* core) {
		ms_mu_karr_core.lock();
		m_core = core;
		m_core->m_nRefCount++;
		ms_mu_karr_core.unlock();
	}

	KaiArrayCore<T>* m_core;

	void m_copy_list_data(T*& ap, List list, int64 depth, int64 dim, int64* ax_size);

	string m_to_string(Idx& idx, int64 depth, bool full);
	string m_rows_to_string(int64 nfrom, int64 nto, int64 col);

	static T ms_bin_op_add(T a, T b) { return a + b; }
	static T ms_bin_op_sub(T a, T b) { return a - b; }
	static T ms_bin_op_mul(T a, T b) { return a * b; }
	static T ms_bin_op_div(T a, T b) { return a / b; }

	static bool ms_log_bin_op_equal(T a, T b) { return a == b; }
	static bool ms_log_bin_op_greator(T a, T b) { return a > b; }
	static bool ms_log_bin_op_less(T a, T b) { return a < b; }
	static T ms_log_bin_op_or(T a, T b) { return a || b; }

public:
	static T ms_bin_op_sub_reversed(T a, T b) { return b - a; }

protected:
	friend class CudaConn;
	friend class CudaNote;
	friend class CudaMath;

	//KaiArray(Shape shape, T* cuda_p, bool needRegist);
*/
};
