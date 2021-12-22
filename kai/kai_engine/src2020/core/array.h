/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"
#include "dim.h"
#include "value.h"
#include "../cuda/cuda_conn.cuh"

#include <stdio.h>

class Ax { // ArrayAxisAccess
public:
	Ax() { m_from = -1; m_to = 0; m_gap = 0; }
	Ax(int64 n) { m_from = n; m_to = n + 1; m_gap = 1; }
	Ax(int64 from, int64 to, int64 gap = 1) { m_from = from; m_to = to; m_gap = gap; }

public:
	int64 m_from;
	int64 m_to;
	int64 m_gap;
};

extern Ax _all_;

class Axis {
public:
	Axis() { m_nDim = 0;}
	Axis(Ax ax1, Ax ax2 = _none_, Ax ax3 = _none_, Ax ax4 = _none_, Ax ax5 = _none_, Ax ax6 = _none_);
	Axis(int64 dim);

	Ax operator [](int64 nth) const { return m_axises[nth];  }

private:
	void _regist(Ax ax) { if (ax.m_from != -999) m_axises[m_nDim++] = ax; }

public:
	int64 m_nDim;
	Ax m_axises[KAI_MAX_DIM];

protected:
	static Ax _none_;
};

template <class T>
class ArrayLoopAxisInfo {
public:
	void setup(int64 ax_size, int64 pack_size);
	void setup(Array<T> array, int64 nth);
	void setup(Array<T> array, Ax ax, int64 nth);

	ArrayLoopAxisInfo& operator = (const Array<T>& arr);
public:
	int64 m_start_offset;
	int64 m_skip_size;
	int64 m_loop_count;
	bool m_cont;
};

template <class T> class ArrayCore;

extern mutex_wrap ms_mu_arr_data;
extern mutex_wrap ms_mu_arr_core;

template <class T>
class ArrayDataCore {
public:
	ArrayDataCore(int64 size) {
		ms_mu_arr_data.lock();
		m_nRefCount = 1;
		m_size = size * sizeof(T);
		m_data = (T*)malloc(m_size);
		m_isCuda = false;
		ms_mu_arr_data.unlock();
	}

	ArrayDataCore(int64 size, T* cuda_p) {
		ms_mu_arr_data.lock();
		m_nRefCount = 1;
		m_size = size * sizeof(T);
		m_data = cuda_p;
		m_isCuda = true;
		ms_mu_arr_data.unlock();
	}

	virtual ~ArrayDataCore() {
		if (m_isCuda) CudaConn::Free(m_data, this);
		else free(m_data);
	}

	void destroy() {
		ms_mu_arr_data.lock();
		bool del = (this && --m_nRefCount <= 0);
		ms_mu_arr_data.unlock();
		if (del) delete this;
	}

protected:
	friend class ArrayCore<T>;
	friend class Array<T>;
	friend class CudaConn;
	friend class CudaNote;

	int64 m_nRefCount;
	int64 m_size;
	T* m_data;
	bool m_isCuda;
};

template <class T>
class ArrayCore {
public:
	ArrayCore() : m_nRefCount(1), m_dimension(), m_mdata(NULL) {
	}

	virtual ~ArrayCore() { m_mdata->destroy(); }
	void destroy() {
		ms_mu_arr_core.lock();
		bool del = (this && --m_nRefCount <= 0);
		ms_mu_arr_core.unlock();
		if (del) delete this;
	}

protected:
	T* data() { return m_mdata->m_data; }
	int64 data_size() { return m_mdata->m_size; }
	ArrayDataCore<T>* share_data() {
		ms_mu_arr_data.lock();
		m_mdata->m_nRefCount++;
		ms_mu_arr_data.unlock();
		return m_mdata;
	}

protected:
	friend class Array<T>;
	friend class ArrayLoopAxisInfo<T>;
	friend class HostMath;
	friend class Value;
	friend class CudaConn;
	friend class CudaNote;

	int m_nRefCount;
	Dim m_dimension;
	ArrayDataCore<T>* m_mdata;

};

template <class T>
class ArrayCopyInfo {
public:
	ArrayCopyInfo(Array<T>& array, Axis axis) : m_array(array) { m_axis = axis; }
	ArrayCopyInfo& operator = (const Array<T>& arr);
	ArrayCopyInfo& operator = (T val);
	ArrayCopyInfo& operator += (const Array<T>& arr);
	ArrayCopyInfo& operator -= (const Array<T>& arr);
	void operator = (ArrayCopyInfo src);

public:
	Array<T>& m_array;
	Axis m_axis;
};

template <class T>
class Array {
public:
	Array() {
		ms_mu_arr_core.lock();
		m_core = new ArrayCore<T>();
		ms_mu_arr_core.unlock();
	}
	Array(const Array& src) {
		ms_mu_arr_core.lock();
		m_core = src.m_core;
		m_core->m_nRefCount++;
		ms_mu_arr_core.unlock();
	}
	Array& operator = (const Array& src) {
		if (this == &src) return *this;
		m_core->destroy();
		ms_mu_arr_core.lock();
		m_core = src.m_core;
		m_core->m_nRefCount++;
		ms_mu_arr_core.unlock();
		return *this;
	}

	virtual ~Array() {
		m_core->destroy();
		m_core = NULL;
	}

	Array(Shape shape);
	Array(Shape shape, T* values);
	Array(List list);

	Array data_share_clone();

	static Array zeros(Shape shape);
	static Array ones(Shape shape, T coef=1);

	void reset(); // set to zero(false) all array elements

	Array transpose();
	Array transpose(Idx idx);
	Array reshape(Shape shape);
	Array init_grid(Shape shape);
	Array tile(int64 num);
	Array untile(int64 num);
	Array flatten();
	Array maxarg(int64 axis, Array<int64>& arg);
	Array sum(int64 axis);
	Array avg(int64 axis);
	Array var(int64 axis, Array* pavg = NULL);
	Array square();
	Array abs();
	Array<int64> round();
	Array<int64> binary_row_to_int();
	Array dotsum(Array other);
	//Array sort(sortdir dir = sortdir::sd_asc, Array<int64>& sort_idx = NULL);
	//Array sort(sortdir dir, Array<int64>& sort_idx);
	Array<float> to_float();
	Array logical_or(Array other);
	Array extract_selected(Array<bool> selector);
	Array fill_selected(Array other, Array<bool> selector);
	Array fetch_rows(vector<int64> rows);
	Array wvec_select(Array<int64> other);
	Array wvec_select_idx(Array<int64> other, int64 dic_count, int64* voc_counts);
	Array select_col(Array<int64> selector);
	Array minus_1_on_idx(Array<int64> selector);
	Array minus_1_on_1st();
	Array get_row(int64 nth);
	Array get_col(int64 nth);
	Array filter(int64 axis, int64 index);
	Array mult_dim(int64 axis=-1);
	Array unfilter(Array filtered, int64 axis, int64 index);
	vector<Array> hsplit();

	Array expand_images(Shape stride);
	Array expand_undo_images(Shape stride);


	void set_row(int64 nth, Array other);
	void set_row(int64 nth, T other);
	void sub_row(int64 nth, Array other);

	Array<int> to_int();
	Array<int64> to_int64();

	int64 true_count();

	void split_axis_size(int64 axis, int64* psize1, int64* psize2, int64* psize3);

	Array merge_time_axis();
	Array split_time_axis(int64 mb_size);

	Array add_axis(int64 pos);

	int64 count_in_range(T min_val, T max_val);
	
	T sum();

	void setarg(Array<int64>& arg, Array values);

	bool is_empty() const { return dim() == 0; }

	Array deepcopy() const;

	Array to_host();

	Dim dimension() { return m_core->m_dimension; }

	Shape shape() const { return Shape(m_core->m_dimension); }

	int64 dim() const { return m_core->m_dimension.dim(); }
	int64 axis_size(int64 axis) const { return m_core->m_dimension[axis]; }
	int64 total_size() const { return m_core->m_dimension.total_size(); }

	bool is_empty() { return m_core->m_mdata == NULL; }
	bool is_cuda() { return m_core->m_mdata->m_isCuda; }

	void copy(Idx dest_idx, Array src, Idx src_idx);
	void copy(Array src);

	ArrayCopyInfo<T> operator [](Axis axis) { return ArrayCopyInfo<T>(*this, axis); }
	ArrayCopyInfo<T> operator [](Axis axis) const { return ArrayCopyInfo<T>((Array&)*this, axis); }

	void get(Array src, Axis axis);

	static int64 element_size() { return  sizeof(T); }
	
	T* data_ptr() { return m_core->data(); }

	int64 mem_size() { return m_core->m_mdata ? m_core->m_mdata->m_size : 0; }

	ArrayLoopAxisInfo<T> operator [](int64 nth);

	T& operator [](Idx idx);
	T operator [](Idx idx) const;

	Array operator +=(const Array other);
	Array operator -=(const Array other);
	Array operator *=(const Array other);
	Array operator /=(const Array other);

	Array operator +=(T other);
	Array operator -=(T other);
	Array operator *=(T other);
	Array operator /=(T other);

	Array operator +(const Array other) const;
	Array operator -(const Array other) const;
	Array operator *(const Array other) const;
	Array operator /(const Array other) const;

	Array operator +(T other) const;
	Array operator -(T other) const;
	Array operator *(T other) const;
	Array operator /(T other) const;

	Array<bool> operator == (const Array other) const;

	Array<bool> operator >(T other) const;
	Array<bool> operator <(T other) const;
	Array<bool> operator == (T other) const;

	Array& operator =(const ArrayCopyInfo<T>& src);

public:  // debugging methods
	void print(string title, bool full = false);
	void print_rows(string title, int64 nfrom, int64 nto, int64 col=0);
	void print_shape(string title);
	void prod_check() { m_core->m_dimension.prod_check(); }
	void dump_sparse(string title, int64 max_lines=100);

protected:
	friend class ArrayLoopAxisInfo<T>;
	friend class HostMath;
	friend class Value;

	Array(ArrayCore<T>* core) {
		ms_mu_arr_core.lock();
		m_core = core;
		m_core->m_nRefCount++;
		ms_mu_arr_core.unlock();
	}

	ArrayCore<T>* m_core;

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

	//Array(Shape shape, T* cuda_p, bool needRegist);
};

template <class T>
Array<T> operator -(T a, Array<T>& arr);