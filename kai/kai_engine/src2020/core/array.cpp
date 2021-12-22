/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "array.h"
#include "host_math.h"
#include "idx.h"
#include "log.h"

#include <stdlib.h>
#include <stdarg.h> 
#include <typeinfo>

mutex_wrap ms_mu_arr_data(true);
mutex_wrap ms_mu_arr_core(true);

Ax _all_(-1);		// -1 means all elements in the axis
Ax Axis::_none_(-999);	// -999 means place holder in generator function

Axis::Axis(Ax ax1, Ax ax2, Ax ax3, Ax ax4, Ax ax5, Ax ax6) {
	m_nDim = 0;
	_regist(ax1);
	_regist(ax2);
	_regist(ax3);
	_regist(ax4);
	_regist(ax5);
	_regist(ax6);
}

Axis::Axis(int64 dim) {
	m_nDim = 0;
	for (int64 n = 0; n < dim; n++) _regist(_all_);
}

template<class T>
void ArrayCopyInfo<T>::operator = (ArrayCopyInfo<T> src){
	kmath->array_copy(m_array, src.m_array, m_axis, src.m_axis);
}

template<class T>
ArrayCopyInfo<T>& ArrayCopyInfo<T>::operator = (const Array<T>& arr) {
	Axis src_all(arr.dim());
	kmath->array_copy(m_array, arr, m_axis, src_all);
	return *this;
}

template<class T>
ArrayCopyInfo<T>& ArrayCopyInfo<T>::operator = (T val) {
	kmath->array_copy(m_array, m_axis, val);
	return *this;
}

template<class T>
ArrayCopyInfo<T>& ArrayCopyInfo<T>::operator += (const Array<T>& arr) {
	Axis src_all(arr.dim());
	kmath->array_add(m_array, arr, m_axis, src_all);
	return *this;
}

template<class T>
ArrayCopyInfo<T>& ArrayCopyInfo<T>::operator -= (const Array<T>& arr) {
	Axis src_all(arr.dim());
	kmath->array_sub(m_array, arr, m_axis, src_all);
	return *this;
}

template<class T>
ArrayLoopAxisInfo<T>& ArrayLoopAxisInfo<T>::operator = (const Array<T>& arr) {
	throw KaiException(KERR_ASSERT);
	return *this;
}

template<class T>
void ArrayLoopAxisInfo<T>::setup(int64 ax_size, int64 pack_size) {
	m_start_offset = 0;
	m_skip_size = pack_size;
	m_loop_count = ax_size;
}

template<class T>
void ArrayLoopAxisInfo<T>::setup(Array<T> array, int64 nth) {
	int64 pack_size = array.dimension().pack_size(nth);

	if (nth < 0) {
		m_start_offset = 0;
		m_skip_size = pack_size;
		m_loop_count = array.m_core->m_dimension.axis_size(nth);
	}
	else {
		m_start_offset = 0;
		m_skip_size = pack_size;
		m_loop_count = array.m_core->m_dimension.axis_size(nth);
	}
}

template<class T>
void ArrayLoopAxisInfo<T>::setup(Array<T> array, Ax ax, int64 nth) {
	int64 pack_size = array.dimension().pack_size(nth);

	if (ax.m_from < 0) {
		m_start_offset = 0;
		m_skip_size = pack_size;
		m_loop_count = array.m_core->m_dimension.axis_size(nth);
	}
	else {
		m_start_offset = ax.m_from * pack_size;
		m_skip_size = ax.m_gap * pack_size;
		int64 sign = (ax.m_gap > 0) ? 1 : -1;
		m_loop_count = (ax.m_to - ax.m_from + ax.m_gap - sign) / ax.m_gap;
	}
}

template<class T>
Array<T>::Array(Shape shape) {
	int64 size = shape.total_size();

	ms_mu_arr_core.lock();
	m_core = new ArrayCore<T>();
	m_core->m_dimension = Dim(shape);
	m_core->m_mdata = new ArrayDataCore<T>(size);
	ms_mu_arr_core.unlock();
}

template<class T>
Array<T>::Array(Shape shape, T* values) {
	int64 size = shape.total_size();

	ms_mu_arr_core.lock();
	m_core = new ArrayCore<T>();
	m_core->m_dimension = Dim(shape);
	m_core->m_mdata = new ArrayDataCore<T>(size);
	ms_mu_arr_core.unlock();
	T* ap = data_ptr();
	memcpy(ap, values, sizeof(T) * size);
}

template<class T>
Array<T>::Array(List list) {
	int64 dim = 0;
	int64 ax_size[KAI_MAX_DIM];

	List head_list = list;

	while (true) {
		if (dim >= KAI_MAX_DIM) throw KaiException(KERR_ASSERT);
		ax_size[dim++] =  head_list.size();
		if (head_list[0].type() == vt::kfloat) break;
		assert (head_list[0].type() == vt::list);
		head_list = head_list[0];
	}
	
	Shape shape(ax_size, dim);

	ms_mu_arr_core.lock();

	m_core = new ArrayCore<T>();
	m_core->m_dimension = Dim(shape);
	m_core->m_mdata = new ArrayDataCore<T>(shape.total_size());

	ms_mu_arr_core.unlock();

	T* ap = data_ptr();

	m_copy_list_data(ap, list, 0, dim, ax_size);
}

template<class T>
void Array<T>::m_copy_list_data(T*& ap, List list, int64 depth, int64 last, int64* ax_size) {
	int64 size =  list.size();
	assert(size == ax_size[depth]);
	if (++depth == last) {
		for (int64 n = 0; n < size; n++) {
			assert(list[n].type() == vt::kfloat);
			*ap++ = list[n];
		}
	}
	else {
#ifdef KAI2021_WINDOWS
		for (int64 n = 0; n < size; n++) {
#else
		for (int64 n = 0; n < size; n++) {
#endif
			assert(list[n].type() == vt::list);
			m_copy_list_data(ap, list[n], depth, last, ax_size);
		}
	}
}

template<class T>
Array<T> Array<T>::data_share_clone() {
	Array<T> clone;

	clone.m_core->m_dimension = m_core->m_dimension.deepcopy();
	clone.m_core->m_mdata = m_core->share_data();

	return clone;
}

template<class T>
Array<T> Array<T>::deepcopy() const {
	if (m_core->m_mdata->m_isCuda) {
		return CudaConn::Copy(m_core->data());

		/*
		T* cuda_p = CudaConn::CopyAsPtr2Ptr(m_core->m_mdata->m_data);

		int64 size = m_core->m_dimension.total_size();

		clone.m_core->m_dimension = m_core->m_dimension.deepcopy();
		clone.m_core->m_mdata = new ArrayDataCore<T>(size, cuda_p);

		return clone;
		*/
	}
	else {
		Array<T> clone;

		int64 size = m_core->m_dimension.total_size();

		clone.m_core->m_dimension = m_core->m_dimension.deepcopy();
		clone.m_core->m_mdata = new ArrayDataCore<T>(size);

		memcpy(clone.m_core->data(), m_core->data(), clone.mem_size());

		return clone;
	}
}

template<class T>
Array<T> Array<T>::to_host() {
	return CudaConn::ToHostArray(*this, "");
}

template<class T>
Array<T>& Array<T>::operator = (const ArrayCopyInfo<T>& src) {
	*this = kmath->copy_array(src.m_array, src.m_axis);
	return *this;
}

template<class T>
void Array<T>::reset() {
	assert(!is_cuda());
	memset(m_core->data(), 0, mem_size());
}

template<class T>
Array<T> Array<T>::zeros(Shape shape) {
	int64 size = shape.total_size();

	Array<T> arr;
	arr.m_core->m_dimension = Dim(shape);
	arr.m_core->m_mdata = new ArrayDataCore<T>(size);
	memset(arr.m_core->data(), 0, arr.mem_size());

	return arr;
}

template<class T>
Array<T> Array<T>::ones(Shape shape, T coef) {
	int64 size = shape.total_size();

	Array<T> arr;
	arr.m_core->m_dimension = Dim(shape);
	arr.m_core->m_mdata = new ArrayDataCore<T>(size);

	T* pdata = arr.m_core->data();

	for (int64 n = 0; n < size; n++) pdata[n] = coef;

	return arr;
}

template<class T>
void Array<T>::get(Array<T> src, Axis axis) {
	*this = kmath->copy_array(src, axis);
}

template<class T>
Array<T> Array<T>::reshape(Shape shape) { return kmath->reshape(*this, shape); }

template<class T>
Array<T> Array<T>::init_grid(Shape shape) { *this = kmath->init_grid(shape); return *this; }

template<class T>
Array<T> Array<T>::tile(int64 num) { return kmath->tile(*this, num); }

template<class T>
Array<T> Array<T>::untile(int64 num) { return kmath->untile(*this, num); }

template<class T>
Array<T> Array<T>::flatten() { return kmath->flatten(*this); }

template<class T>
Array<T> Array<T>::transpose() { return kmath->transpose(*this); }

template<class T>
Array<T> Array<T>::transpose(Idx idx) { return kmath->transpose(*this, idx); }

template<class T>
Array<T> Array<T>::maxarg(int64 axis, Array<int64>& arg) { return kmath->maxarg(*this, axis, arg); }

template<class T>
void Array<T>::setarg(Array<int64>& arg, Array<T> values) { kmath->setarg(*this, arg, values); }

template<class T>
Array<T> Array<T>::sum(int64 axis) { return kmath->sum(*this, axis); }

template<class T>
Array<T> Array<T>::avg(int64 axis) { return kmath->avg(*this, axis); }

template<class T>
Array<T> Array<T>::var(int64 axis, Array<T>* pavg) { return kmath->var(*this, axis, pavg); }

template<class T>
Array<T> Array<T>::square() { return kmath->square(*this); }

template<class T>
Array<T> Array<T>::abs() { return kmath->abs(*this); }

template<class T>
Array<int64> Array<T>::round() { return kmath->round(*this); }

template<class T>
Array<int64> Array<T>::binary_row_to_int() { return kmath->binary_row_to_int(*this); }

template<class T>
Array<T> Array<T>::fetch_rows(vector<int64> rows) { return kmath->fetch_rows(*this, rows); }

template<class T>
Array<T> Array<T>::dotsum(Array<T> other) { return kmath->dotsum(*this, other); } 

//template<class T>
//Array<T> Array<T>::sort(sortdir dir, Array<int64>& sort_idx) { return kmath->sort(*this, dir, sort_idx); }

template<class T>
Array<float> Array<T>::to_float() { return kmath->to_float(*this); }

template<class T>
Array<T> Array<T>::merge_time_axis() {
	return reshape(shape().merge_time_axis());
}

template<class T>
void Array<T>::split_axis_size(int64 axis, int64* psize1, int64* psize2, int64* psize3) {
	axis = ax_mod(axis, dim());
	*psize1 = *psize2 = *psize3 = 1;
	for (int64 n = 0; n < dim(); n++) {
		if (n < axis) *psize1 *= axis_size(n);
		else if (n > axis) *psize3 *= axis_size(n);
		else if (n == axis) *psize2 *= axis_size(n);
	}
}

template<class T>
Array<T> Array<T>::split_time_axis(int64 mb_size) {
	return reshape(shape().split_time_axis(mb_size));
}

template<class T>
Array<T> Array<T>::add_axis(int64 pos) {
	return reshape(shape().add_nth(pos, 1));
}

template<class T>
T Array<T>::sum() { return kmath->sum(*this); }

template<class T>
int64 Array<T>::true_count() { return kmath->true_count(*this); }

template<class T>
int64 Array<T>::count_in_range(T min_val, T max_val) { return kmath->count_in_range(*this, min_val, max_val); }

/*
template<class T>
Array<T> Array<T>::flatten() {
	Array<T> clone;

	int64 size = m_core->m_dimension.total_size();
	Dim dim_1d(size);

	clone.m_core->m_dimension = dim_1d;
	clone.m_core->m_mdata = m_core->share_data();

	return clone;
}
*/

template<class T>
void Array<T>::copy(Idx dest_idx, Array<T> src, Idx src_idx) {
	abort();
}

template<class T>
void Array<T>::copy(Array<T> src) {
	assert(total_size() == src.total_size());
	T* dp = data_ptr();
	T* sp = src.data_ptr();

	memcpy(dp, sp, sizeof(T) * total_size());
}

#ifdef KAI2021_WINDOWS
#else
template<class T>
ArrayLoopAxisInfo<T> Array<T>::operator [](int64 nth) {
	throw KaiException(KERR_ASSERT);
	ArrayLoopAxisInfo<T> a;
	return a;
}
#endif

template<class T>
T& Array<T>::operator [](Idx idx) {
	int64 offset = m_core->m_dimension.get_offset(idx);
	return m_core->data()[offset];
}

template<class T>
T Array<T>::operator [](Idx idx) const {
	int64 offset = m_core->m_dimension.get_offset(idx);
	return m_core->data()[offset];
}

template<class T>
Array<T> Array<T>::operator +=(const Array<T> other) {
	kmath->bin_op(*this, other, ms_bin_op_add);
	return *this;
}

template<class T>
Array<T> Array<T>::operator -=(const Array<T> other) {
	kmath->bin_op(*this, other, ms_bin_op_sub);
	return *this;
}

template<class T>
Array<T> Array<T>::operator *=(const Array<T> other) {
	kmath->bin_op(*this, other, ms_bin_op_mul);
	return *this;
}

template<class T>
Array<T> Array<T>::operator /=(const Array<T> other) {
	kmath->bin_op(*this, other, ms_bin_op_div);
	return *this;
}

template<class T>
Array<T> Array<T>::operator +=(T other) {
	kmath->bin_op(*this, other, ms_bin_op_add);
	return *this;
}

template<class T>
Array<T> Array<T>::operator -=(T other) {
	kmath->bin_op(*this, other, ms_bin_op_sub);
	return *this;
}

template<class T>
Array<T> Array<T>::operator *=(T other) {
	kmath->bin_op(*this, other, ms_bin_op_mul);
	return *this;
}

template<class T>
Array<T> Array<T>::operator /=(T other) {
	kmath->bin_op(*this, other, ms_bin_op_div);
	return *this;
}

template<class T>
Array<T> Array<T>::operator +(const Array other) const {
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_add);
	return clone;
}

template<class T>
Array<T> Array<T>::operator -(const Array other) const {
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_sub);
	return clone;
}

template<class T>
Array<T> Array<T>::operator *(const Array other) const {
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_mul);
	return clone;
}

template<class T>
Array<T> Array<T>::operator /(const Array other) const {
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_div);
	return clone;
}

template<class T>
Array<T> Array<T>::logical_or(Array<T> other) {
	Shape shape = this->shape();
	Array<T> result(shape);
	kmath->log_bin_op(result, *this, other, ms_log_bin_op_or);
	return result;
}

template<class T>
Array<T> Array<T>::extract_selected(Array<bool> selector) {
	return kmath->extract_selected(*this, selector);
}

template<class T>
Array<T> Array<T>::fill_selected(Array<T> other, Array<bool> selector) {
	return kmath->fill_selected(*this, other, selector);
}

template<class T>
Array<T> Array<T>::wvec_select(Array<int64> other) {
	return kmath->wvec_select(*this, other);
}

template<class T>
Array<T> Array<T>::wvec_select_idx(Array<int64> other, int64 dic_count, int64* voc_counts) {
	return kmath->wvec_select_idx(*this, other, dic_count, voc_counts);
}

template<class T>
Array<T> Array<T>::select_col(Array<int64> other) {
	return kmath->select_col(*this, other);
}

template<class T>
Array<T> Array<T>::minus_1_on_idx(Array<int64> other) {
	return kmath->minus_1_on_idx(*this, other);
}

template<class T>
Array<T> Array<T>::minus_1_on_1st() {
	return kmath->minus_1_on_1st(*this);
}

template<class T>
Array<T> Array<T>::mult_dim(int64 axis) {
	return kmath->mult_dim(*this, axis);
}

template<class T>
Array<T> Array<T>::filter(int64 axis, int64 index) {
	return kmath->filter(*this, axis, index);
}

template<class T>
Array<T> Array<T>::unfilter(Array<T> filtered, int64 axis, int64 index) {
	return kmath->unfilter(*this, filtered, axis, index);
}

template<class T>
vector<Array<T>> Array<T>::hsplit() {
	return kmath->hsplit_last(*this);
}

template<class T>
Array<T> Array<T>::expand_images(Shape stride) {
	return kmath->expand_images(*this, stride);
}

template<class T>
Array<T> Array<T>::expand_undo_images(Shape stride) {
	return kmath->expand_undo_images(*this, stride);
}

template<class T>
Array<T> Array<T>::get_row(int64 nth) {
	return kmath->get_row(*this, nth);
}

template<class T>
Array<T> Array<T>::get_col(int64 nth) {
	return kmath->get_col(*this, nth);
}

template<class T>
void Array<T>::set_row(int64 nth, Array<T> other) {
	kmath->set_row(*this, nth, other);
}

template<class T>
void Array<T>::set_row(int64 nth, T other) {
	kmath->set_row(*this, nth, other);
}

template<class T>
void Array<T>::sub_row(int64 nth, Array<T> other) {
	kmath->sub_row(*this, nth, other);
}

template<class T>
Array<int> Array<T>::to_int() {
	return kmath->to_int(*this);
}

template<class T>
Array<int64> Array<T>::to_int64() {
	return kmath->to_int64(*this);
}

template<class T>
Array<T> Array<T>::operator +(T other) const {
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_add);
	return clone;
}

template<class T>
Array<T> Array<T>::operator -(T other) const {
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_sub);
	return clone;
}

template<class T>
Array<T> Array<T>::operator *(T other) const {
	return kmath->mult_scalar(*this, other);
	/*
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_mul);
	return clone;
	*/
}

template<class T>
Array<T> Array<T>::operator /(T other) const {
	Array<T> clone = deepcopy();
	kmath->bin_op(clone, other, ms_bin_op_div);
	return clone;
}

template<class T>
Array<bool> Array<T>::operator == (const Array other) const {
	Shape shape = this->shape();
	Array<bool> result(shape);
	kmath->log_bin_op(result, *this, other, ms_log_bin_op_equal);
	return result;
}

template<class T>
Array<bool> Array<T>::operator >(T other) const {
	Array<bool> result(shape());
	kmath->log_bin_op(result, *this, other, ms_log_bin_op_greator);
	return result;
}

template<class T>
Array<bool> Array<T>::operator <(T other) const {
	Array<bool> result(shape());
	kmath->log_bin_op(result, *this, other, ms_log_bin_op_less);
	return result;
}

template<class T>
Array<bool> Array<T>::operator ==(T other) const {
	Array<bool> result(shape());
	kmath->log_bin_op(result, *this, other, ms_log_bin_op_equal);
	return result;
}

template <class T>
Array<T> operator -(T a, Array<T>& arr) {
	Array<T> clone = arr.deepcopy();
	kmath->bin_op(clone, a, Array<T>::ms_bin_op_sub_reversed);
	return clone;
}

template<class T>
void Array<T>::print_shape(string title) {
	if (!KArgs::trace_print) return;
	if (is_cuda()) {
		CudaConn::DumpShape(data_ptr(), title);
	}
	else {
		logger.Print("[%s] %s host", title.c_str(), shape().desc().c_str());
	}
}

template<class T>
void Array<T>::dump_sparse(string title, int64 max_lines) {
	CudaConn cuda("dump_sparse", NULL);

	T* p_data = data_ptr();
	int64 size = total_size();
	
	if (is_cuda()) p_data = cuda.get_host_data(*this);

	logger.Print("[%s] %s %s", title.c_str(), shape().desc().c_str(), is_cuda() ? "cuda" : "host");

	int64 n, lines = 0;

	for (n = 0; n < size; n++) {
		if (p_data[n] == 0) continue;
		if (lines++ >= max_lines) continue;
		logger.Print("    [%lld] = %8.6f", n, (float)p_data[n]);
	}

	if (is_cuda()) logger.Print("    ...");
	logger.Print("    %lld nonzero lines", lines);
}

template<class T>
void Array<T>::print(string title, bool full) {
	if (!KArgs::trace_print) return;
	if (is_cuda()) {
		CudaConn::DumpArr(data_ptr(), title, shape(), full);
	}
	else {
		logger.Print("[%s] %s host", title.c_str(), shape().desc().c_str());
		Idx idx;
		idx.set_size(dim());
		string contents = m_to_string(idx, 0, full);
		logger.Print("    %s", contents.c_str());
		//throw KaiException(KERR_UNIMPEMENTED_YET);
	}
}

template<class T>
void Array<T>::print_rows(string title, int64 nfrom, int64 nto, int64 col) {
	if (!KArgs::trace_print) return;
	if (is_cuda()) {
		CudaConn::Print_rows(data_ptr(), title, nfrom, nto, col);
	}
	else {
		logger.Print("[%s:%d-%d] %s host", title.c_str(), nfrom, nto, shape().desc().c_str());
		string contents = m_rows_to_string(nfrom, nto, col);
		logger.Print("    %s", contents.c_str());
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}
}

template<class T>
string Array<T>::m_to_string(Idx& idx, int64 depth, bool full) {
	string buffer;
	string infix = ",";
	string delimeter = "[";

	int64 axis_size = dimension().axis_size(depth);

	if (depth < dim() - 1) {
		infix = ",\n     ";
		for (int64 n = 0; n < depth; n++) infix += " ";
	}

	for (int64 n = 0; n < axis_size; n++) {
		idx[depth] = n;
		if (!full && axis_size >= 10 && n >= 3 && n <= axis_size - 3) {
			if (n == 3) buffer += delimeter + "...";
			continue;
		}
		if (depth == dim() - 1) {
			string type_name = typeid(T).name();
			if (type_name == "f" || type_name == "float") {
				char fbuf[1024];
				float* farr = (float*) data_ptr();
				if (farr) { 
					int64 offset = m_core->m_dimension.get_offset(idx);
#ifdef KAI2021_WINDOWS
					sprintf_s<1024>(fbuf, "%16.9e", (float)farr[offset]);
#else
					sprintf(fbuf, "%16.9e", farr[offset]);
#endif
				}
				else {
#ifdef KAI2021_WINDOWS
					mysprintf(fbuf, "null");
#else
					sprintf(fbuf, "null");
#endif
				}
				buffer += delimeter + (string)fbuf; // +"(" + idx.desc() + ")";
			}
			else {
				buffer += delimeter + to_string((*this)[idx]); // +"(" + idx.desc() + ")";
			}
		}
		else {
			buffer += delimeter + m_to_string(idx, depth + 1, full);
		}
		delimeter = infix;
	}

	return buffer + "]";
}

template<class T>
string Array<T>::m_rows_to_string(int64 nfrom, int64 nto, int64 col) {
	if (col <= 0) col = total_size() / axis_size(0);

	assert(total_size() % col == 0);

	int64 nrows = total_size() / col;

	if (nto < 0) nto = nrows;

	string buffer;

	T* pdata = data_ptr();

	for (int64 n = nfrom; n < nto; n++) {
		buffer += (n == 0) ? "[" : "]\n     ";

		for (int64 m = 0; m < col; m++) {
			buffer += (m == 0) ? "[" : ",";

			int64 nth = n * col + m;

			string type_name = typeid(T).name();
			if (type_name == "f" || type_name == "float") {
				char fbuf[1024];
#ifdef KAI2021_WINDOWS
				sprintf_s<1024>(fbuf, "%16.9e", (float)pdata[nth]);
				//mysprintf(fbuf, "%16.9e", (float)pdata[nth]);
#else
				sprintf(fbuf, "%16.9e", (float)pdata[nth]);
#endif
				buffer += (string)fbuf; // +"(" + idx.desc() + ")";
			}
			else {
				buffer += to_string(pdata[nth]); // +"(" + idx.desc() + ")";
			}
		}
	}

	return buffer + "]]";
}

template void ArrayCopyInfo<float>::operator = (ArrayCopyInfo<float> src);

template ArrayCopyInfo<float>& ArrayCopyInfo<float>::operator = (const Array<float>& arr);
template ArrayCopyInfo<float>& ArrayCopyInfo<float>::operator = (float val);
template ArrayCopyInfo<float>& ArrayCopyInfo<float>::operator += (const Array<float>& arr);
template ArrayCopyInfo<float>& ArrayCopyInfo<float>::operator -= (const Array<float>& arr);

template ArrayLoopAxisInfo<float>& ArrayLoopAxisInfo<float>::operator = (const Array<float>& src);

template void ArrayLoopAxisInfo<float>::setup(int64 ax_size, int64 pack_size);
template void ArrayLoopAxisInfo<float>::setup(Array<float> array, int64 nth);
template void ArrayLoopAxisInfo<float>::setup(Array<float> array, Ax ax, int64 nth);

template void ArrayLoopAxisInfo<int64>::setup(int64 ax_size, int64 pack_size);
template void ArrayLoopAxisInfo<int64>::setup(Array<int64> array, Ax ax, int64 nth);

template void ArrayLoopAxisInfo<unsigned char>::setup(int64 ax_size, int64 pack_size);
template void ArrayLoopAxisInfo<unsigned char>::setup(Array<unsigned char> array, Ax ax, int64 nth);

template Array<float>::Array(Shape shape);
template Array<bool>::Array(Shape shape);
template Array<short>::Array(Shape shape);
template Array<int>::Array(Shape shape);
template Array<int64>::Array(Shape shape);
template Array<unsigned char>::Array(Shape shape);

template Array<int64>::Array(Shape shape, int64* values);

template Array<float>::Array(List list);
template void Array<float>::m_copy_list_data(float*& ap, List list, int64 depth, int64 last, int64* ax_size);

template void Array<unsigned char>::get(Array<unsigned char> src, Axis axis);
template void Array<float>::get(Array<float> src, Axis axis);

template void Array<float>::copy(Array<float> src);
template void Array<int>::copy(Array<int> src);
template void Array<int64>::copy(Array<int64> src);
template void Array<bool>::copy(Array<bool> src);

template Array<float> Array<float>::data_share_clone();
template Array<int64> Array<int64>::data_share_clone();
template Array<unsigned char> Array<unsigned char>::data_share_clone();

template Array<float> Array<float>::deepcopy() const;
template Array<int> Array<int>::deepcopy() const;
template Array<int64> Array<int64>::deepcopy() const;
template Array<bool> Array<bool>::deepcopy() const;
template Array<unsigned char> Array<unsigned char>::deepcopy() const;

template Array<float> Array<float>::to_host();
template Array<int> Array<int>::to_host();
template Array<int64> Array<int64>::to_host();
template Array<bool> Array<bool>::to_host();

template Array<float>& Array<float>::operator = (const ArrayCopyInfo<float>& src);
template Array<int64>& Array<int64>::operator = (const ArrayCopyInfo<int64>& src);

template Array<float> Array<float>::zeros(Shape shape);
template Array<int64> Array<int64>::zeros(Shape shape);
template Array<float> Array<float>::ones(Shape shape, float coef);
template Array<int64> Array<int64>::ones(Shape shape, int64 coef);

template Array<float> Array<float>::init_grid(Shape shape);

template void Array<bool>::reset();

template Array<float> Array<float>::reshape(Shape shape);
template Array<int64> Array<int64>::reshape(Shape shape);
template Array<unsigned char> Array<unsigned char>::reshape(Shape shape);
template Array<float> Array<float>::tile(int64 num);
template Array<float> Array<float>::untile(int64 num);
template Array<float> Array<float>::flatten();
template Array<int64> Array<int64>::flatten();
template Array<float> Array<float>::transpose();
template Array<float> Array<float>::transpose(Idx idx);
template Array<unsigned char> Array<unsigned char>::transpose(Idx idx);
template Array<float> Array<float>::maxarg(int64 axis, Array<int64>& idx);
template Array<float> Array<float>::sum(int64 axis);
template Array<float> Array<float>::avg(int64 axis);
template Array<float> Array<float>::var(int64 axis, Array<float>* pavg);
template Array<float> Array<float>::square();
template Array<float> Array<float>::abs();
template Array<int64> Array<float>::round();
template Array<int64> Array<int64>::binary_row_to_int();
template Array<float> Array<float>::fetch_rows(vector<int64> rows);
template Array<float> Array<float>::dotsum(Array<float> other);
//template Array<float> Array<float>::sort(sortdir dir, Array<int64>& sort_idx);

template Array<unsigned char> Array<unsigned char>::extract_selected(Array<bool> selector);
template Array<int64> Array<int64>::extract_selected(Array<bool> selector);
template Array<float> Array<float>::extract_selected(Array<bool> selector);

template Array<unsigned char> Array<unsigned char>::fill_selected(Array<unsigned char> other, Array<bool> selector);
template Array<int64> Array<int64>::fill_selected(Array<int64> other, Array<bool> selector);
template Array<float> Array<float>::fill_selected(Array<float> other, Array<bool> selector);

template Array<float> Array<unsigned char>::to_float();
template Array<float> Array<int64>::to_float();
template Array<float> Array<bool>::to_float();
template int64 Array<float>::count_in_range(float min_val, float max_val);
template Array<float> Array<float>::wvec_select(Array<int64> other);
template Array<float> Array<float>::wvec_select_idx(Array<int64> other, int64 dic_count, int64* voc_counts);
template Array<float> Array<float>::select_col(Array<int64> other);
template Array<float> Array<float>::minus_1_on_idx(Array<int64> other);
template Array<float> Array<float>::minus_1_on_1st();
template Array<int64> Array<float>::to_int64();

template Array<float> Array<float>::mult_dim(int64 axis);

template Array<int64> Array<int64>::filter(int64 axis, int64 index);
template Array<float> Array<float>::filter(int64 axis, int64 index);

template Array<int64> Array<int64>::unfilter(Array<int64> filtered, int64 axis, int64 index);
template Array<float> Array<float>::unfilter(Array<float> filtered, int64 axis, int64 index);

template vector<Array<int64>> Array<int64>::hsplit();
template vector<Array<float>> Array<float>::hsplit();

template Array<float> Array<float>::expand_images(Shape stride);
template Array<float> Array<float>::expand_undo_images(Shape stride);

template Array<float> Array<float>::get_row(int64 nth);

template Array<int64> Array<int64>::get_col(int64 nth);

template void Array<float>::set_row(int64 nth, Array<float> other);
template void Array<float>::set_row(int64 nth, float other);
template void Array<float>::sub_row(int64 nth, Array<float> other);

template float Array<float>::sum();
template int64 Array<int64>::sum();

template int64 Array<bool>::true_count();

template Array<float> Array<float>::add_axis(int64 pos);

template void Array<float>::split_axis_size(int64 axis, int64* psize1, int64* psize2, int64* psize3);

template Array<float> Array<float>::merge_time_axis();
template Array<float> Array<float>::split_time_axis(int64 mb_size);

template Array<int64> Array<int64>::merge_time_axis();
template Array<int64> Array<int64>::split_time_axis(int64 mb_size);

template void Array<float>::setarg(Array<int64>& arg, Array<float> values);

template void Array<float>::print_shape(string title);
template void Array<unsigned char>::print_shape(string title);

template void Array<float>::dump_sparse(string title, int64 max_lines);

template void Array<float>::print(string title, bool full);
template void Array<int64>::print(string title, bool full);
template void Array<bool>::print(string title, bool full);
template void Array<unsigned char>::print(string title, bool full);

template void Array<float>::print_rows(string title, int64 nfrom, int64 nto, int64 col);
template void Array<int64>::print_rows(string title, int64 nfrom, int64 nto, int64 col);
template void Array<bool>::print_rows(string title, int64 nfrom, int64 nto, int64 col);
template void Array<unsigned char>::print_rows(string title, int64 nfrom, int64 nto, int64 col);

template string Array<float>::m_to_string(Idx& idx, int64 depth, bool full);
template string Array<int>::m_to_string(Idx& idx, int64 depth, bool full);
template string Array<int64>::m_to_string(Idx& idx, int64 depth, bool full);
template string Array<bool>::m_to_string(Idx& idx, int64 depth, bool full);
template string Array<unsigned char>::m_to_string(Idx& idx, int64 depth, bool full);

template string Array<float>::m_rows_to_string(int64 nfrom, int64 nto, int64 col);
template string Array<int64>::m_rows_to_string(int64 nfrom, int64 nto, int64 col);
template string Array<bool>::m_rows_to_string(int64 nfrom, int64 nto, int64 col);
template string Array<unsigned char>::m_rows_to_string(int64 nfrom, int64 nto, int64 col);

#ifdef KAI2021_WINDOWS
#else
template ArrayLoopAxisInfo<float> Array<float>::operator [](int64 nth);
#endif

template unsigned char& Array<unsigned char>::operator [](Idx idx);
template unsigned char Array<unsigned char>::operator [](Idx idx) const;

template int& Array<int>::operator [](Idx idx);
template int Array<int>::operator [](Idx idx) const;

template int64& Array<int64>::operator [](Idx idx);
template int64 Array<int64>::operator [](Idx idx) const;

template float& Array<float>::operator [](Idx idx);
template float Array<float>::operator [](Idx idx) const;

template bool& Array<bool>::operator [](Idx idx);
template bool Array<bool>::operator [](Idx idx) const;

template Array<float> Array<float>::operator +=(const Array<float> other);
template Array<float> Array<float>::operator -=(const Array<float> other);
template Array<float> Array<float>::operator *=(const Array<float> other);
template Array<float> Array<float>::operator /=(const Array<float> other);

template Array<float> Array<float>::operator +=(float other);
template Array<float> Array<float>::operator -=(float other);
template Array<float> Array<float>::operator *=(float other);
template Array<float> Array<float>::operator /=(float other);

template Array<int64> Array<int64>::operator +=(int64 other);

template Array<float> Array<float>::operator +(const Array<float> other) const;
template Array<float> Array<float>::operator -(const Array<float> other) const;
template Array<float> Array<float>::operator *(const Array<float> other) const;

template Array<float> Array<float>::operator /(const Array<float> other) const;

template Array<bool> Array<bool>::logical_or(const Array<bool> other);

template Array<float> Array<float>::operator +(float other) const;
template Array<float> Array<float>::operator -(float other) const;
template Array<float> Array<float>::operator *(float other) const;
template Array<float> Array<float>::operator /(float other) const;

template Array<bool> Array<bool>::operator == (const Array<bool> other) const;
template Array<bool> Array<int64>::operator == (const Array<int64> other) const;

template Array<bool> Array<float>::operator >(float other) const;
template Array<bool> Array<float>::operator <(float other) const;
template Array<bool> Array<int64>::operator >(int64 other) const;
template Array<bool> Array<int64>::operator ==(int64 other) const;
template Array<bool> Array<unsigned char>::operator ==(unsigned char other) const;

template Array<float> operator -(float a, Array<float>& arr);
