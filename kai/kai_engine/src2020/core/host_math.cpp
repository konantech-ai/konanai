/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "host_math.h"
#include "value.h"
#include "array.h"
#include "random.h"
#include "util.h"
#include "func_timer.h"
#include "log.h"

#include <float.h>
#include <stdio.h>
#ifdef KAI2021_WINDOWS
#else
#include <unistd.h>
#endif

#include <chrono>
#include <thread>

using namespace std;

HostMath hmath;

Array<float> HostMath::random_uniform(Shape shape, CudaConn* pConn) {
	return Random::uniform(shape);
}

Array<float> HostMath::random_bernoulli(Shape shape, float prob, CudaConn* pConn) {
	return Random::bernoulli(shape, prob);
}

Array<float> HostMath::random_normal(float mean, float std, Shape shape, CudaConn* pConn) {
	return Random::normal(mean, std, shape);
}

Array<float> HostMath::zeros(Shape shape, CudaConn* pConn) {
	return Array<float>::zeros(shape);
}

Array<float> HostMath::ones(Shape shape, float coef, CudaConn* pConn) {
	return Array<float>::ones(shape, coef);
}

Array<float> HostMath::onehot(Array<int64> idxs, int64 nom_cnt) {
	int64 dat_cnt = idxs.total_size();
	Array<float> arr = Array<float>::zeros(Shape(dat_cnt, nom_cnt));
	float* ap = arr.data_ptr();
	int64* xp = idxs.data_ptr();
	for (int64 n = 0; n < dat_cnt; n++) {
		ap[*xp++] = 1.0f;
		ap += nom_cnt;
	}
	return arr;
}

Array<int64> HostMath::zeros_int(Shape shape, CudaConn* pConn) {
	return Array<int64>::zeros(shape);
}

Array<int64> HostMath::ones_int(Shape shape, int64 coef, CudaConn* pConn) {
	return Array<int64>::ones(shape, coef);
}

void HostMath::shuffle(int64 size, int64* nums) {
	FuncTimer func_timer("shuffle");
	for (int64 n = 0; n < size; n++) {
		int64 m = Random::dice(size);
#ifndef NORANDOM
		swap<int64>(nums[n], nums[m]);
#endif
	}
}

Array<int64> HostMath::arange(int64 range) {
	FuncTimer func_timer("arange");
	Shape shape(range);
	Array<int64> arr(shape);
	int64* nums = arr.data_ptr();
	for (int64 n = 0; n < range; n++) nums[n] = n;
	return arr;
}
 
void HostMath::array_copy(Array<float>& dst, Array<float> src, Axis dst_axis, Axis src_axis) {
	FuncTimer func_timer("array_copy1(float)");
	assert(dst_axis.m_nDim == src_axis.m_nDim);

	ArrayLoopAxisInfo<float>* src_info = new ArrayLoopAxisInfo<float>[src_axis.m_nDim];
	ArrayLoopAxisInfo<float>* dst_info = new ArrayLoopAxisInfo<float>[dst_axis.m_nDim];

	for (int64 n = 0; n < src_axis.m_nDim; n++) {
		src_info[n].setup(src, src_axis[n], n);
		dst_info[n].setup(dst, dst_axis[n], n);
		assert(src_info[n].m_loop_count == dst_info[n].m_loop_count);
	}

	float* pSrc = src.data_ptr();
	float* pDst = dst.data_ptr();

	m_array_copy(pSrc, pDst, src_info, dst_info, 0, src_axis.m_nDim);

	delete[] src_info;
	delete[] dst_info;
}

void HostMath::array_copy(Array<float>& dst, Axis dst_axis, float val) {
	FuncTimer func_timer("array_copy2(float)");
	ArrayLoopAxisInfo<float>* dst_info = new ArrayLoopAxisInfo<float>[dst_axis.m_nDim];

	for (int64 n = 0; n < dst_axis.m_nDim; n++) {
		dst_info[n].setup(dst, dst_axis[n], n);
	}

	float* pDst = dst.data_ptr();
	m_array_copy(pDst, dst_info, 0, dst_axis.m_nDim, val);
	delete[] dst_info;
}

void HostMath::array_add(Array<float>& dst, Array<float> src, Axis dst_axis, Axis src_axis) {
	FuncTimer func_timer("array_add");
	assert(dst_axis.m_nDim == src_axis.m_nDim);

	ArrayLoopAxisInfo<float>* src_info = new ArrayLoopAxisInfo<float>[src_axis.m_nDim];
	ArrayLoopAxisInfo<float>* dst_info = new ArrayLoopAxisInfo<float>[dst_axis.m_nDim];

	for (int64 n = 0; n < src_axis.m_nDim; n++) {
		src_info[n].setup(src, src_axis[n], n);
		dst_info[n].setup(dst, dst_axis[n], n);
		assert(src_info[n].m_loop_count == dst_info[n].m_loop_count);
	}

	float* pSrc = src.data_ptr();
	float* pDst = dst.data_ptr();

	m_array_add(pSrc, pDst, src_info, dst_info, 0, src_axis.m_nDim);

	delete[] src_info;
	delete[] dst_info;
}

void HostMath::array_sub(Array<float>& dst, Array<float> src, Axis dst_axis, Axis src_axis) {
	FuncTimer func_timer("array_sub");
	assert(dst_axis.m_nDim == src_axis.m_nDim);

	ArrayLoopAxisInfo<float>* src_info = new ArrayLoopAxisInfo<float>[src_axis.m_nDim];
	ArrayLoopAxisInfo<float>* dst_info = new ArrayLoopAxisInfo<float>[dst_axis.m_nDim];

	for (int64 n = 0; n < src_axis.m_nDim; n++) {
		src_info[n].setup(src, src_axis[n], n);
		dst_info[n].setup(dst, dst_axis[n], n);
		assert(src_info[n].m_loop_count == dst_info[n].m_loop_count);
	}

	float* pSrc = src.data_ptr();
	float* pDst = dst.data_ptr();

	m_array_sub(pSrc, pDst, src_info, dst_info, 0, src_axis.m_nDim);

	delete[] src_info;
	delete[] dst_info;
}

template <class T>
Array<T> HostMath::copy_array(Array<T> src, Axis src_axis) {
	FuncTimer func_timer("array_copy3(T)");
	int64 ax_size[KAI_MAX_DIM];

	ArrayLoopAxisInfo<T>* src_info = new ArrayLoopAxisInfo<T>[src_axis.m_nDim];
	ArrayLoopAxisInfo<T>* dst_info = new ArrayLoopAxisInfo<T>[src_axis.m_nDim];

	int64 pack_size = 1;

	for (int64 n = src_axis.m_nDim - 1; n >= 0; n--) {
		src_info[n].setup(src, src_axis[n], n);
		ax_size[n] = src_info[n].m_loop_count;
		dst_info[n].setup(ax_size[n], pack_size);
		pack_size *= ax_size[n];
	}

	Shape shape(ax_size, src_axis.m_nDim);
	Array<T> dst(shape);

	T* pSrc = src.data_ptr();
	T* pDst = dst.data_ptr();

	m_array_copy(pSrc, pDst, src_info, dst_info, 0, src_axis.m_nDim);

	delete[] src_info;
	delete[] dst_info;

	return dst;
}

template <class T>
void HostMath::m_array_copy(T* pSrc, T* pDst, ArrayLoopAxisInfo<T>* src_info, ArrayLoopAxisInfo<T>* dst_info, int64 nth, int64 dim) {
	if (nth == dim) {
		*pDst = *pSrc;
	}
	else {
		pSrc += src_info[nth].m_start_offset;
		pDst += dst_info[nth].m_start_offset;

		int64 count = src_info[nth].m_loop_count;

		for (int64 n = 0; n < count; n++) {
			m_array_copy(pSrc, pDst, src_info, dst_info, nth+1, dim);
			pSrc += src_info[nth].m_skip_size;
			pDst += dst_info[nth].m_skip_size;
		}
	}
}

void HostMath::m_array_copy(float* pDst, ArrayLoopAxisInfo<float>* dst_info, int64 nth, int64 dim, float val) {
	if (nth == dim) *pDst = val;
	else {
		pDst += dst_info[nth].m_start_offset;

		int64 count = dst_info[nth].m_loop_count;

		for (int64 n = 0; n < count; n++) {
			m_array_copy(pDst, dst_info, nth+1, dim, val);
			pDst += dst_info[nth].m_skip_size;
		}
	}
}

void HostMath::m_array_add(float* pSrc, float* pDst, ArrayLoopAxisInfo<float>* src_info, ArrayLoopAxisInfo<float>* dst_info, int64 nth, int64 dim) {
	if (nth == dim) *pDst += *pSrc;
	else {
		pSrc += src_info[nth].m_start_offset;
		pDst += dst_info[nth].m_start_offset;

		int64 count = src_info[nth].m_loop_count;

		for (int64 n = 0; n < count; n++) {
			m_array_add(pSrc, pDst, src_info, dst_info, nth + 1, dim);
			pSrc += src_info[nth].m_skip_size;
			pDst += dst_info[nth].m_skip_size;
		}
	}
}

void HostMath::m_array_sub(float* pSrc, float* pDst, ArrayLoopAxisInfo<float>* src_info, ArrayLoopAxisInfo<float>* dst_info, int64 nth, int64 dim) {
	if (nth == dim) *pDst += *pSrc;
	else {
		pSrc += src_info[nth].m_start_offset;
		pDst += dst_info[nth].m_start_offset;

		int64 count = src_info[nth].m_loop_count;

		for (int64 n = 0; n < count; n++) {
			m_array_sub(pSrc, pDst, src_info, dst_info, nth + 1, dim);
			pSrc += src_info[nth].m_skip_size;
			pDst += dst_info[nth].m_skip_size;
		}
	}
}

Array<float> HostMath::matmul(Array<float> a, Array<float> b) {
	FuncTimer func_timer("matmul_no_cuda");

	Shape a_shape = a.shape();
	Shape b_shape = b.shape();

	assert(a_shape.size() >= 2);
	assert(b_shape.size() >= 2);

	int64 nrows = a_shape[-2], nvecs = a_shape[-1];
	int64 ncols = b_shape[-1];
		
	assert(nvecs == b_shape[-2]);

	int64 arest = a_shape.total_size() / (nrows * nvecs);

	assert(arest == b_shape.total_size() / (ncols * nvecs));

	Array<float> mult = zeros(Shape(a_shape.replace_nth(-1, ncols)));

	float* a_base = a.data_ptr();
	float* b_base = b.data_ptr();
	float* mp = mult.data_ptr();

	for (int64 r = 0; r < arest; r++) {
		for (int64 n = 0; n < nrows; n++) {
			for (int64 m = 0; m < ncols; m++) {
				float dot_sum = 0;
				float* ap = a_base + n * nvecs;
				float* bp = b_base + m;
				for (int64 k = 0; k < nvecs; k++) {
					dot_sum += *ap * *bp;
					ap++;
					bp += ncols;
				}
				*mp++ = dot_sum;
			}
		}
		a_base += nrows * nvecs;
		b_base += ncols * nvecs;
	}

	return mult;
}

Array<float> HostMath::dotmul(Array<float> a, Array<float> b) {
	FuncTimer func_timer("dotmul_no_cuda");

	Shape a_shape = a.shape();
	Shape b_shape = b.shape();
	assert(a_shape.size() >= 2);
	assert(b_shape.size() >= 2);
	assert(a_shape[0] == b_shape[0]);
	assert(a_shape[-1] == b_shape[-1]);

	int64 mb_size = a_shape[0];
	int64 vec_size = a_shape[-1];

	int64 a_rest = a.total_size() / (mb_size * vec_size);
	int64 b_rest = b.total_size() / (mb_size * vec_size);

	Array<float> mult = zeros(Shape(mb_size, a_rest * b_rest));

	float* ap = a.data_ptr();
	float* bp_base = b.data_ptr();
	float* mp = mult.data_ptr();

	for (int64 n = 0; n < mb_size; n++, bp_base += b_rest * vec_size) {
		for (int64 ma = 0; ma < a_rest; ma++, ap += vec_size) {
			float* bp = bp_base;
			for (int64 mb = 0; mb < b_rest; mb++, mp++, bp += vec_size) {
				for (int64 k = 0; k < vec_size; k++) {
					*mp += ap[k] * bp[k];
				}
			}
		}
	}

	assert(ap == a.data_ptr() + a.total_size());
	assert(bp_base == b.data_ptr() + b.total_size());
	assert(mp == mult.data_ptr() + mult.total_size());

	return mult;
}

Array<float> HostMath::dotmul_derv(Array<float> G_mult, Array<float> a) {
	FuncTimer func_timer("dotmul_derv");

	Shape g_shape = G_mult.shape();
	Shape a_shape = a.shape();

	if (a_shape.size() == 2) {
		a_shape = Shape(a_shape[0], 1, a_shape[1]);
		a = a.reshape(a_shape);
	}

	assert(g_shape.size() == 2);
	assert(a_shape.size() == 3);
	assert(g_shape[0] == a_shape[0]);
	assert(g_shape[1] % a_shape[1] == 0);

	int64 mb_size = g_shape[0];
	int64 nom_size = g_shape[1];
	int64 vec_size = a_shape[2];

	int64 a_rest = a.total_size() / (mb_size * vec_size);
	int64 b_rest = nom_size / a_rest;

	assert(nom_size % a_rest == 0);

	Array<float> G_b = zeros(Shape(mb_size, b_rest, vec_size));

	float* ap_base = a.data_ptr();
	float* bp_base = G_b.data_ptr();
	float* mp = G_mult.data_ptr();

	for (int64 n = 0; n < mb_size; n++, ap_base += a_rest * vec_size, bp_base += b_rest * vec_size) {
		for (int64 ma = 0; ma < nom_size; ma++, mp++) {
			float* ap = ap_base + ma / b_rest * vec_size;
			float* bp = bp_base + ma / a_rest * vec_size;
			for (int64 k = 0; k < vec_size; k++) {
				*bp += ap[k] * *mp;
			}
		}
	}

	assert(ap_base == a.data_ptr() + a.total_size());
	assert(bp_base == G_b.data_ptr() + G_b.total_size());
	assert(mp == G_mult.data_ptr() + G_mult.total_size());

	return G_b;
}

template <class T>
void HostMath::bin_op(Array<T> a, Array<T> b, T(*op)(T a, T b)) {
	FuncTimer func_timer("bin_op1");

	Shape a_shape = a.shape();
	Shape b_shape = b.shape();

	assert(a_shape.size() >= b_shape.size());

	for (int64 n = 0; n < b_shape.size(); n++) {
		int64 a_nth = a_shape.size() - b_shape.size() + n;
		if (a_shape[a_nth] != b_shape[n]) throw KaiException(KERR_ASSERT);
	}

	int64 a_size = a_shape.total_size();
	int64 b_size = b_shape.total_size();

	T* ap = a.data_ptr();
	T* bp = b.data_ptr();
	T* bp_end = bp + b_size;

	if (a_size == b_size) {
		for (int64 n = 0; n < a_size; n++, ap++) {
			*ap = op(*ap, *bp++);
		}
	}
	else {
		for (int64 n = 0; n < a_size; n++, ap++) {
			*ap = op(*ap, *bp++);
			if (bp == bp_end) bp = b.data_ptr();
		}
	}
}

template <class T>
void HostMath::bin_op(Array<T> a, T b, T(*op)(T a, T b)) {
	FuncTimer func_timer("bin_op2");
	T* p = a.data_ptr();
	int64 size = a.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		*p = op(*p, b);
	}
}

template <class T>
void HostMath::log_bin_op(Array<bool> result, Array<T> a, Array<T> b, bool(*op)(T a, T b)) {
	FuncTimer func_timer("log_bin_op1");
	Shape r_shape = result.shape();
	Shape a_shape = a.shape();
	Shape b_shape = b.shape();

	assert(r_shape == b_shape);
	assert(a_shape == b_shape);

	int64 a_size = a_shape.total_size();

	T* ap = a.data_ptr();
	T* bp = b.data_ptr();
	bool* rp = result.data_ptr();

	for (int64 n = 0; n < a_size; n++, ap++, bp++, rp++) {
		*rp = op(*ap, *bp);
	}
}

template <class T>
void HostMath::log_bin_op(Array<bool> result, Array<T> a, T b, bool(*op)(T a, T b)) {
	FuncTimer func_timer("log_bin_op2");
	Shape r_shape = result.shape();
	Shape a_shape = a.shape();

	assert(r_shape == a_shape);

	int64 a_size = a_shape.total_size();


	T* ap = a.data_ptr();
	bool* rp = result.data_ptr();

	for (int64 n = 0; n < a_size; n++, ap++, rp++) {
		*rp = op(*ap, b);
	}
}

Array<float> HostMath::abs(Array<float> a) {
	FuncTimer func_timer("abs");
	Array<float> dst = a.deepcopy();

	float* p = dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		if (*p < 0) *p *= -1;
	}

	return dst;
}

Array<float> HostMath::square(Array<float> a) {
	FuncTimer func_timer("square");
	Array<float> dst = a.deepcopy();

	float* p = dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		*p = *p * *p;
	}

	return dst;
}

Array<float> HostMath::sqrt(Array<float> a) {
	FuncTimer func_timer("sqrt");
	Array<float> dst = a.deepcopy();

	float* p = dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		*p = (float) ::sqrtf(*p);
	}

	return dst;
}

int64 HostMath::count_in_range(Array<float> a, float min_val, float max_val) {
	FuncTimer func_timer("count_in_range");
	int64 size = a.total_size();
	int64 cnt = 0;

	float* p = a.data_ptr();

	for (int64 n = 0; n < size; n++, p++) {
		if (*p >= min_val && *p <= max_val) cnt++;
	}

	return cnt;
}

int64 HostMath::true_count(Array<bool> a) {
	FuncTimer func_timer("true_count");
	int64 size = a.total_size();
	int64 cnt = 0;

	bool* p = a.data_ptr();

	for (int64 n = 0; n < size; n++, p++) {
		if (*p) cnt++;
	}

	return cnt;
}

float HostMath::sum(Array<float> a) {
	FuncTimer func_timer("sum");
	int64 size = a.shape().total_size();
	float sum = 0;

	float* p = a.data_ptr();

	for (int64 n = 0; n < size; n++, p++) sum += *p;

	return sum;
}

int64 HostMath::sum(Array<int64> a) {
	FuncTimer func_timer("sum");
	int64 size = a.shape().total_size();
	int64 sum = 0;

	int64* p = a.data_ptr();

	for (int64 n = 0; n < size; n++, p++) sum += *p;

	return sum;
}

float HostMath::max(Array<float> a, int64* pargmax) {
	FuncTimer func_timer("sum");
	int64 size = a.shape().total_size();

	float* p = a.data_ptr();
	float max_val = *p++;

	int64 arg = 0;

	for (int64 n = 1; n < size; n++, p++) {
		if (*p > max_val) {
			max_val = *p;
			arg = n;
		}
	}

	if (pargmax) *pargmax = arg;

	return max_val;
}

float HostMath::mean(Array<float> a) {
	FuncTimer func_timer("mean(float)");
	int64 size = a.shape().total_size();
	float sum = 0;

	float* p = a.data_ptr();

	for (int64 n = 0; n < size; n++, p++) sum += *p;

	return sum / (float)size;
}

float HostMath::mean(Array<bool> a) {
	FuncTimer func_timer("mean(bool)");
	int64 size = a.shape().total_size();
	int64 sum = 0;

	bool* p = a.data_ptr();

	for (int64 n = 0; n < size; n++, p++) if (*p) sum++;

	return (float)sum / (float)size;
}

float HostMath::mean(List a) {
	FuncTimer func_timer("mean(list)");
	int64 size =  a.size();
	float sum = 0;

	for (int64 n = 0; n < size; n++) sum += (float)a[n];

	return sum / (float)size;
}

float HostMath::square_sum(Array<float> arr) {
	return arr.square().sum();
}

Array<float> HostMath::sign(Array<float> a) {
	FuncTimer func_timer("sign");
	Array<float> dst = a.deepcopy();

	float* p = (float*)dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		*p = (float)((*p > 0) ? 1 : (*p < 0) ? -1 : 0);
	}

	return dst;
}

Array<int64> HostMath::round(Array<float> a) {
	FuncTimer func_timer("round");
	Array<int64> dst(a.shape());

	int64* p = dst.data_ptr();
	float* q = a.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++, q++) {
		*p =  (int64) ::lroundf(*q);
	}

	return dst;
}

Array<int64> HostMath::binary_row_to_int(Array<int64> a) {
	FuncTimer func_timer("binary_row_to_int");
	assert(a.dim() == 2);
	Array<int64> dst(Shape(a.axis_size(0)));

	int64* p = dst.data_ptr();
	int64* q = a.data_ptr();
	int64 size = a.axis_size(0);
	int64 cols = a.axis_size(1);

	for (int64 n = 0; n < size; n++, p++) {
		int64 val = 0;
		for (int64 m = 0; m < cols; m++, q++) {
			assert(*q == 0 || *q == 1);
			val = 2 * val + *q;
		}
		*p = val;
	}

	return dst;
}

Array<float> HostMath::fetch_rows(Array<float> a, vector<int64> rows) {
	FuncTimer func_timer("fetch_rows");
	assert(a.dim() == 2);

	int64 size = rows.size();
	int64 ncol = a.axis_size(1);

	Array<float> dst(Shape(size, ncol));

	float* p = dst.data_ptr();
	float* q = a.data_ptr();

	for (int64 n = 0; n < size; n++, p += ncol) {
		assert(rows[n] >= 0 && rows[n] < a.axis_size(0));
		memcpy(p, q + rows[n] * ncol, sizeof(float) * ncol);
	}

	return dst;
}

Array<float> HostMath::dotsum(Array<float> a, Array<float> b) {
	FuncTimer func_timer("dotsum");
	assert(a.dim() == 2);
	assert(b.dim() == 1);
	assert(a.axis_size(1) == b.axis_size(0));

	int64 nrow = a.axis_size(0), ncol = a.axis_size(1);

	Shape shape(nrow);
	Array<float> dst(shape);

	float* p = dst.data_ptr();
	float* q = a.data_ptr();
	float* r = b.data_ptr();

	for (int64 n = 0; n < nrow; n++) {
		p[n] = 0;
		for (int64 m = 0; m < ncol; m++) {
			p[n] += *q++ * r[m];
		}
	}

	return dst;
}

template <class T>
Array<T> HostMath::extract_selected(Array<T> arr, Array<bool> selector) {
	FuncTimer func_timer("extract_selected");
	assert(selector.dim() == 1);
	assert(arr.axis_size(0) == selector.axis_size(0));

	int64 nrow = arr.axis_size(0), ncol = arr.total_size() / nrow;
	int64 nresult = selector.true_count();

	Array<T> dst(arr.shape().remove_front().add_front(nresult));

	T* p = dst.data_ptr();
	T* q = arr.data_ptr();
	bool* r = selector.data_ptr();

	for (int64 n = 0; n < nrow; n++, q += ncol) {
		if (!r[n]) continue;
		memcpy(p, q, sizeof(T) * ncol);
		p += ncol;
	}

	assert(p == dst.data_ptr() + nresult*ncol);

	return dst;
}

template <class T>
Array<T> HostMath::fill_selected(Array<T> arr, Array<T> slices, Array<bool> selector) {
	FuncTimer func_timer("fill_selected");
	assert(selector.dim() == 1);
	assert(arr.axis_size(0) == selector.axis_size(0));

	int64 nrow = arr.axis_size(0), ncol = arr.total_size() / nrow;

	Array<T> dst = arr.deepcopy();

	T* p = slices.data_ptr();
	T* q = dst.data_ptr();
	bool* r = selector.data_ptr();

	for (int64 n = 0; n < nrow; n++, q += ncol) {
		if (!r[n]) continue;
		memcpy(q, p, sizeof(T) * ncol);
		p += ncol;
	}

	assert(p == slices.data_ptr() + selector.true_count() * ncol);

	return dst;
}

Array<float> HostMath::get_row(Array<float> arr, int64 nth) {
	FuncTimer func_timer("get_row");
	assert(arr.dim() == 2);
	assert(nth >= 0 && nth < arr.axis_size(0));

	int64 ncol = arr.axis_size(1);

	Shape shape(ncol);
	Array<float> dst(shape);

	float* p = arr.data_ptr() + nth * ncol;
	float* q = dst.data_ptr();

	memcpy(q, p, sizeof(float) * ncol);

	return dst;
}

Array<int64> HostMath::get_col(Array<int64> arr, int64 nth) {
	FuncTimer func_timer("get_col");
	assert(arr.dim() == 2);
	assert(nth >= 0 && nth < arr.axis_size(1));

	int64 nrow = arr.axis_size(0);
	int64 ncol = arr.axis_size(1);

	Shape shape(nrow);
	Array<int64> dst(shape);

	int64* p = arr.data_ptr() + nth;
	int64* q = dst.data_ptr();

	for (int64 n = 0; n < nrow; n++) {
		*q++ = *p;
		p += ncol;
	}

	return dst;
}

void HostMath::set_row(Array<float> arr, int64 nth, Array<float> row) {
	FuncTimer func_timer("set_row(row)");
	assert(arr.dim() == row.dim() + 1);
	assert(arr.total_size() == arr.axis_size(0) * row.total_size());
	assert(nth >= 0 && nth < arr.axis_size(0));

	int64 ncol = row.total_size();

	float* p = arr.data_ptr() + nth * ncol;
	float* q = row.data_ptr();

	memcpy(p, q, sizeof(float) * ncol);
}

void HostMath::set_row(Array<float> arr, int64 nth, float val) {
	FuncTimer func_timer("set_row(val)");
	assert(nth >= 0 && nth < arr.axis_size(0));

	int64 ncol = arr.total_size() / arr.axis_size(0);

	float* p = arr.data_ptr() + nth * ncol;

	for (int64 n = 0; n < ncol; n++, p++) {
		*p = val;
	}
}

void HostMath::sub_row(Array<float> arr, int64 nth, Array<float> row) {
	FuncTimer func_timer("sub_row");
	assert(arr.dim() == 2);
	assert(row.dim() == 1);
	assert(nth >= 0 && nth < arr.axis_size(0));
	assert(arr.axis_size(1) == row.axis_size(0));

	int64 ncol = arr.axis_size(1);

	float* p = arr.data_ptr() + nth * ncol;
	float* q = row.data_ptr();

	for (int64 n = 0; n < ncol; n++) {
		*p++ -= *q++;
	}
}

Array<float> HostMath::get_hash_match_point(Array<float> code1, Array<float> code2) {
	throw KaiException(KERR_ASSERT);
	return code1;
}

Array<float> HostMath::get_vector_dist(Array<float> code1, Array<float> code2) {
	throw KaiException(KERR_ASSERT);
	return code1;
}

static void m_quick_sort(float* ap, int64* xp, int64 nlow, int64 nhigh, int sign, int64 max_cnt) {
	if (nlow >= nhigh) return;

	float pivot = ap[xp[(nlow+nhigh)/2]];

	int64 nleft = nlow - 1;
	int64 nright = nhigh + 1;

	while (true) {
		while ((ap[xp[++nleft]] - pivot) * sign < 0);
		while ((ap[xp[--nright]] - pivot) * sign > 0);
		if (nleft >= nright) break;
		swap<int64>(xp[nleft], xp[nright]);
	}

	/*
	for (int64 n = nlow; n < nright; n++) {
		if ((ap[xp[n]] - pivot) * sign > 0) throw KaiException(KERR_ASSERT);
	}

	for (int64 n = nright+1; n < nhigh+1; n++) {
		if ((ap[xp[n]] - pivot) * sign < 0) throw KaiException(KERR_ASSERT);
	}
	*/

	m_quick_sort(ap, xp, nlow, nright, sign, max_cnt);
	if (nright < max_cnt) m_quick_sort(ap, xp, nright+1, nhigh, sign, max_cnt);
}

Array<int64> HostMath::sort_columns(Array<float> arr, sortdir dir, int64 max_cnt) {
	FuncTimer func_timer("sort");
	Shape shape = arr.shape();

	assert(shape.size() == 2);

	int64 nrow = arr.axis_size(0);
	int64 ncol = arr.axis_size(1);

	Array<int64> sort_idx(shape);

	int sign = (dir == sortdir::sd_asc) ? 1 : -1;

	for (int64 n = 0; n < nrow; n++) {
		float* ap = arr.data_ptr() + n * ncol;
		int64* xp = sort_idx.data_ptr() + n * ncol;

		for (int64 m = 0; m < ncol; m++) xp[m] = m;

		m_quick_sort(ap, xp, 0, ncol-1, sign, max_cnt);

		for (int64 m = 0; m < max_cnt - 1; m++) {
			if ((ap[xp[m]] - ap[xp[m + 1]]) * sign > 0) throw KaiException(KERR_ASSERT);
		}

		for (int64 m = max_cnt; m < ncol - 1; m++) {
			if ((ap[xp[max_cnt-1]] - ap[xp[m + 1]]) * sign > 0) throw KaiException(KERR_ASSERT);
		}
	}

	return sort_idx;
}

static float m_comp1(float* ap1, float* ap2, int64* xp, int64 n1, int64 n2, int sign1, int sign2) {
	float diff = (ap1[xp[n1]] - ap1[xp[n2]]) * sign1;
	if (diff != 0) return diff;
	return (ap2[xp[n1]] - ap2[xp[n2]]) * sign2;
}

static float m_comp2(float* ap1, float* ap2, int64* xp, int64 n1, float pivot1, float pivot2, int sign1, int sign2) {
	float diff = (ap1[xp[n1]] - pivot1) * sign1;
	if (diff != 0) return diff;
	return (ap2[xp[n1]] - pivot2) * sign2;
}

static void m_quick_sort(float* ap1, float* ap2, int64* xp, int64 nlow, int64 nhigh, int sign1, int sign2, int64 max_cnt) {
	if (nlow >= nhigh) return;

	float pivot1 = ap1[xp[(nlow + nhigh) / 2]];
	float pivot2 = ap2[xp[(nlow + nhigh) / 2]];

	int64 nleft = nlow - 1;
	int64 nright = nhigh + 1;

	while (true) {
		while (m_comp2(ap1, ap2, xp, ++nleft, pivot1, pivot2, sign1, sign2) < 0);
		while (m_comp2(ap1, ap2, xp, --nright, pivot1, pivot2, sign1, sign2) > 0);
		if (nleft >= nright) break;
		swap<int64>(xp[nleft], xp[nright]);
	}

	/*
	for (int64 n = nlow; n < nright; n++) {
		if (m_comp2(ap1, ap2, xp, n, pivot1, pivot2, sign1, sign2) > 0) throw KaiException(KERR_ASSERT);
	}

	for (int64 n = nright + 1; n < nhigh + 1; n++) {
		if (m_comp2(ap1, ap2, xp, n, pivot1, pivot2, sign1, sign2) < 0) throw KaiException(KERR_ASSERT);
	}
	*/

	m_quick_sort(ap1, ap2, xp, nlow, nright, sign1, sign2, max_cnt);
	
	if (nright < max_cnt) m_quick_sort(ap1, ap2, xp, nright + 1, nhigh, sign1, sign2, max_cnt);
}

Array<int64> HostMath::sort_columns(Array<float> arr1, sortdir dir1, Array<float> arr2, sortdir dir2, int64 max_cnt) {
	FuncTimer func_timer("sort");
	
	Shape shape = arr1.shape();

	assert(shape.size() == 2);
	assert(arr2.shape() == shape);

	int64 nrow = arr1.axis_size(0);
	int64 ncol = arr1.axis_size(1);

	Array<int64> sort_idx(shape);

	int sign1 = (dir1 == sortdir::sd_asc) ? 1 : -1;
	int sign2 = (dir2 == sortdir::sd_asc) ? 1 : -1;

	for (int64 n = 0; n < nrow; n++) {
		float* ap1 = arr1.data_ptr() + n * ncol;
		float* ap2 = arr2.data_ptr() + n * ncol;
		int64* xp = sort_idx.data_ptr() + n * ncol;

		for (int64 m = 0; m < ncol; m++) xp[m] = m;

		m_quick_sort(ap1, ap2, xp, 0, ncol - 1, sign1, sign2, max_cnt);

		for (int64 m = 0; m < max_cnt - 1; m++) {
			if (m_comp1(ap1, ap2, xp, m, m+1, sign1, sign2) > 0) throw KaiException(KERR_ASSERT);
		}
	}

	return sort_idx;
}

Array<float> HostMath::to_float(Array<unsigned char> a) {
	FuncTimer func_timer("to_float(uchar)");
	Array<float> dst(a.shape());

	float* p = dst.data_ptr();
	unsigned char* s = a.data_ptr();
	int64 size = dst.total_size();

	for (int64 n = 0; n < size; n++, p++, s++) {
		*p = (float)*s;
	}

	return dst;
}

Array<float> HostMath::to_float(Array<int64> a) {
	FuncTimer func_timer("to_float(int64)");
	Array<float> dst(a.shape());

	float* p = dst.data_ptr();
	int64* s = a.data_ptr();
	int64 size = dst.total_size();

	for (int64 n = 0; n < size; n++, p++, s++) {
		*p = (float)*s;
	}

	return dst;
}

Array<float> HostMath::to_float(Array<bool> a) {
	FuncTimer func_timer("to_float(bool)");
	Array<float> dst(a.shape());

	float* p = dst.data_ptr();
	bool* s = a.data_ptr();
	int64 size = dst.total_size();

	for (int64 n = 0; n < size; n++, p++, s++) {
		*p = (float)*s;
	}

	return dst;
}

Array<float> HostMath::log(Array<float> a) {
	FuncTimer func_timer("log");
	Array<float> dst = a.deepcopy();

	float* p = (float*)dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		*p = (float) ::log(*p);
	}

	return dst;
}

Array<float> HostMath::exp(Array<float> a) {
	FuncTimer func_timer("exp");
	Array<float> dst = a.deepcopy();

	float* p = (float*)dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		*p = (float) ::expf(*p);
	}

	return dst;
}

Array<float> HostMath::maximum(Array<float> ans, Array<float> est) {
	FuncTimer func_timer("maximum(array)");
	throw KaiException(KERR_ASSERT);
	return Array<float>();
}

Array<float> HostMath::maximum(Array<float> ans, float est) {
	FuncTimer func_timer("maximum(val)");
	throw KaiException(KERR_ASSERT);
	return Array<float>();
}

Array<float> HostMath::sigmoid_cross_entropy_with_logits(Array<float> a, Array<float> b) { // 간단한 단일 루프 형태로 수정할 것
	FuncTimer func_timer("sigmoid_cross_entropy_with_logits");
	Array<float> clone = a.deepcopy();
	HostMath::bin_op(clone, b, HostMath::ms_bin_op_sigmoid_cross_entropy_with_logits);
	return clone;
}

float HostMath::ms_bin_op_sigmoid_cross_entropy_with_logits(float z, float x) {
	float relu_term = x > 0 ? x : 0;
	float prod_term = x * z;
	float abs_term = x > 0 ? x : -x;
	float log_term = (float) ::log(1.0 + ::expf(-abs_term));
	return relu_term - prod_term + log_term;
}

Array<float> HostMath::softmax_cross_entropy_with_logits(Array<float> ans, Array<float> est) {
	FuncTimer func_timer("softmax_cross_entropy_with_logits");
	Array<float> probs = softmax(est);
	Array<float> logs = hmath.log(probs + 1.0e-10f);
	Array<float> log_ans = ans * logs;
	Array<float> sums = hmath.sum(log_ans, 0);
		
	return sums * -1.0f;
}

Array<float> HostMath::softmax_cross_entropy_with_logits_idx(Array<int64> ans, Array<float> est) {
	FuncTimer func_timer("softmax_cross_entropy_with_logits_idx");
	Array<float> probs = softmax(est);
	Array<float> logs = hmath.log(probs + 1.0e-10f);
	Array<float> log_ans = logs.select_col(ans);
	Array<float> sums = hmath.sum(log_ans, 0);

	return sums * -1.0f;
}

Array<float> HostMath::softmax_cross_entropy_with_logits_1st(Array<float> est) {
	FuncTimer func_timer("softmax_cross_entropy_with_logits_1st");
	Array<float> probs = softmax(est);
	Array<float> logs = hmath.log(probs + 1.0e-10f);
	Array<float> log_ans;
	log_ans = logs[Axis(0, _all_)];
	Array<float> sums = hmath.sum(log_ans, 0);

	return sums * -1.0f;
}

Array<float> HostMath::relu(Array<float> affine) {
	FuncTimer func_timer("relu");
	Array<float> dst = affine.deepcopy();

	float* p = dst.data_ptr();
	int64 size = dst.shape().total_size();

	for(int64 n = 0; n < size; n++, p++) {
		if (*p < 0) *p = 0;
	}

	return dst;
}

Array<float> HostMath::sigmoid(Array<float> affine) {
	FuncTimer func_timer("sigmoid");
	Array<float> dst = affine.deepcopy();

	float* p = dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		float x = *p;
		float term1 = (x > 0) ? 1 : (float)::expf(x);
		float term2 = (float) (1.0f + (float)::expf((x > 0) ? -x : x));
		*p = term1 / term2;
	}

	return dst;
}

Array<float> HostMath::softmax(Array<float> a) {
	FuncTimer func_timer("softmax");
	
	assert(a.shape().size() >= 2);

	Shape ashape = a.shape();

	a = a.reshape(Shape(-1, a.axis_size(-1)));
	
	Array<float> max_elem = hmath.max(a, (int64)0);
	Array<float> diff = (a.transpose() - max_elem).transpose(); // 최적화 필요
	Array<float> exp_diff = hmath.exp(diff);
	Array<float> sum_exp = hmath.sum(exp_diff, 0);
	Array<float> probs = (exp_diff.transpose() / sum_exp).transpose(); // 최적화 필요

	probs = probs.reshape(ashape);

	return probs;
}

Array<float> HostMath::tanh(Array<float> affine) {
	FuncTimer func_timer("tanh");
	Array<float> dst = affine.deepcopy();

	float* p = dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		float x = *p;
		float term1 = (x > 0) ? (1.0f - (float)::expf(-x)) : ((float)::expf(x) - 1.0f);
		float term2 = (x > 0) ? (1.0f + (float)::expf(-x)) : ((float)::expf(x) + 1.0f);
		*p = term1 / term2;
	}

	return dst;
}

Array<float> HostMath::leaky_relu(Array<float> affine, float alpha) {
	FuncTimer func_timer("leaky_relu");
	Array<float> dst = affine.deepcopy();

	float* p = dst.data_ptr();
	int64 size = dst.shape().total_size();

	for (int64 n = 0; n < size; n++, p++) {
		if (*p < 0) *p *= alpha;
	}

	return dst;
}

Array<float> HostMath::gelu(Array<float> x) {
	FuncTimer func_timer("gelu");
	Array<float> coef = tanh(x * 0.797885f + power(x, 3.0f) * 0.035677f) + 1.0f;
	return x * 0.5f * coef;
}

Array<float> HostMath::relu_derv(Array<float> y) {
	FuncTimer func_timer("relu_derv");
	return hmath.sign(y);
}

Array<float> HostMath::sigmoid_derv(Array<float> y) {
	FuncTimer func_timer("sigmoid_derv");
	return y * (1.0f - y);
}

Array<float> HostMath::softmax_derv(Array<float> y) {
	FuncTimer func_timer("softmax_derv_no_cuda");
	//assert(y.dim() == 2);

	int64 nom_size = y.axis_size(-1);
	int64 head_size = y.total_size() / nom_size;

	Array<float> derv(y.shape().append(nom_size));

	float* dp = derv.data_ptr();
	float* yp = y.data_ptr();

	for (int64 n = 0; n < head_size; n++, yp += nom_size) {
		for (int64 m = 0; m < nom_size; m++) {
			float* dp1 = dp + m;
			for (int64 k = 0; k < nom_size; k++, dp++) {
				*dp = - yp[m] * yp[k];
			}
			*dp1 += yp[m];
		}
	}

	return derv;
}

Array<float> HostMath::tanh_derv(Array<float> y) {
	FuncTimer func_timer("tanh_derv");
	return (y + 1.0f) * (1.0f - y);
}

Array<float> HostMath::leaky_relu_derv(Array<float> y, float alpha) {
	FuncTimer func_timer("leaky_relu_derv");
	Array<float> grad(y.shape());

	float* yp = y.data_ptr();
	float* gp = grad.data_ptr();

	int64 size = y.shape().total_size();

	for (int64 n = 0; n < size; n++, yp++, gp++) {
		*gp = (*yp > 0) ? 1.0f : alpha;
	}

	return grad;
}

Array<float> HostMath::gelu_derv(Array<float> x) {
	FuncTimer func_timer("gelu_derv");
	Array<float> A = tanh(x * 0.797885f + hmath.power(x, 3.0f) * 0.035677f);
	Array<float> B = x * x * 2 * 0.035677f + 0.797885f;
	return (A + 1) * (-0.5f) * (x * (A - 1) * B + 1.0f);
}

Array<float> HostMath::avg(Array<float> a, int64 axis) {
	FuncTimer func_timer("avg");
	Array<float> sumarr = sum(a, axis);
	int64 size = a.total_size() / a.axis_size(axis);
	return sumarr / (float)size;
}

Array<float> HostMath::var(Array<float> a, int64 axis, Array<float>* pavg) {
	FuncTimer func_timer("var");
	
	int64 size1, size2, size3;
	
	axis = ax_mod(axis, a.dim());

	a.split_axis_size(axis, &size1, &size2, &size3);

	Array<float> s = hmath.zeros(Shape(size2));
	Array<float> v = hmath.zeros(Shape(size2));

	float* ap = a.data_ptr();
	float* sp = s.data_ptr();
	float* vp = v.data_ptr();

	for (int64 n = 0; n < size1; n++) {
		for (int64 m = 0; m < size2; m++) {
			for (int64 k = 0; k < size3; k++) {
				float an = *ap++;
				sp[m] += an;
				vp[m] += an * an;
			}
		}
	}

	int64 cnt = size1 * size3;

	for (int64 m = 0; m < size2; m++) {
		sp[m] /= (float) cnt;
		vp[m] = vp[m] / (float) cnt - sp[m] * sp[m];
	}

	if (pavg != NULL) *pavg = s;

	return v;
}

Array<float> HostMath::sum(Array<float> a, int64 axis) {
	FuncTimer func_timer("sum");
	Shape a_shape = a.shape();

	axis = ax_mod(axis, a_shape.size());

	Array<float> sumarr = hmath.zeros(a_shape[axis]);

	float* ap = a.data_ptr();
	float* bp_base = sumarr.data_ptr();

	int64 size = a_shape.total_size();

	int64 right = 1;
	int64 me_size = a_shape[axis];

	for (int64 n = axis + 1; n < a_shape.size(); n++) {
		right *= a_shape[n];
	}

	for (int64 n = 0; n < size; n++, ap++) {
		float* bp = bp_base + n / right % me_size;
		*bp += *ap;
	}

	return sumarr;
}

Array<float> HostMath::max(Array<float> a, int64 axis) {
	FuncTimer func_timer("max");
	Shape a_shape = a.shape();

	axis = ax_mod(axis, a_shape.size());

	Array<float> max_arr = hmath.ones(a_shape[axis])* (-FLT_MAX);

	float* ap = a.data_ptr();
	float* bp_base = max_arr.data_ptr();

	int64 size = a_shape.total_size();

	int64 right = 1;
	int64 me_size = a_shape[axis];

	for (int64 n = axis + 1; n < a_shape.size(); n++) {
		right *= a_shape[n];
	}

	for (int64 n = 0; n < size; n++, ap++) {
		float* bp = bp_base + n / right % me_size;
		if (*ap > *bp) *bp = *ap;
	}

	return max_arr;
}

Array<int64> HostMath::argmax(Array<float> a, int64 axis) {
	FuncTimer func_timer("argmax");
	Array<int64> arg;
	maxarg(a, axis, arg);
	return arg;
}

Array<float> HostMath::expand(Array<float> arr, int64 ratio, int64 axis) {
	FuncTimer func_timer("expand");
	Shape shape = arr.shape();

	shape[axis] *= ratio;

	Array<float> expanded(shape);

	float* ap = arr.data_ptr();
	float* ep = expanded.data_ptr();

	int64 sub_size = 1;
	for (int64 n = axis+1; n < shape.size(); n++) sub_size *= shape[n];
	int64 sup_size = arr.total_size() / sub_size;

	for (int64 n = 0; n < sup_size; n++, ap += sub_size) {
		for (int64 m = 0; m < ratio; m++, ep += sub_size) {
			memcpy(ep, ap, sizeof(float) * sub_size);
		}
	}

	return expanded;
}

Array<float> HostMath::sum_on_expanded(Array<float> expanded, int64 ratio, int64 axis) {
	FuncTimer func_timer("sum_on_expanded");
	Shape shape = expanded.shape();
	assert(shape[axis] % ratio == 0);
	shape[axis] /= ratio;

	Array<float> sum(shape);

	float* ap = sum.data_ptr();
	float* ep = expanded.data_ptr();

	memset(ap, 0, sizeof(float) * sum.total_size());

	int64 sub_size = 1;
	for (int64 n = axis+1; n < shape.size(); n++) sub_size *= shape[n];
	int64 sup_size = sum.total_size() / sub_size;

	for (int64 n = 0; n < sup_size; n++, ap += sub_size) {
		for (int64 m = 0; m < ratio; m++, ep += sub_size) {
			for (int64 k = 0; k < sub_size; k++) ap[k] += ep[k];
		}
	}

	return sum;
}

Array<bool> HostMath::compare_rows(Array<float> arr1, Array<float> arr2) {
	int64 nrow = arr1.axis_size(0);
	int64 ncol = arr1.total_size() / nrow;

	assert(nrow == arr2.axis_size(0));
	assert(arr1.total_size() == arr2.total_size());

	Shape shape(nrow);
	Array<bool> comp(shape);

	bool* cp = comp.data_ptr();

	for (int64 n = 0; n < nrow; n++) {
		float* ap1 = arr1.data_ptr() + n * ncol;
		float* ap2 = arr2.data_ptr() + n * ncol;

		cp[n] = true;

		for (int64 m = 0; m < ncol; m++) {
			if (ap1[m] != ap2[m]) {
				cp[n] = false;
				break;
			}
		}
	}

	return comp;
}

Array<bool> HostMath::compare_rows(Array<int64> arr1, Array<int64> arr2) {
	int64 nrow = arr1.axis_size(0);
	int64 ncol = arr1.total_size() / nrow;

	assert(nrow == arr2.axis_size(0));
	assert(arr1.total_size() == arr2.total_size());

	Shape shape(nrow);
	Array<bool> comp(shape);

	bool* cp = comp.data_ptr();

	for (int64 n = 0; n < nrow; n++) {
		int64* ap1 = arr1.data_ptr() + n * ncol;
		int64* ap2 = arr2.data_ptr() + n * ncol;

		cp[n] = true;

		for (int64 m = 0; m < ncol; m++) {
			if (ap1[m] != ap2[m]) {
				cp[n] = false;
				break;
			}
		}
	}

	return comp;
}

Array<float> HostMath::maxarg(Array<float> a, int64 axis, Array<int64>& arg) {
	FuncTimer func_timer("maxarg");
	assert(axis == 0);

	Shape a_shape = a.shape();

	axis = ax_mod(axis, a_shape.size());

	Array<float> max_arr = hmath.ones(a_shape[axis]) * (-FLT_MAX);
	Array<int64> arg_arr(a_shape[axis]);

	float* ap = a.data_ptr();
	float* bp = max_arr.data_ptr();
	int64* rp = arg_arr.data_ptr();

	int64 res_size = max_arr.total_size();
	int64 vec_size = a.total_size() / res_size;

	for (int64 n = 0; n < res_size; n++, rp++, bp++) {
		*rp = 0;
		for (int64 m = 0; m < vec_size; m++, ap++) {
			if (*ap > * bp) {
				*bp = *ap;
				*rp = m;
			}
		}
	}

	arg = arg_arr;
	return max_arr;
}

Array<float> HostMath::mult_scalar(Array<float> arr, float other) {
	Array<float> clone(arr.shape());

	float* ap = arr.data_ptr();
	float* cp = clone.data_ptr();

	int64 size = arr.total_size();

	for (int64 n = 0; n < size; n++) {
		*cp++ = *ap++ * other;
	}

	return clone;
}

void HostMath::setarg(Array<float> a, Array<int64>& arg, Array<float> values) {
	FuncTimer func_timer("setarg");
	assert(a.dim() == 2 && arg.dim() == 1 && values.dim() == 1);
	assert(a.axis_size(0) == arg.axis_size(0));
	assert(a.axis_size(0) == values.axis_size(0));

	float* ap = a.data_ptr();
	float* bp = values.data_ptr();
	int64* xp = arg.data_ptr();

	int64 size = arg.total_size();
	int64 range = a.axis_size(1);

	for (int64 n = 0; n < size; n++, ap += range, bp++, xp++) {
		assert(*xp >= 0 && *xp < range);
		*(ap + *xp) = *bp;
	}
}

Array<float> HostMath::sigmoid_cross_entropy_with_logits_derv(Array<float> ans, Array<float> est) {
	FuncTimer func_timer("sigmoid_cross_entropy_with_logits_derv");
	return sigmoid(est) - ans;
}

Array<float> HostMath::softmax_cross_entropy_with_logits_derv(Array<float> ans, Array<float> est) {
	FuncTimer func_timer("softmax_cross_entropy_with_logits_derv");
	// 순전파 처리에서 계산한 softmax 결과 재활용 방안 고려 필요
	return softmax(est) - ans;
}

Array<float> HostMath::softmax_cross_entropy_with_logits_idx_derv(Array<int64> ans, Array<float> est) {
	FuncTimer func_timer("softmax_cross_entropy_with_logits_idx_derv");
	// 순전파 처리에서 계산한 softmax 결과 재활용 방안 고려 필요
	Array<float> probs = softmax(est);
	Array<float> entropy = probs.minus_1_on_idx(ans);
	return entropy;
}

Array<float> HostMath::softmax_cross_entropy_with_logits_1st_derv(Array<float> est) {
	FuncTimer func_timer("softmax_cross_entropy_with_logits_1st_derv");
	// 순전파 처리에서 계산한 softmax 결과 재활용 방안 고려 필요
	Array<float> probs = softmax(est);
	Array<float> entropy = probs.minus_1_on_1st();
	return entropy;
}

Array<float> HostMath::init_grid(Shape shape) {
	FuncTimer func_timer("init_grid");
	int64 dim = shape.size();
	Array<float> arr(shape.append(dim));

	int64 size = shape.total_size();

	float* ap = arr.data_ptr();

	ShapeCounter idx(shape);

	for (int64 n = 0; n < size; n++, ++idx) {
		for (int64 m = 0; m < dim; m++) {
			*ap++ = (float) idx[m];
		}
	}

	return arr;
}

Array<float> HostMath::reshape(Array<float> arr, Shape shape) {
	FuncTimer func_timer("reshape(float)");
	Shape a_shape = arr.shape();
	shape = shape.fix_unknown(arr.total_size());

	assert(a_shape.total_size() == shape.total_size());

	Array<float> clone = arr.data_share_clone();

	clone.m_core->m_dimension = Dim(shape);

	return clone;
}

Array<int64> HostMath::reshape(Array<int64> arr, Shape shape) {
	FuncTimer func_timer("reshape(int64)");
	Shape a_shape = arr.shape();
	shape = shape.fix_unknown(arr.total_size());

	assert(a_shape.total_size() == shape.total_size());

	Array<int64> clone = arr.data_share_clone();

	clone.m_core->m_dimension = Dim(shape);

	return clone;
}

Array<unsigned char> HostMath::reshape(Array<unsigned char> arr, Shape shape) {
	FuncTimer func_timer("reshape(uchar)");
	Shape a_shape = arr.shape();
	shape = shape.fix_unknown(arr.total_size());

	assert(a_shape.total_size() == shape.total_size());

	Array<unsigned char> clone = arr.data_share_clone();

	clone.m_core->m_dimension = Dim(shape);

	return clone;
}

Array<float> HostMath::tile(Array<float> arr, int64 num) {
	FuncTimer func_timer("tile");
	Shape ashape = arr.shape();
	Shape tshape = ashape.append(num);

	Array<float> tiled(tshape);

	float* ap = arr.data_ptr();
	float* tp = tiled.data_ptr();

	int64 size = arr.total_size();

	for (int64 n = 0; n < size; n++, ap++) {
		for (int64 m = 0; m < num; m++, tp++) {
			*tp = *ap;
		}
	}

	return tiled;
}

Array<float> HostMath::untile(Array<float> arr, int64 num) {
	FuncTimer func_timer("untile");
	Shape ashape = arr.shape();
	Shape mshape = ashape;

	assert(arr.axis_size(-1) % num == 0);
	mshape[-1] /= num;

	Array<float> merged = hmath.zeros(mshape);

	float* ap = arr.data_ptr();
	float* mp = merged.data_ptr();

	int64 size = merged.total_size();

	for (int64 n = 0; n < size; n++, mp++) {
		*mp = 0;
		for (int64 m = 0; m < num; m++, ap++) {
			*mp += *ap;
		}
	}

	return merged;
}

Array<float> HostMath::flatten(Array<float> arr) {
	FuncTimer func_timer("flatten(float)");
	return reshape(arr, Shape(-1));
}

Array<int64> HostMath::flatten(Array<int64> arr) {
	FuncTimer func_timer("flatten(array)");
	return reshape(arr, Shape(-1));
}

Array<float> HostMath::conv(Array<float> arr, Array<float> kernel, Array<float> bias) {
	FuncTimer func_timer("conv_no_cuda");

	int64 ychn = kernel.axis_size(-1);
	int64 kh = kernel.axis_size(0), kw = kernel.axis_size(1);;

	Array<float> regs = m_get_ext_regions(arr, kh, kw, 0);
	regs = regs.transpose(Idx(2, 0, 1, 3, 4, 5));
	Array<float> x_flat = regs.reshape(Shape(-1, kernel.total_size() / ychn));
	Array<float> k_flat = kernel.reshape(Shape(-1, ychn));
	Array<float> conv_flat = hmath.matmul(x_flat, k_flat);
	Array<float> conv = conv_flat.reshape(arr.shape().replace_nth(-1, ychn)) + bias;

	return conv;
}

Array<float> HostMath::conv_backprop(Array<float> G_y, Array<float> x, Array<float> kernel, Array<float>& G_k, Array<float>& G_b) {
	FuncTimer func_timer("conv_backprop_no_cuda");

	int64 xh = x.axis_size(1), xw = x.axis_size(2);
	int64 kh = kernel.axis_size(0), kw = kernel.axis_size(1);
	int64 xchn = kernel.axis_size(2), ychn = kernel.axis_size(3);

	Array<float> regs = m_get_ext_regions(x, kh, kw, 0);
	regs = regs.transpose(Idx(2, 0, 1, 3, 4, 5));
	Array<float> x_flat = regs.reshape(Shape(-1, kernel.total_size() / ychn));
	Array<float> k_flat = kernel.reshape(Shape(-1, ychn));

	Array<float> G_conv_flat = G_y.reshape(Shape(-1, ychn));
	Array<float> g_conv_k_flat = x_flat.transpose();
	Array<float> g_conv_x_flat = k_flat.transpose();
	Array<float> G_k_flat = hmath.matmul(g_conv_k_flat, G_conv_flat);
	Array<float> G_x_flat = hmath.matmul(G_conv_flat, g_conv_x_flat);

	G_b = hmath.sum(G_conv_flat, -1);
	G_k = G_k_flat.reshape(Shape(kh, kw, xchn, ychn));

	Array<float> G_regs = G_x_flat.reshape(Shape(-1, xh, xw, kh, kw, xchn));
	Array<float> G_regs2 = G_regs.transpose(Idx(1, 2, 0, 3, 4, 5));
	Array<float> G_x = m_undo_ext_regions(G_regs2, x, kernel);

	return G_x;
}

Array<float> HostMath::m_get_ext_regions(Array<float> x, int64 kh, int64 kw, float fill) {
	Shape xss = x.shape();
	int64 xh = xss[-3], xw = xss[-2], xchn = xss[-1];
	int64 mb_size = x.total_size() / (xh * xw * xchn);
	//Shape xs(xh, xw);

	int64 eh = xh + kh - 1, ew = xw + kw - 1;
	int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

	Array<float> x_reshape = x.reshape(Shape(mb_size, xh, xw, xchn));
	Array<float> x_ext = hmath.zeros(Shape(mb_size, eh, ew, xchn));
	
	if (fill != 0) x_ext += fill;

	x_ext[Axis(_all_, Ax(bh, bh + xh), Ax(bw, bw + xw), _all_)] = x_reshape;

	//m_aux["x_shape"] = xs;
	//m_aux["ext_shape"] = x_ext.shape();

	Array<float> regs = hmath.zeros(Shape(xh, xw, mb_size * kh * kw * xchn));

	for (int64 r = 0; r < xh; r++) {
		for (int64 c = 0; c < xh; c++) {
			Array<float> part;
			part = x_ext[Axis(_all_, Ax(r, r + kh), Ax(c, c + kw), _all_)];
			regs[Axis(r, c, _all_)] = part.reshape(Shape(1, 1, -1));
		}
	}

	return regs.reshape(Shape(xh, xw, mb_size, kh, kw, xchn));
}

Array<float> HostMath::m_undo_ext_regions(Array<float> G_regs, Array<float> x, Array<float> kernel) {
	Shape xss = x.shape(), kss = kernel.shape();
	int64 mb_size = xss[0], xh = xss[1], xw = xss[2], xchn = xss[3];
	int64 kh = kss[0], kw = kss[1];

	int64 eh = xh + kh - 1, ew = xw + kw - 1;
	int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

	G_regs = G_regs.reshape(Shape(xh, xw, -1));

	Array<float> G_ext = hmath.zeros(Shape(mb_size, eh, ew, xchn));

	for (int64 r = 0; r < xh; r++) {
		for (int64 c = 0; c < xw; c++) {
			Array<float> part;
			part = G_regs[Axis(r, c, _all_)];
			part = part.reshape(Shape(mb_size, kh, kw, xchn));
			G_ext[Axis(_all_, Ax(r, r + kh), Ax(c, c + kw), _all_)] += part;
		}
	}

	G_regs = G_ext[Axis(_all_, Ax(bh, bh + xh), Ax(bw, bw + xw), _all_)];

	return G_regs;
}

Array<float> HostMath::expand_images(Array<float> x, Shape stride) {
	assert(x.dim() == 4);
	assert(stride.size() == 2);

	Shape xs = x.shape();

	x = x.transpose(Idx(0, 3, 1, 2)).reshape(Shape(-1));

	int64 xsize = x.total_size();
	int64 ssize = stride[0] * stride[1];

	Array<float> y(x.shape().add(ssize));

	float* xp = x.data_ptr();
	float* yp = y.data_ptr();

	for (int64 n = 0; n < xsize; n++, xp++) {
		for (int64 m = 0; m < ssize; m++) {
			*yp++ = *xp;
		}
	}

	y = y.reshape(Shape(xs[0], xs[3], xs[1], xs[2], stride[0], stride[1]));
	y = y.transpose(Idx(0, 2, 4, 3, 5, 1));
	y = y.reshape(Shape(xs[0], xs[1] * stride[0], xs[2] * stride[1], xs[3]));

	return y;
}

Array<float> HostMath::expand_undo_images(Array<float> x, Shape stride) {
	assert(x.dim() == 4);
	assert(stride.size() == 2);

	Shape xs = x.shape();
	Shape ys(xs[0], xs[1] / stride[0], xs[2] / stride[1], xs[3]);

	assert(xs[1] % stride[0] == 0);
	assert(xs[2] % stride[1] == 0);

	Array<float> y = Array<float>::zeros(Shape(ys[0], ys[3], ys[1], ys[2]));

	x = x.reshape(Shape(xs[0], ys[1], stride[0], ys[2], stride[1], xs[3]));
	x = x.transpose(Idx(0, 5, 1, 3, 2, 4));

	int64 ysize = y.total_size();
	int64 ssize = stride[0] * stride[1];

	float* xp = x.data_ptr();
	float* yp = y.data_ptr();

	for (int64 n = 0; n < ysize; n++, yp++) {
		for (int64 m = 0; m < ssize; m++) {
			*yp += *xp++;
		}
	}

	y.transpose(Idx(0, 2, 3, 1));

	return y;
}

Array<float> HostMath::transpose(Array<float> arr) {
	FuncTimer func_timer("transpose1_no_cuda");
	assert(arr.dim() == 2);

	Shape a_shape = arr.shape();
	Array<float> trans(Shape(a_shape[1], a_shape[0]));

	float* bp_base = trans.data_ptr();
	float* bp_end = bp_base + a_shape.total_size();
		
	float* bp = bp_base++;
	float* ap = arr.data_ptr();

	int64 a_size = a_shape.total_size();
	int64 rows = a_shape[0];

	for (int64 n = 0; n < a_size; n++, ap++) {
		*bp = *ap;
		bp += rows;
		if (bp >= bp_end) bp = bp_base++;
	}

	return trans;
}

Array<float> HostMath::transpose(Array<float> src, Idx idx) {
	FuncTimer func_timer("transpose2_no_cuda");
	Shape sshape = src.shape();
	Shape dshape = src.shape();

	int64 dim = idx.size();
	int64 dat_size = src.total_size();

	assert(sshape.size() == dim);

	int64 coord[KAI_MAX_DIM], dprod[KAI_MAX_DIM + 1], padd[KAI_MAX_DIM];
	int64 prod = 1;

	for (int64 n = dim - 1; n >= 0; n--) {
		coord[n] = 0;
		dprod[n] = prod;
		prod *= dshape[n] = sshape[idx[n]];
	}

	dprod[dim] = 0;

	for (int64 n = 0; n < dim; n++) {
		padd[idx[n]] = dprod[n];
	}

	for (int64 n = 0; n < dim; n++) {
		if (idx[n] == 0) continue;
		padd[idx[n] - 1] -= dprod[n] * dshape[n];
	}

	//Array<float> dst(dshape);
	Array<float> dst = hmath.zeros(dshape);

	float* sp = src.data_ptr();
	float* dp = dst.data_ptr();

	int64 offset = 0;

	for (int64 i = 0; i < dat_size; i++, sp++) {
		*(dp + offset) = *sp;
		for (int64 n = dim - 1; n >= 0; n--) {
			coord[n]++;
			offset += padd[n];
			if (coord[n] < sshape[n]) break;
			coord[n] = 0;
		}
	}

	return dst;
}

Array<unsigned char> HostMath::transpose(Array<unsigned char> src, Idx idx) {
	FuncTimer func_timer("transpose2_no_cuda");
	Shape sshape = src.shape();
	Shape dshape = src.shape();

	int64 dim = idx.size();
	int64 dat_size = src.total_size();

	assert(sshape.size() == dim);

	int64 coord[KAI_MAX_DIM], dprod[KAI_MAX_DIM + 1], padd[KAI_MAX_DIM];
	int64 prod = 1;

	for (int64 n = dim - 1; n >= 0; n--) {
		coord[n] = 0;
		dprod[n] = prod;
		prod *= dshape[n] = sshape[idx[n]];
	}

	dprod[dim] = 0;

	for (int64 n = 0; n < dim; n++) {
		padd[idx[n]] = dprod[n];
	}

	for (int64 n = 0; n < dim; n++) {
		if (idx[n] == 0) continue;
		padd[idx[n] - 1] -= dprod[n] * dshape[n];
	}

	//Array<float> dst(dshape);
	Array<unsigned char> dst(dshape);

	unsigned char* sp = src.data_ptr();
	unsigned char* dp = dst.data_ptr();

	int64 offset = 0;

	for (int64 i = 0; i < dat_size; i++, sp++) {
		*(dp + offset) = *sp;
		for (int64 n = dim - 1; n >= 0; n--) {
			coord[n]++;
			offset += padd[n];
			if (coord[n] < sshape[n]) break;
			coord[n] = 0;
		}
	}

	return dst;
}

Array<float> HostMath::wvec_select(Array<float> arr, Array<int64> idxs) {
	FuncTimer func_timer("wvec_select");
	assert(arr.dim() == 2);

	int64 vec_size = arr.axis_size(1);

	Array<float> selected(idxs.shape().append(vec_size));

	float* ap_base = arr.data_ptr();
	float* sp = selected.data_ptr();
	int64* kmath = idxs.data_ptr();

	int64 ncol = vec_size, nrow = selected.total_size() / ncol;

	for (int64 n = 0; n < nrow; n++, kmath++, sp += ncol) {
		float* ap = ap_base + *kmath * ncol;
		memcpy(sp, ap, sizeof(float)*ncol);
	}

	return selected;
}

Array<float> HostMath::wvec_select_idx(Array<float> arr, Array<int64> idxs, int64 dic_count, int64* voc_counts) {
	FuncTimer func_timer("wvec_select_idx");
	assert(arr.dim() == 2);

	int64 vec_size = arr.axis_size(1);

	Shape shape = idxs.shape().replace_nth(-1, vec_size);

	Array<float> selected = hmath.zeros(shape);
	
	float* ap_base = arr.data_ptr();
	float* sp = selected.data_ptr();
	int64* np = idxs.data_ptr();

	int64 ncol = vec_size, nrow = selected.total_size() / ncol;

	for (int64 n = 0; n < nrow; n++, sp += ncol) {
		float* ap1 = ap_base;
		for (int64 nd = 0; nd < dic_count; nd++, np++) {
			int64 nid = (int64) *np;
			float* ap = ap1 + nid * ncol;
			if (nid < 0 || nid >= voc_counts[nd]) {
				logger.Print("HostMath::wvec_select_idx bad idx: n = %lld, nd = %lld, nid = %lld, voc_counts[%lld] = %lld", n, nd, nid, nd, voc_counts[nd]);
				throw KaiException(KERR_ASSERT);
			}
			for (int64 k = 0; k < ncol; k++) {
				sp[k] += ap[k];
			}
			ap1 += voc_counts[nd] * ncol;
		}
	}

	return selected;
}

Array<float> HostMath::select_col(Array<float> arr, Array<int64> idxs) {
	FuncTimer func_timer("select_col");

	/*
	if (idxs.dim() != 1) {
		idxs = idxs.reshape(Shape(-1));
	}
	*/

	assert(idxs.dim() == 1);
	assert(arr.axis_size(0) == idxs.axis_size(0));

	if (arr.dim() == 2) {
		int64 nrow = arr.axis_size(0), ncol = arr.axis_size(1);

		Shape shape(nrow);
		Array<float> selected(shape);

		float* ap = arr.data_ptr();
		float* sp = selected.data_ptr();
		int64* kmath = idxs.data_ptr();

		for (int64 n = 0; n < nrow; n++, ap += ncol, sp++, kmath++) {
			*sp = ap[*kmath];
		}

		return selected;
	}
	else if (arr.dim() == 3) {
		int64 nrow = arr.axis_size(0), ncol = arr.axis_size(1), nvec = arr.axis_size(2);

		Shape shape(nrow, nvec);
		Array<float> selected(shape);

		float* ap = arr.data_ptr();
		float* sp = selected.data_ptr();
		int64* kmath = idxs.data_ptr();

		for (int64 n = 0; n < nrow; n++, ap += ncol*nvec) {
			float* pp = ap + *kmath++ * nvec;
			for (int64 m = 0; m < nvec; m++) {
				*sp++ = *pp++;
			}
		}

		return selected;
	}
	else {
		throw KaiException(KERR_ASSERT);
		return arr;
	}
}

Array<float> HostMath::select_rnn_last_col(Array<float> arr) {
	assert(arr.dim() == 3);

	int64 nrow = arr.axis_size(0), ncol = arr.axis_size(1), nvec = arr.axis_size(2);

	Shape shape(nrow, nvec);
	Array<float> selected(shape);

	float* ap = arr.data_ptr();
	float* sp = selected.data_ptr();

	for (int64 n = 0; n < nrow; n++, ap += ncol * nvec) {
		float* pp = ap + (ncol - 1) * nvec;
		for (int64 m = 0; m < nvec; m++) {
			*sp++ = *pp++;
		}
	}

	return selected;
}

template <class T>
Array<T> HostMath::mult_dim(Array<T> arr, int64 axis) {
	FuncTimer func_timer("mult_dim");
	assert(axis == -1);

	Shape ashape = arr.shape();

	Array<T> mult = hmath.ones(ashape.remove_end());

	T* ap = arr.data_ptr();
	T* mp = mult.data_ptr();

	int64 ncol = arr.axis_size(-1);
	int64 size = mult.total_size();

	for (int64 n = 0; n < size; n++, mp++) {
		for (int64 m = 0; m < ncol; m++, ap++) {
			*mp *= *ap;
		}
	}

	return mult;
}

template <class T>
Array<T> HostMath::filter(Array<T> arr, int64 axis, int64 index) {
	FuncTimer func_timer("filter");
	Shape ashape = arr.shape();

	assert(axis >= 0 && axis < ashape.size());
	assert(index >= 0 && index < ashape[axis]);

	int64 nprod = 1;

	for (int64 n = axis + 1; n < ashape.size(); n++) nprod *= ashape[n];

	int64 nstart = index * nprod, nskip = ashape[axis] * nprod;

	Array<T> selected(ashape.remove_nth(axis));

	T* ap = arr.data_ptr() + index * nstart;
	T* sp = selected.data_ptr();

	int64 nrow = arr.total_size() / nskip;

	for (int64 n = 0; n < nrow; n++, ap += nskip, sp += nprod) {
		memcpy(sp, ap, sizeof(T) * nprod);
	}

	return selected;
}

template <class T>
Array<T> HostMath::unfilter(Array<T> arr, Array<T> filtered, int64 axis, int64 index) {
	FuncTimer func_timer("unfilter");
	Shape ashape = arr.shape();
	Shape fshape = filtered.shape();

	assert(axis >= 0 && axis < ashape.size());
	assert(index >= 0 && index < ashape[axis]);
	assert(ashape.remove_nth(axis) == fshape);

	int64 nprod = 1;

	for (int64 n = axis + 1; n < ashape.size(); n++) nprod *= ashape[n];

	int64 nstart = index * nprod, nskip = ashape[axis] * nprod;

	Array<T> clone = arr.deepcopy();

	T* ap = clone.data_ptr() + index * nstart;
	T* sp = filtered.data_ptr();

	int64 nrow = clone.total_size() / nskip;

	for (int64 n = 0; n < nrow; n++, ap += nskip, sp += nprod) {
		memcpy(ap, sp, sizeof(T) * nprod);
	}

	return clone;
}

Array<float> HostMath::minus_1_on_idx(Array<float> arr, Array<int64> idxs) {
	FuncTimer func_timer("minus_1_on_idx");
	assert(arr.dim() == 2);
	assert(idxs.dim() == 1);
	assert(arr.axis_size(0) == idxs.axis_size(0));

	int64 nrow = arr.axis_size(0), ncol = arr.axis_size(1);

	Array<float> touched(arr.shape());

	float* ap = arr.data_ptr();
	float* tp = touched.data_ptr();
	int64* np = idxs.data_ptr();

	memcpy(tp, ap, sizeof(float) * arr.total_size());

	for (int64 n = 0; n < nrow; n++, tp += ncol, np++) {
		tp[*np] -= 1;
	}

	return touched;
}

Array<float> HostMath::minus_1_on_1st(Array<float> arr) {
	FuncTimer func_timer("minus_1_on_1st");
	assert(arr.dim() == 2);

	int64 nrow = arr.axis_size(0), ncol = arr.axis_size(1);

	Array<float> touched(arr.shape());

	float* ap = arr.data_ptr();
	float* tp = touched.data_ptr();

	memcpy(tp, ap, sizeof(float) * arr.total_size());

	for (int64 n = 0; n < nrow; n++, tp += ncol) {
		tp[0] -= 1;
	}

	return touched;
}

Array<int> HostMath::to_int(Array<float> arr) {
	FuncTimer func_timer("to_int");
	Array<int> clone(arr.shape());

	float* ap = arr.data_ptr();
	int* bp = clone.data_ptr();

	int64 size = arr.total_size();

	for (int64 n = 0; n < size; n++) {
		*bp++ =  (int) *ap++;
	}

	return clone;
}

Array<int64> HostMath::to_int64(Array<float> arr) {
	FuncTimer func_timer("to_int");
	Array<int64> clone(arr.shape());

	float* ap = arr.data_ptr();
	int64* bp = clone.data_ptr();

	int64 size = arr.total_size();

	for (int64 n = 0; n < size; n++) {
		*bp++ =  (int64) *ap++;
	}

	return clone;
}

void HostMath::acc_abs_n_sq_num(Array<float> params, float& abs_sum, float& sq_sum) {
	abs_sum += params.abs().sum();
	sq_sum += params.square().sum();
}

void HostMath::acc_nearzero_sum_sqnum(Array<float> params, float& sum, float& sq_sum, int64& near_zero_cnt, float threshold) {
	near_zero_cnt += params.count_in_range(-threshold, threshold);
	sum += params.sum();
	sq_sum += params.square().sum();
}

int64* HostMath::alloc_int64_array(int64* host_p, int64 count) {
	return host_p;
}

void HostMath::free_int64_array(int64* cuda_p) {
}

Array<float> HostMath::power(Array<float> arr, float exp) {
	FuncTimer func_timer("power");
	Array<float> result(arr.shape());

	float* ap = arr.data_ptr();
	float* bp = result.data_ptr();

	int64 size = arr.total_size();

	for (int64 n = 0; n < size; n++) {
		*bp++ = ::powf(*ap++, exp);
	}

	return result;
}

Array<float> HostMath::mask_future_timesteps(Array<float> arr, int64 timesteps) {
	FuncTimer func_timer("mask_future_timesteps");
	
	Shape shape = arr.shape();

	assert(shape.size() >= 2);
	assert(shape[-2] == timesteps);
	assert(shape[-1] == timesteps);

	Array<float> masked = arr.deepcopy();

	float* mp = masked.data_ptr();

	int64 size = arr.total_size();

	for (int64 n = 0; n < size; n++, mp++) {
		int64 t1 = (n / timesteps) % timesteps;
		int64 t2 = n % timesteps;
		if (t2 > t1) {
			*mp -= 10000.0f;
		}
	}

	return masked;
}

Array<float> HostMath::vstack(Array<float> arr1, Array<float> arr2) {
	FuncTimer func_timer("vstack");
	Shape shape1 = arr1.shape(), shape2 = arr2.shape();
	int64 dim1 = shape1.size();

	assert(dim1 == shape2.size());

	for (int64 n = 1; n < dim1; n++) {
		assert(shape1[n] == shape2[n]);
	}

	Shape shape = shape1;
	shape[0] = shape1[0] + shape2[0];

	Array<float> stack(shape);

	float* ap = arr1.data_ptr();
	float* bp = arr2.data_ptr();
	float* sp1 = stack.data_ptr();
	float* sp2 = sp1 + arr1.total_size();

	memcpy(sp1, ap, sizeof(float) * arr1.total_size());
	memcpy(sp2, bp, sizeof(float) * arr2.total_size());

	return stack;
}

Array<float> HostMath::vstack(vector<Array<float>> arrs) {
	FuncTimer func_timer("vstack2");

	int64 arr_cnt = arrs.size();

	if (arr_cnt == 0) return Array<float>();

	Shape ashape = arrs[0].shape().remove_front();

	int64 asize = ashape.total_size();
	int64 dim = 0;

	for (int64 n = 0; n < arr_cnt; n++) {
		int64 dim_n = arrs[n].axis_size(0);
		if (arrs[n].total_size() != asize * dim_n) throw KaiException(KERR_ASSERT);
		dim += dim_n;
	}

	Shape vshape = ashape.add_front(dim);

	Array<float> stack(vshape);

	float* vp = stack.data_ptr();

	for (int64 n = 0; n < arr_cnt; n++) {
		int64 arr_size = arrs[n].total_size();
		float* ap = arrs[n].data_ptr();
		memcpy(vp, ap, sizeof(float) * arr_size);
		vp += arr_size;
	}

	return stack;
}

/*
Array<short> HostMath::vstack(vector<Array<short>> arrs) {
	FuncTimer func_timer("vstack2");

	int64 dim1 = arrs.size();

	if (dim1 == 0) return Array<short>();

	Shape ashape = arrs[0].shape();
	Shape vshape = ashape.add_front(dim1);

	int64 asize = arrs[0].total_size();

	Array<short> stack(vshape);

	short* vp = stack.data_ptr();

	for (int64 n = 0; n < dim1; n++) {
		if (arrs[n].total_size() != asize) throw KaiException(KERR_ASSERT);
		short* ap = arrs[n].data_ptr();
		memcpy(vp, ap, sizeof(short) * asize);
		vp += asize;
	}

	return stack;
}
*/

Array<float> HostMath::hstack(Array<float> arr1, Array<float> arr2) {
	FuncTimer func_timer("hstack1");
	Shape shape = arr1.shape();
	int64 dim = shape.size();

	assert(dim == arr2.dim());

	for (int64 n = 0; n < dim - 1; n++) {
		assert(shape[n] == arr2.dimension()[n]);
	}

	int64 acol = shape[dim - 1];
	int64 bcol = arr2.dimension()[dim - 1];
		
	shape[dim - 1] = acol + bcol;

	Array<float> stack(shape);
		
	float* ap = arr1.data_ptr();
	float* bp = arr2.data_ptr();
	float* sp = stack.data_ptr();

	int64 s_size = shape.total_size();
	int64 rows = s_size / shape[dim - 1];

	for (int64 n = 0; n < rows; n++) {
		for (int64 m = 0; m < acol; m++) {
			*sp++ = *ap++;
		}
		for (int64 m = 0; m < bcol; m++) {
			*sp++ = *bp++;
		}
	}

	return stack;
}

Array<float> HostMath::hstack(vector<Array<float>> arrs) {
	FuncTimer func_timer("hstack2");
	Shape shape = arrs[0].shape();
	shape[-1] = 0;

	int64 total_chn = 0, arr_cnt =  arrs.size();
	int64* bchns = new int64[arr_cnt];
	float** bpp = new float* [arr_cnt];

	for (int64 n = 0; n < arr_cnt; n++) {
		Shape bshape = arrs[n].shape();
		total_chn += bchns[n] = bshape[-1];
		bshape[-1] = 0;
		assert(shape == bshape);
		bpp[n] = arrs[n].data_ptr();
	}

	shape[-1] = total_chn;

	Array<float> stack(shape);

	float* sp = stack.data_ptr();

	int64 s_size = shape.total_size();
	int64 rows = s_size / total_chn;

	for (int64 n = 0; n < rows; n++) {
		for (int64 m = 0; m < arr_cnt; m++) {
			for (int64 k = 0; k < bchns[m]; k++) {
				*sp++ = *bpp[m]++;
			}
		}
	}

	delete [] bchns;
	delete [] bpp;

	return stack;
}

void HostMath::hsplit(Array<float> arr, int64 p1_size, Array<float>& piece1, Array<float>& piece2) {
	vector<int64> bchns;
	bchns.push_back(p1_size);
	bchns.push_back(arr.axis_size(-1)- p1_size);

	vector<Array<float>> pieces = hsplit(arr, bchns);

	piece1 = pieces[0];
	piece2 = pieces[1];
}

vector<Array<float>> HostMath::hsplit(Array<float> arr, vector<int64> bchns) {
	FuncTimer func_timer("hsplit1");
	Shape bshape = arr.shape();

	int64 total_chn = 0, arr_cnt = bchns.size();
	float** bpp = new float* [arr_cnt];

	vector<Array<float>> result;

	for (int64 n = 0; n < arr_cnt; n++) {
		total_chn += bshape[-1] = bchns[n];
		Array<float> branch(bshape);
		result.push_back(branch);
		bpp[n] = branch.data_ptr();
	}

	assert(arr.shape()[-1] == total_chn);

	float* sp = arr.data_ptr();

	int64 s_size = arr.total_size();
	int64 rows = s_size / total_chn;

	for (int64 n = 0; n < rows; n++) {
		for (int64 m = 0; m < arr_cnt; m++) {
			for (int64 k = 0; k < bchns[m]; k++) {
				*bpp[m]++ = *sp++;
			}
		}
	}

	delete[] bpp;

	return result;
}

template <class T>
vector<Array<T>> HostMath::hsplit_last(Array<T> arr) {
	FuncTimer func_timer("hsplit2");
	int64 chn_cnt = arr.axis_size(-1);

	vector<Array<T>> result;

	Shape bshape = arr.shape().remove_end();
	T** bpp = new T * [chn_cnt];

	for (int64 n = 0; n < chn_cnt; n++) {
		Array<T> branch(bshape);
		result.push_back(branch);
		bpp[n] = branch.data_ptr();
	}

	T* sp = arr.data_ptr();

	int64 rows = bshape.total_size();

	for (int64 n = 0; n < rows; n++) {
		for (int64 m = 0; m < chn_cnt; m++) {
			*bpp[m]++ = *sp++;
		}
	}

	delete[] bpp;

	return result;
}

int64 HostMath::fread_int_msb(FILE* fid) {
	FuncTimer func_timer("fread_int_msb");
	int64 num_msb, num_lsb = 0;
	if (fread(&num_msb, sizeof(int), 1, fid) != 1) {
		throw KaiException(KERR_ASSERT);
	}
	num_lsb |= (num_msb & 0x000000FF) << 24;
	num_lsb |= (num_msb & 0x0000FF00) << 8;
	num_lsb |= (num_msb & 0x00FF0000) >> 8;
	num_lsb |= (num_msb & 0xFF000000) >> 24;

	return num_lsb;
}

void HostMath::load_from_file(string filepath, Array<unsigned char>& arr) {
	FuncTimer func_timer("load_from_file");
	FILE* fid = Util::fopen(filepath.c_str(), "rb");
	assert(fid != NULL);
	int64 magic_num;
	magic_num = fread_int_msb(fid);
	if (magic_num == 2049) { // 2049 MSB first
		int64 component_num = fread_int_msb(fid);
		arr = Array<unsigned char>(Shape(component_num));
		unsigned char* p = arr.data_ptr();
		size_t size = arr.total_size();
		if (fread(p, 1, size, fid) != size) {
			throw KaiException(KERR_ASSERT);
		}
	}
	else if (magic_num == 2051) { // 2051 MSB first
		int64 component_num = fread_int_msb(fid);
		int64 row_num = fread_int_msb(fid);
		int64 col_num = fread_int_msb(fid);
		arr = Array<unsigned char>(Shape(component_num, row_num, col_num));
		unsigned char* p = arr.data_ptr();
		size_t size = arr.total_size();
		if (fread(p, 1, size, fid) != size) {
			throw KaiException(KERR_ASSERT);
		}
	}
	fclose(fid);
}

Array<int64> HostMath::get_hash_idx(Array<float> code) {
	throw KaiException(KERR_ASSERT);
	Array<int64> binary = code.round(); // 기준이 0.5에서 0으로 변경되었으므로 수정 필요함, 일단 쿠다에서 처리
	Array<int64> hash_idx = binary.binary_row_to_int();
	return hash_idx;
}

Array<int64> HostMath::eval_hash_diff(Array<int64>m_hash, Array<int64>key_hash_cuda, int64 nvec) {
	throw KaiException(KERR_ASSERT);	// 일단 쿠다에서 처리
	return Array<int64>();
}

Array<float> HostMath::fft(Array<float> arr) {
	throw KaiException(KERR_ASSERT);
	return Array<float>();
}

Array<float> HostMath::fft(Array<short> arr) {
	throw KaiException(KERR_ASSERT);
	return Array<float>();
}

template void HostMath::bin_op(Array<float> a, Array<float> b, float(*op)(float a, float b));
template void HostMath::bin_op(Array<float> a, float b, float(*op)(float a, float b));

template void HostMath::bin_op(Array<int> a, int b, int(*op)(int a, int b));
template void HostMath::bin_op(Array<int64> a, int64 b, int64(*op)(int64 a, int64 b));

template void HostMath::log_bin_op(Array<bool> result, Array<float> a, Array<float> b, bool(*op)(float a, float b));
template void HostMath::log_bin_op(Array<bool> result, Array<int> a, Array<int> b, bool(*op)(int a, int b));
template void HostMath::log_bin_op(Array<bool> result, Array<int64> a, Array<int64> b, bool(*op)(int64 a, int64 b));
template void HostMath::log_bin_op(Array<bool> result, Array<bool> a, Array<bool> b, bool(*op)(bool a, bool b));

template void HostMath::log_bin_op(Array<bool> result, Array<float> a, float b, bool(*op)(float a, float b));
template void HostMath::log_bin_op(Array<bool> result, Array<int> a, int b, bool(*op)(int a, int b));
template void HostMath::log_bin_op(Array<bool> result, Array<int64> a, int64 b, bool(*op)(int64 a, int64 b));
template void HostMath::log_bin_op(Array<bool> result, Array<unsigned char> a, unsigned char b, bool(*op)(unsigned char a, unsigned char b));

template Array<int64> HostMath::copy_array(Array<int64> src, Axis src_axis);
template Array<float> HostMath::copy_array(Array<float> src, Axis src_axis);
template Array<unsigned char> HostMath::copy_array(Array<unsigned char> src, Axis src_axis);

template void HostMath::m_array_copy(int64* pSrc, int64* pDst, ArrayLoopAxisInfo<int64>* src_info, ArrayLoopAxisInfo<int64>* dst_info, int64 nth, int64 dim);
template void HostMath::m_array_copy(float* pSrc, float* pDst, ArrayLoopAxisInfo<float>* src_info, ArrayLoopAxisInfo<float>* dst_info, int64 nth, int64 dim);
template void HostMath::m_array_copy(unsigned char* pSrc, unsigned char* pDst, ArrayLoopAxisInfo<unsigned char>* src_info, ArrayLoopAxisInfo<unsigned char>* dst_info, int64 nth, int64 dim);

template Array<int64> HostMath::extract_selected(Array<int64> arr, Array<bool> selector);
template Array<float> HostMath::extract_selected(Array<float> arr, Array<bool> selector);
template Array<unsigned char> HostMath::extract_selected(Array<unsigned char> arr, Array<bool> selector);

template Array<int64> HostMath::fill_selected(Array<int64> arr, Array<int64> slices, Array<bool> selector);
template Array<float> HostMath::fill_selected(Array<float> arr, Array<float> slices, Array<bool> selector);
template Array<unsigned char> HostMath::fill_selected(Array<unsigned char> arr, Array<unsigned char> slices, Array<bool> selector);

template Array<float> HostMath::mult_dim(Array<float> arr, int64 axis);

template Array<int64> HostMath::filter(Array<int64> arr, int64 axis, int64 index);
template Array<float> HostMath::filter(Array<float> arr, int64 axis, int64 index);

template Array<int64> HostMath::unfilter(Array<int64> arr, Array<int64> filtered, int64 axis, int64 index);
template Array<float> HostMath::unfilter(Array<float> arr, Array<float> filtered, int64 axis, int64 index);

template vector<Array<int64>> HostMath::hsplit_last(Array<int64> arr);
template vector<Array<float>> HostMath::hsplit_last(Array<float> arr);
