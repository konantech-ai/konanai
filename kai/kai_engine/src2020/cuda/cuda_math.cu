/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "cuda_math.h"
#include "cuda_conn.cuh"
#include "cuda_kernels.h"

#include "../core/value.h"
#include "../core/array.h"
#include "../core/host_math.h"
#include "../core/util.h"
#include "../core/func_timer.h"
#include "../core/log.h"

#include <float.h>
#include <stdio.h>

#ifdef KAI2021_WINDOWS
#else
#include <unistd.h>
#endif

#include <chrono>
#include <thread>

using namespace std;

#define USE_HOST_RAND

CudaMath cmath;

Array<float> CudaMath::random_uniform(Shape shape, CudaConn* pConn) {
	Array<float> arr = pConn->create_farray(shape, "CudaMath::random_uniform");

	float* cuda_p = arr.data_ptr();

#ifndef USE_HOST_RAND
	CudaConn::random_uniform(cuda_p, shape.total_size());
#else
	Array<float> rnd = hmath.random_uniform(shape);
	cudaMemcpy(cuda_p, rnd.data_ptr(), shape.total_size() * sizeof(float), cudaMemcpyHostToDevice);
#endif

	return arr;
}

Array<float> CudaMath::random_bernoulli(Shape shape, float prob_threshod, CudaConn* pConn) {
	Array<float> arr = pConn->create_farray(shape, "CudaMath::random_binomial");

	float* cuda_p = arr.data_ptr();

#ifndef USE_HOST_RAND
	CudaConn::random_bernoulli(cuda_p, prob_threshod, shape.total_size());
#else
	Array<float> rnd = hmath.random_bernoulli(shape, prob_threshod);
	cudaMemcpy(cuda_p, rnd.data_ptr(), shape.total_size() * sizeof(float), cudaMemcpyHostToDevice);
#endif

	return arr;
}

Array<float> CudaMath::random_normal(float mean, float std, Shape shape, CudaConn* pConn) {
	Array<float> arr = pConn->create_farray(shape, "CudaMath::random_normal");

	float* cuda_p = arr.data_ptr();

#ifndef USE_HOST_RAND
	CudaConn::random_normal(cuda_p, shape.total_size(), mean, std);
#else
	Array<float> rnd = hmath.random_normal(mean, std, shape);
	cudaMemcpy(cuda_p, rnd.data_ptr(), shape.total_size() * sizeof(float), cudaMemcpyHostToDevice);
#endif

	return arr;
}

Array<float> CudaMath::zeros(Shape shape, CudaConn* pConn) {
	Array<float> arr = pConn->create_farray(shape, "CudaMath::zeros");
	cudaMemset(arr.data_ptr(), 0, arr.total_size() * sizeof(float));
	return arr;
}

Array<float> CudaMath::ones(Shape shape, float coef, CudaConn* pConn) {
	Array<float> arr = pConn->create_farray(shape, "CudaMath::ones");
	int64 size = arr.total_size();
	cu_call(ker_init, size, (size, arr.data_ptr(), coef));
	return arr;
}

Array<int64> CudaMath::ones_int(Shape shape, int64 coef, CudaConn* pConn) {
	Array<int64> arr = pConn->create_n64array(shape, "CudaMath::ones_int");
	int64 size = arr.total_size();
	cu_call(ker_init_int64, size, (size, arr.data_ptr(), coef));
	return arr;
}

Array<float> CudaMath::sigmoid(Array<float> arr) {
	CudaConn cuda("sigmoid", NULL);

	float* cuda_x = CudaConn::GetCudaMem(arr, "sigmoid-in");
	float* cuda_y = cuda.alloc_float_mem(arr.shape(), "sigmoid-buf");

	int64 size = arr.total_size();

	cu_call(ker_sigmoid, size, (size, cuda_y, cuda_x));

	return cuda.detach(cuda_y, "sigmoid-out");
}

Array<float> CudaMath::softmax(Array<float> arr) {
	CudaConn cuda("softmax", NULL);

	float* cuda_x = CudaConn::GetCudaMem(arr, "softmax-in");
	float* cuda_y = cuda.alloc_float_mem(arr.shape(), "softmax-buf");

	int64 size = arr.total_size();
	int64 nvec = arr.axis_size(-1);

	cu_call(ker_softmax, size, (size, cuda_y, cuda_x, nvec));

	return cuda.detach(cuda_y, "softmax-out");
}

float CudaMath::square_sum(Array<float> arr) {
	CudaConn cuda("square", NULL);

	float* cuda_a = CudaConn::GetCudaMem(arr, "square_sum");
	float* cuda_y = cuda.alloc_float_mem(arr.shape(), "square_sum");

	int64 size = arr.total_size();

	cu_call(ker_sqr, size, (size, cuda_y, cuda_a));

	for (int64 range = ADD_RANGE, ssize = size; ssize > 1; range *= ADD_RANGE) {
		ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
		cu_call(ker_sum, ssize, (ssize, cuda_y, size, range));
	}

	float result = cuda.get_nth_element(cuda_y, 0);

	return result;
}

Array<float> CudaMath::mult_scalar(Array<float> arr, float other) {
	CudaConn cuda("mult_scalar", NULL);

	float* cuda_a = CudaConn::GetCudaMem(arr, "mult_scalar");
	float* cuda_y = cuda.alloc_float_mem(arr.shape(), "mult_scalar");

	int64 size = arr.total_size();

	cu_call(ker_mult_scalar_to, size, (size, cuda_y, cuda_a, other));

	return cuda.detach(cuda_y, "cuda_y");
}

Array<float> CudaMath::vstack(Array<float> arr1, Array<float> arr2) {
	CudaConn cuda("vstack", NULL);

	Shape shape1 = arr1.shape(), shape2 = arr2.shape();

	int64 size1 = shape1.total_size(), vol1 = shape1[0], rest1 = size1 / vol1;
	int64 size2 = shape2.total_size(), vol2 = shape2[0], rest2 = size2 / vol2;

	if (rest1 != rest2) {
		logger.Print("vol1 = %lld, rest1 = %lld", vol1, rest1);
		logger.Print("vol2 = %lld, rest2 = %lld", vol2, rest2);
	}

	assert(rest1 == rest2);

	Shape sshape = shape1.replace_nth(0, vol1 + vol2);

	float* cuda_a1 = CudaConn::GetCudaMem(arr1, "vstack:x1");
	float* cuda_a2 = CudaConn::GetCudaMem(arr2, "vstack:x2");
	float* cuda_vs = cuda.alloc_float_mem(sshape, "vstack:s");

	int64 ssize = sshape.total_size();

	cu_call(ker_vstack, ssize, (ssize, cuda_vs, cuda_a1, cuda_a2, vol1, vol2, rest1));

	return cuda.detach(cuda_vs, "vstack");
}

Array<float> CudaMath::vstack(vector<Array<float>> arrs) {
	CudaConn cuda("vstack2", NULL);

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

	float* cuda_vs = cuda.alloc_float_mem(vshape, "vstack:s");
	float* cuda_vp = cuda_vs;

	for (int64 n = 0; n < arr_cnt; n++) {
		int64 arr_size = arrs[n].total_size();
		float* ap = arrs[n].data_ptr();
		cudaMemcpy(cuda_vp, ap, sizeof(float) * arr_size, arrs[n].is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
		cuda_vp += arr_size;
	}

	return cuda.detach(cuda_vs, "vstack");
}

/*
Array<short> CudaMath::vstack(vector<Array<short>> arrs) {
	throw KaiException(KERR_ASSERT);
	return Array<short>();
}
*/

Array<float> CudaMath::hstack(Array<float> arr1, Array<float> arr2) {
	CudaConn cuda("hstack", NULL);

	Shape shape1 = arr1.shape(), shape2 = arr2.shape();

	int64 size1 = shape1.total_size(), vec1 = shape1[-1], rest1 = size1 / vec1;
	int64 size2 = shape2.total_size(), vec2 = shape2[-1], rest2 = size2 / vec2;

	if (rest1 != rest2) throw KaiException(KERR_ASSERT);

	Shape sshape = shape1.replace_nth(-1, vec1 + vec2);

	float* cuda_a1 = CudaConn::GetCudaMem(arr1, "hstack:x1");
	float* cuda_a2 = CudaConn::GetCudaMem(arr2, "hstack:x2");
	float* cuda_vs = cuda.alloc_float_mem(sshape, "hstack:s");

	int64 ssize = sshape.total_size();

	cu_call(ker_hstack, ssize, (ssize, cuda_vs, cuda_a1, cuda_a2, vec1, vec2));

	return cuda.detach(cuda_vs, "hstack");
}

Array<float> CudaMath::hstack(vector<Array<float>> arrs) {
	throw KaiException(KERR_ASSERT);
	return Array<float>();
}

void CudaMath::acc_abs_n_sq_num(Array<float> params, float& abs_sum, float& sqr_sum) {
	CudaConn cuda("acc_abs_n_sq_num", NULL);

	float* cuda_x = CudaConn::GetCudaMem(params, "params");
	float* cuda_a = cuda.alloc_float_mem(params.shape(), "abs-buf");
	float* cuda_s = cuda.alloc_float_mem(params.shape(), "sqr-buf");

	int64 size = params.total_size();

	cu_call(ker_abs, size, (size, cuda_a, cuda_x));
	cu_call(ker_sqr, size, (size, cuda_s, cuda_x));

	for (int64 range = ADD_RANGE, ssize = size; ssize > 1; range *= ADD_RANGE) {
		ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
		cu_call(ker_sum, ssize, (ssize, cuda_a, size, range));
		cu_call(ker_sum, ssize, (ssize, cuda_s, size, range));
	}

	abs_sum += cuda.get_nth_element(cuda_a, 0);
	sqr_sum += cuda.get_nth_element(cuda_s, 0);
}

void CudaMath::acc_nearzero_sum_sqnum(Array<float> params, float& sum, float& sq_sum, int64& near_zero_cnt, float threshold) {
	CudaConn cuda("acc_nearzero_sum_sqnum", NULL);

	float* cuda_x = CudaConn::GetCudaMem(params, "params");
	float* cuda_s = cuda.copy_to_buffer(params, "sum-buf");
	float* cuda_q = cuda.alloc_float_mem(params.shape(), "sqr-buf");
	float* cuda_z = cuda.alloc_float_mem(params.shape(), "near-zero-buf");

	int64 size = params.total_size();

	cu_call(ker_sqr, size, (size, cuda_q, cuda_x));
	cu_call(ker_near_zero, size, (size, cuda_z, cuda_x, threshold));

	for (int64 range = ADD_RANGE, ssize = size; ssize > 1; range *= ADD_RANGE) {
		ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
		cu_call(ker_sum, ssize, (ssize, cuda_s, size, range));
		cu_call(ker_sum, ssize, (ssize, cuda_q, size, range));
		cu_call(ker_sum, ssize, (ssize, cuda_z, size, range));
	}

	sum += cuda.get_nth_element(cuda_s, 0);
	sq_sum += cuda.get_nth_element(cuda_q, 0);
	near_zero_cnt += (int64) cuda.get_nth_element(cuda_z, 0);
}

void CudaMath::set_row(Array<float> arr, int64 nth, float other) {
	int64 nrow = arr.axis_size(0), nvec = arr.total_size() / nrow;

	float* cuda_p = arr.data_ptr();

	cu_call(ker_init, nvec, (nvec, cuda_p, other));
}

void CudaMath::set_row(Array<float> arr, int64 nth, Array<float> other) {
	throw KaiException(KERR_ASSERT);
}

void CudaMath::hsplit(Array<float> arr, int64 p1_size, Array<float>& piece1, Array<float>& piece2) {
	CudaConn cuda("CudaMath::hsplit", NULL);

	Shape shape1 = arr.shape();
	Shape shape2 = arr.shape();

	int64 vec_size = arr.axis_size(-1);

	shape1[-1] = p1_size;
	shape2[-1] = vec_size - p1_size;

	float* cuda_src = arr.data_ptr();
	float* cuda_p1 = cuda.alloc_float_mem(shape1, "p1");
	float* cuda_p2 = cuda.alloc_float_mem(shape2, "p2");

	int64 size = arr.total_size();

	cu_call(ker_hsplit, size, (size, cuda_p1, cuda_p2, cuda_src, vec_size, p1_size));

	piece1 = cuda.detach(cuda_p1, "piece1");
	piece2 = cuda.detach(cuda_p2, "piece2");
}

vector<Array<float>> CudaMath::hsplit(Array<float> arr, vector<int64> bchns) {
	throw KaiException(KERR_ASSERT);
	return vector<Array<float>>();
}

int64* CudaMath::alloc_int64_array(int64* host_p, int64 count) {
	int64* cuda_p;
	cudaMalloc(&cuda_p, count * sizeof(int64));
	cudaMemcpy(cuda_p, host_p, count * sizeof(int64), cudaMemcpyHostToDevice);
	return cuda_p;
}

void CudaMath::free_int64_array(int64* cuda_p) {
	cudaFree(cuda_p);
}

Array<float> CudaMath::extract_selected(Array<float> arr, Array<bool> selector) {
	bool m_trace = false;

	CudaConn cuda("extract_selected", NULL);

	arr = cuda.ToCudaArray(arr, "arr");
	selector = cuda.ToCudaArray(selector, "selector");

	if (m_trace) arr.print("arr");
	if (m_trace) selector.print("selector");

	int64 ssize = selector.total_size();
	int64 count = 0;

	bool* p_selector = selector.data_ptr();
	int64* p_map = new int64[ssize];

	for (int64 n = 0; n < ssize; n++) {
		if (p_selector[n]) {
			p_map[count++] = n;
		}
	}

	if (m_trace) logger.Print("count = %lld", count);

	int64* cuda_map = cuda.alloc_int64_mem(Shape(count), "map");
	cuda.Copy_host_to_cuda(cuda_map, p_map, sizeof(int64) * count);
	delete p_map;

	Shape dshape = arr.shape().remove_front().add_front(count);

	float* cuda_dst = cuda.alloc_float_mem(dshape, "dst");
	float* cuda_arr = arr.data_ptr();

	int64 dsize = dshape.total_size();
	int64 drest = dsize / count;

	if (m_trace) logger.Print("dsize = %lld, drest = %lld", dsize, drest);

	cu_call(ker_extract_selected_pickup, dsize, (dsize, cuda_dst, cuda_arr, cuda_map, drest));

	Array<float> dst = cuda.detach(cuda_dst, "dst");

	if (m_trace) dst.print("dst");

	return dst;
}

Array<int64> CudaMath::extract_selected(Array<int64> arr, Array<bool> selector) {
	bool m_trace = false;

	CudaConn cuda("extract_selected", NULL);

	arr = cuda.ToCudaArray(arr, "arr");
	selector = cuda.ToCudaArray(selector, "selector");

	if (m_trace) arr.print("arr");
	if (m_trace) selector.print("selector");

	int64 ssize = selector.total_size();
	int64 count = 0;

	bool* p_selector = selector.data_ptr();
	int64* p_map = new int64[ssize];

	for (int64 n = 0; n < ssize; n++) {
		if (p_selector[n]) {
			p_map[count++] = n;
		}
	}

	if (m_trace) logger.Print("count = %lld", count);

	int64* cuda_map = cuda.alloc_int64_mem(Shape(count), "map");
	cuda.Copy_host_to_cuda(cuda_map, p_map, sizeof(int64) * count);
	delete p_map;

	Shape dshape = arr.shape().remove_front().add_front(count);

	int64* cuda_dst = cuda.alloc_int64_mem(dshape, "dst");
	int64* cuda_arr = arr.data_ptr();

	int64 dsize = dshape.total_size();
	int64 drest = dsize / count;

	if (m_trace) logger.Print("dsize = %lld, drest = %lld", dsize, drest);

	cu_call(ker_extract_selected_pickup_int, dsize, (dsize, cuda_dst, cuda_arr, cuda_map, drest));

	Array<int64> dst = cuda.detach(cuda_dst, "dst");

	if (m_trace) dst.print("dst");

	return dst;
}

Array<float> CudaMath::fill_selected(Array<float> arr, Array<float> slices, Array<bool> selector) {
	bool m_trace = false;

	CudaConn cuda("fill_selected", NULL);

	arr = cuda.ToCudaArray(arr, "arr");
	slices = cuda.ToCudaArray(slices, "slices");
	selector = cuda.ToCudaArray(selector, "selector");

	if (m_trace) arr.print("arr");
	if (m_trace) slices.print("slices");
	if (m_trace) selector.print("selector");

	float* cuda_arr = arr.data_ptr();
	float* cuda_slices = slices.data_ptr();

	int64 ssize = selector.total_size();
	int64 msize = slices.axis_size(0);
	int64 count = 0;

	bool* p_selector = selector.data_ptr();
	int64* p_map = new int64[msize];

	for (int64 n = 0; n < ssize; n++) {
		if (p_selector[n]) {
			p_map[count++] = n;
		}
	}

	assert(count == msize);

	int64* cuda_map = cuda.alloc_int64_mem(Shape(msize), "map");
	cuda.Copy_host_to_cuda(cuda_map, p_map, sizeof(int64) * msize);
	delete p_map;

	int64 dsize = slices.total_size();
	int64 drest = dsize / count;

	if (m_trace) logger.Print("dsize = %lld, drest = %lld", dsize, drest);

	// cudaError: an illegal memory access was encountered in fill_selected
	cu_call(ker_extract_selected_fill, dsize, (dsize, cuda_arr, cuda_slices, cuda_map, drest));

	return arr;
}

Array<int64> CudaMath::argmax(Array<float> a, int64 axis) {
	assert(a.dim() == 2);
	assert(axis == 0);

	CudaConn conn("argmax", NULL);

	int64 nrow = a.axis_size(0), ncol = a.axis_size(1);

	float* cuda_a = conn.GetCudaMem(a, "argmax(a)");
	int64* cuda_n = conn.alloc_int64_mem(Shape(nrow), "argmax(n)");

	cu_call(ker_argmax, nrow, (nrow, cuda_n, cuda_a, ncol));

	return conn.detach(cuda_n, "result");
}

/*
Array<int64> CudaMath::get_hash_idx(Array<float> code) {
	assert(code.dim() == 2);

	CudaConn conn("get_hash_idx", NULL);

	int64 nrow = code.axis_size(0), ncol = code.axis_size(1);

	float* cuda_code = conn.GetCudaMem(code, "code");
	int64* cuda_n = conn.alloc_int64_mem(Shape(nrow), "n");

	cu_call(ker_get_hash_idx, nrow, (nrow, cuda_n, cuda_code, ncol));

	return conn.detach(cuda_n, "result_cuda");
}

Array<int64> CudaMath::eval_hash_diff(Array<int64> hash1, Array<int64> hash2, int64 nvec) {
	assert(hash1.dim() == 1);
	assert(hash1.shape() == hash2.shape());

	CudaConn conn("eval_hash_diff", NULL);

	int64* cuda_h1 = conn.GetCudaMem(hash1, "hash1");
	int64* cuda_h2 = conn.GetCudaMem(hash2, "hash2");
	int64* cuda_n = conn.alloc_int64_mem(hash1.shape(), "hash_diff");

	int64 nrow = hash1.total_size();

	cu_call(ker_get_hash_diff, nrow, (nrow, cuda_n, cuda_h1, cuda_h2, nvec));

	return conn.detach(cuda_n, "result_cuda");
}
*/

Array<float> CudaMath::get_hash_match_point(Array<float> code1, Array<float> code2) {
	CudaConn conn("get_hash_match_point", NULL);

	assert(code1.dim() == 2);
	assert(code2.dim() == 2);
	assert(code1.axis_size(1) == code2.axis_size(1));

	int64 nrow = code1.axis_size(0);
	int64 ncol = code2.axis_size(0);
	int64 nvec = code1.axis_size(1);

	Shape pshape(nrow, ncol);
	
	int64 psize = pshape.total_size();

	float* cuda_p = conn.alloc_float_mem(pshape);
	float* cuda_c1 = conn.GetCudaMem(code1);
	float* cuda_c2 = conn.GetCudaMem(code2);

	cu_call(ker_eveal_hash_match_point, psize, (psize, cuda_p, cuda_c1, cuda_c2, nrow, ncol, nvec));

	Array<float> point = conn.detach(cuda_p);

	return point;
}

Array<float> CudaMath::get_vector_dist(Array<float> code1, Array<float> code2) {
	CudaConn conn("get_vector_dist", NULL);

	assert(code1.dim() == 2);
	assert(code2.dim() == 2);
	assert(code1.axis_size(1) == code2.axis_size(1));

	int64 nrow = code1.axis_size(0);
	int64 ncol = code2.axis_size(0);
	int64 nvec = code1.axis_size(1);

	Shape dshape(nrow, ncol);

	int64 dsize = dshape.total_size();

	float* cuda_d = conn.alloc_float_mem(dshape);
	float* cuda_c1 = conn.GetCudaMem(code1);
	float* cuda_c2 = conn.GetCudaMem(code2);

	cu_call(ker_eveal_vector_dist, dsize, (dsize, cuda_d, cuda_c1, cuda_c2, nrow, ncol, nvec));

	Array<float> dist = conn.detach(cuda_d);

	return dist;
}

Array<float> CudaMath::fft(Array<float> arr) {
	CudaConn conn("fft", NULL);

	Shape datShape = arr.shape();
	Shape bufShape = datShape.append(2);

	int64 data_num = datShape[-1];

	assert(log2(data_num) - (int)log2(data_num) == 0);

	int64 dsize = datShape.total_size();
	int64 bsize = bufShape.total_size();

	Array<float> data = CudaConn::ToCudaArray(arr);

	float* cuda_d = conn.attach(data);
	float* cuda_y = conn.alloc_float_mem(datShape);
	float* cuda_buf1 = conn.alloc_float_mem(bufShape);
	float* cuda_buf2 = conn.alloc_float_mem(bufShape);

	cu_call(ker_real_to_complex, bsize, (bsize, cuda_buf1, cuda_d));

	fft_core(cuda_buf1, cuda_buf2, cuda_y, data_num, dsize, bsize);

	Array<float> y = conn.detach(cuda_y);

	return y;
}

Array<float> CudaMath::fft(Array<short> arr) {
	CudaConn conn("fft", NULL);

	Shape datShape = arr.shape();
	Shape bufShape = datShape.append(2);

	int64 data_num = datShape[-1];

	assert(log2(data_num) - (int)log2(data_num) == 0);

	int64 dsize = datShape.total_size();
	int64 bsize = bufShape.total_size();

	Array<short> data = CudaConn::ToCudaArray(arr);

	short* cuda_d = conn.attach_short(data);

	float* cuda_y = conn.alloc_float_mem(datShape);
	float* cuda_buf1 = conn.alloc_float_mem(bufShape);
	float* cuda_buf2 = conn.alloc_float_mem(bufShape);

	cu_call(ker_short_to_complex, bsize, (bsize, cuda_buf1, cuda_d));

	fft_core(cuda_buf1, cuda_buf2, cuda_y, data_num, dsize, bsize);

	Array<float> y = conn.detach(cuda_y);

	return y;
}

void CudaMath::fft_core(float* cuda_buf1, float* cuda_buf2, float* cuda_y, int64 data_num, int64 dsize, int64 bsize) {
	float* src = cuda_buf1;
	float* dst = cuda_buf2;

	int64 step = 1;

	while (step < data_num) {
		step = step * 2;
		cu_call(ker_fft_step, bsize, (bsize, dst, src, data_num, step));
		float* tmp = dst;
		dst = src;
		src = tmp;
	}

	cu_call(ker_complex_to_abs, dsize, (dsize, cuda_y, src));
}

void CudaMath::fft_core_split(float* cuda_buf1, float* cuda_buf2, float* cuda_y, int64 data_num, int64 freq_cnt, int64 dsize, int64 bsize, int64 split) {
	float* src = cuda_buf1;
	float* dst = cuda_buf2;

	assert(bsize % (split * data_num * 2) == 0);

	int64 mb_size = bsize / (data_num * 2);
	int64 sp_size = mb_size / split;
	int64 ssize = bsize / split;

	int64 step = 1;

	while (step < data_num) {
		step = step * 2;
		for (int64 n = 0; n < split; n++) {
			int64 nd_base = n * sp_size;
			cu_call(ker_fft_step_split, ssize, (ssize, dst, src, data_num, step, nd_base));
		}
		float* tmp = dst;
		dst = src;
		src = tmp;
	}

	assert(data_num >= 2 * freq_cnt);
	
	cu_call(ker_complex_to_abs_mean, dsize, (dsize, cuda_y, src, data_num, freq_cnt));
}
