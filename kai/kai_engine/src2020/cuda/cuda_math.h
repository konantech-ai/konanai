/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/value.h"
#include "../core/dim.h"
#include "../core/array.h"
#include "../core/shape.h"
#include "../core/idx.h"
#include "../core/host_math.h"

class CudaConn;

class CudaMath : public HostMath {
public:
	CudaMath() {}
	virtual ~CudaMath() {}

	Array<float> random_uniform(Shape shape, CudaConn* pConn = NULL);
	Array<float> random_bernoulli(Shape shape, float prob_threshod, CudaConn* pConn = NULL);
	Array<float> random_normal(float mean, float std, Shape shape, CudaConn* pConn = NULL);

	Array<float> zeros(Shape shape, CudaConn* pConn = NULL);
	Array<float> ones(Shape shape, float coef = 1.0, CudaConn* pConn = NULL);

	Array<int64> ones_int(Shape shape, int64 coef = 1, CudaConn* pConn = NULL);

	Array<float> sigmoid(Array<float> arr);
	Array<float> softmax(Array<float> output);

	float square_sum(Array<float> arr);
	
	Array<float> mult_scalar(Array<float> arr, float other);

	Array<float> vstack(Array<float> arr1, Array<float> arr2);
	Array<float> vstack(vector<Array<float>> arrs);

	//Array<short> vstack(vector<Array<short>> arrs);

	Array<float> hstack(Array<float> arr1, Array<float> arr2);
	Array<float> hstack(vector<Array<float>> arrs);

	void hsplit(Array<float> arr, int64 p1_size, Array<float>& piece1, Array<float>& piece2);

	vector<Array<float>> hsplit(Array<float> arr, vector<int64> bchns);

	void acc_abs_n_sq_num(Array<float> params, float& abs_sum, float& sq_sum);
	void acc_nearzero_sum_sqnum(Array<float> params, float& sum, float& sq_sum, int64& near_zero_cnt, float threshold);

	int64* alloc_int64_array(int64* host_p, int64 count);
	void free_int64_array(int64* cuda_p);

	void set_row(Array<float> arr, int64 nth, float other);
	void set_row(Array<float> arr, int64 nth, Array<float> other);

	Array<int64> extract_selected(Array<int64> arr, Array<bool> selector);
	Array<float> extract_selected(Array<float> arr, Array<bool> selector);

	Array<float> fill_selected(Array<float> arr, Array<float> slices, Array<bool> selector);

	Array<int64> argmax(Array<float> a, int64 axis);
	//Array<int64> get_hash_idx(Array<float> code);
	//Array<int64> eval_hash_diff(Array<int64>m_hash, Array<int64>key_hash_cuda, int64 nvec);
	Array<float> get_hash_match_point(Array<float> code1, Array<float> code2);
	Array<float> get_vector_dist(Array<float> code1, Array<float> code2);

	Array<float> fft(Array<float> arr);
	Array<float> fft(Array<short> arr);

	void fft_core(float* cuda_buf1, float* cuda_buf2, float* cuda_y, int64 data_num, int64 dsize, int64 bsize);
	void fft_core_split(float* cuda_buf1, float* cuda_buf2, float* cuda_y, int64 data_num, int64 freq_cnt, int64 dsize, int64 bsize, int64 split=1);

protected:
	//Array<float> to_host(Array<float> arr);


	//Array<float> matmul(Array<float> a, Array<float> b);
	//Array<float> dotmul(Array<float> a, Array<float> b);
};

extern CudaMath cmath;

