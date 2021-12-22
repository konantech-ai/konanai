/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "value.h"
#include "dim.h"
#include "array.h"
#include "shape.h"
#include "idx.h"

class CudaConn;

class HostMath {
public:
	HostMath() {}
	virtual ~HostMath() {}

	virtual Array<float> random_uniform(Shape shape, CudaConn* pConn = NULL);
	virtual Array<float> random_bernoulli(Shape shape, float prob, CudaConn* pConn = NULL);
	virtual Array<float> random_normal(float mean, float std, Shape shapev, CudaConn* pConn = NULL);

	virtual Array<int64> arange(int64 range);
	virtual void shuffle(int64 size, int64* nums);

	virtual Array<float> zeros(Shape shape, CudaConn* pConn = NULL);
	virtual Array<float> ones(Shape shape, float coef = 1.0, CudaConn* pConn = NULL);

	virtual Array<float> onehot(Array<int64> idxs, int64 nom_cnt);

	virtual Array<int64> zeros_int(Shape shape, CudaConn* pConn = NULL);
	virtual Array<int64> ones_int(Shape shape, int64 coef = 1, CudaConn* pConn = NULL);

	virtual void array_copy(Array<float>& dest, Array<float> src, Axis dest_axis, Axis src_axis);
	virtual void array_copy(Array<float>& dest, Axis dest_axis, float val);
	virtual void array_add(Array<float>& dest, Array<float> src, Axis dest_axis, Axis src_axis);
	virtual void array_sub(Array<float>& dest, Array<float> src, Axis dest_axis, Axis src_axis);

	virtual Array<float> matmul(Array<float> a, Array<float> b);
	virtual Array<float> dotmul(Array<float> a, Array<float> b);

	virtual Array<float> dotmul_derv(Array<float> a, Array<float> b);

	virtual Array<float> square(Array<float> a);
	virtual Array<float> abs(Array<float> a);
	virtual Array<float> sqrt(Array<float> a);
	virtual Array<float> log(Array<float> a);
	virtual Array<float> exp(Array<float> a);
	virtual Array<float> sign(Array<float> a);

	virtual Array<int64> round(Array<float> a);
	virtual Array<int64> binary_row_to_int(Array<int64> a);

	virtual Array<float> fetch_rows(Array<float> a, vector<int64> rows);
	virtual Array<float> dotsum(Array<float> a, Array<float> other);
	virtual Array<int64> sort_columns(Array<float> a, sortdir dir, int64 max_cnt);
	virtual Array<int64> sort_columns(Array<float> a1, sortdir dir1, Array<float> a2, sortdir dir2, int64 max_cnt);

	virtual Array<float> to_float(Array<unsigned char> a);
	virtual Array<float> to_float(Array<int64> a);
	virtual Array<float> to_float(Array<bool> a);

	virtual Array<float> get_hash_match_point(Array<float> code1, Array<float> code2);
	virtual Array<float> get_vector_dist(Array<float> code1, Array<float> code2);

	virtual Array<float> sum(Array<float> a, int64 axis);
	virtual Array<float> avg(Array<float> a, int64 axis);
	virtual Array<float> var(Array<float> a, int64 axis, Array<float>* pavg = NULL);
	virtual Array<float> max(Array<float> a, int64 axis);
	virtual Array<int64> argmax(Array<float> a, int64 axis);
	virtual Array<float> maxarg(Array<float> a, int64 axis, Array<int64>& arg);

	virtual Array<float> mult_scalar(Array<float> arr, float other);

	virtual Array<bool> compare_rows(Array<float> arr1, Array<float> arr2);
	virtual Array<bool> compare_rows(Array<int64> arr1, Array<int64> arr2);

	virtual Array<float> expand(Array<float> arr, int64 ratio, int64 axis);
	virtual Array<float> sum_on_expanded(Array<float> arr, int64 ratio, int64 axis);

	virtual Array<float> get_row(Array<float> arr, int64 nth);

	virtual Array<int64> get_col(Array<int64> arr, int64 nth);

	virtual void set_row(Array<float> arr, int64 nth, Array<float> other);
	virtual void set_row(Array<float> arr, int64 nth, float other);

	virtual void sub_row(Array<float> arr, int64 nth, Array<float> other);

	virtual int64 count_in_range(Array<float> a, float min_val, float max_val);
	virtual int64 true_count(Array<bool> a);

	virtual void setarg(Array<float> a, Array<int64>& arg, Array<float> values);

	virtual float sum(Array<float> a);
	virtual int64 sum(Array<int64> a);
	virtual float max(Array<float> a, int64* pargmax); // pargmax = NULL 기정치 지정 삭제
	virtual float mean(Array<float> a);
	virtual float mean(Array<bool> a);

	virtual float mean(List a);
	virtual float square_sum(Array<float> arr);

	virtual Array<float> maximum(Array<float> ans, Array<float> est);
	virtual Array<float> maximum(Array<float> ans, float est);

	virtual Array<float> sigmoid_cross_entropy_with_logits(Array<float> ans, Array<float> est);
	virtual Array<float> softmax_cross_entropy_with_logits(Array<float> ans, Array<float> est);
	virtual Array<float> softmax_cross_entropy_with_logits_idx(Array<int64> ans, Array<float> est);
	virtual Array<float> softmax_cross_entropy_with_logits_1st(Array<float> est);

	virtual Array<float> relu(Array<float> affine);
	virtual Array<float> sigmoid(Array<float> affine);
	virtual Array<float> tanh(Array<float> affine);
	virtual Array<float> leaky_relu(Array<float> affine, float alpha);
	virtual Array<float> gelu(Array<float> affine);

	virtual Array<float> relu_derv(Array<float> y);
	virtual Array<float> sigmoid_derv(Array<float> y);
	virtual Array<float> tanh_derv(Array<float> y);
	virtual Array<float> leaky_relu_derv(Array<float> y, float alpha);
	virtual Array<float> gelu_derv(Array<float> x);

	virtual Array<float> softmax(Array<float> affine);

	virtual Array<float> softmax_derv(Array<float> y);

	virtual Array<float> sigmoid_cross_entropy_with_logits_derv(Array<float> ans, Array<float> est);
	virtual Array<float> softmax_cross_entropy_with_logits_derv(Array<float> ans, Array<float> est);
	virtual Array<float> softmax_cross_entropy_with_logits_idx_derv(Array<int64> ans, Array<float> est);
	virtual Array<float> softmax_cross_entropy_with_logits_1st_derv(Array<float> est);

	virtual Array<float> conv(Array<float> arr, Array<float> kernel, Array<float> bias);
	virtual Array<float> conv_backprop(Array<float> G_y, Array<float> x, Array<float> kernel, Array<float>& G_k, Array<float>& G_b);

	virtual Array<float> m_get_ext_regions(Array<float> x, int64 kh, int64 kw, float fill);
	virtual Array<float> m_undo_ext_regions(Array<float> G_regs, Array<float> x, Array<float> kernel);

	virtual Array<float> expand_images(Array<float> arr, Shape stride);
	virtual Array<float> expand_undo_images(Array<float> arr, Shape stride);

	virtual Array<float> transpose(Array<float> arr);
	virtual Array<float> transpose(Array<float> arr, Idx idx);
	virtual Array<unsigned char> transpose(Array<unsigned char> arr, Idx idx);

	virtual Array<float> reshape(Array<float> arr, Shape shape);
	virtual Array<int64> reshape(Array<int64> arr, Shape shape);
	virtual Array<unsigned char> reshape(Array<unsigned char> arr, Shape shape);
	virtual Array<float> init_grid(Shape shape);

	virtual Array<float> tile(Array<float> arr, int64 num);
	virtual Array<float> untile(Array<float> arr, int64 num);
	virtual Array<float> flatten(Array<float>  arr);
	virtual Array<int64> flatten(Array<int64>  arr);

	virtual Array<float> vstack(Array<float> arr1, Array<float> arr2);
	virtual Array<float> vstack(vector<Array<float>> arrs);
	
	//virtual Array<short> vstack(vector<Array<short>> arrs);

	virtual Array<float> hstack(Array<float> arr1, Array<float> arr2);
	virtual Array<float> hstack(vector<Array<float>> arrs);

	virtual void hsplit(Array<float> arr, int64 p1_size, Array<float>& piece1, Array<float>& piece2);

	virtual vector<Array<float>> hsplit(Array<float> arr, vector<int64> bchns);

	Array<float> fft(Array<float> arr);
	Array<float> fft(Array<short> arr);

#ifdef KAI2021_WINDOWS
	virtual float power(float base, float exp) { return (float) pow(base, exp); }
#else
	virtual float power(float base, float exp) { return pow(base, exp); }
#endif

	virtual Array<float> power(Array<float> arr, float exp);

	virtual void load_from_file(string filepath, Array<unsigned char>& arr);

	virtual int64 fread_int_msb(FILE* fid);

	virtual Array<float> wvec_select_idx(Array<float> arr, Array<int64> idxs, int64 dic_count, int64* voc_counts);

	virtual Array<float> wvec_select(Array<float> arr, Array<int64> idxs);
	virtual Array<float> select_col(Array<float> arr, Array<int64> idxs);
	virtual Array<float> minus_1_on_idx(Array<float> arr, Array<int64> idxs);
	virtual Array<float> minus_1_on_1st(Array<float> arr);

	virtual Array<float> select_rnn_last_col(Array<float> arr);

	virtual Array<int> to_int(Array<float> arr);
	virtual Array<int64> to_int64(Array<float> arr);

	virtual void acc_abs_n_sq_num(Array<float> params, float& abs_sum, float& sq_sum);
	virtual void acc_nearzero_sum_sqnum(Array<float> params, float& sum, float& sq_sum, int64& near_zero_cnt, float threshold);

	virtual int64* alloc_int64_array(int64* host_p, int64 count);
	virtual void free_int64_array(int64* cuda_p);

	virtual Array<float> mask_future_timesteps(Array<float> arr, int64 timesteps);

	virtual Array<int64> get_hash_idx(Array<float> code);
	virtual Array<int64> eval_hash_diff(Array<int64>m_hash, Array<int64>key_hash_cuda, int64 nvec);

	template <class T>
	Array<T> copy_array(Array<T> src, Axis src_axis);

	template <class T>
	void bin_op(Array<T> a, Array<T> b, T(*op)(T a, T b));

	template <class T>
	void bin_op(Array<T> a, T b, T(*op)(T a, T b));

	template <class T>
	void log_bin_op(Array<bool> result, Array<T> a, Array<T> b, bool(*op)(T a, T b));

	template <class T>
	void log_bin_op(Array<bool> result, Array<T> a, T b, bool(*op)(T a, T b));

	template <class T>
	Array<T> extract_selected(Array<T> arr, Array<bool> selector);

	template <class T>
	Array<T> fill_selected(Array<T> arr, Array<T> slices, Array<bool> selector);

	template <class T>
	vector<Array<T>> hsplit_last(Array<T> arr);

	template <class T>
	Array<T> mult_dim(Array<T> arr, int64 axis);

	template <class T>
	Array<T> filter(Array<T> arr, int64 axis, int64 index);

	template <class T>
	Array<T> unfilter(Array<T> arr, Array<T> filtered, int64 axis, int64 index);

protected:
	template <class T>
	void m_array_copy(T* pSrc, T* pDst, ArrayLoopAxisInfo<T>* src_info, ArrayLoopAxisInfo<T>* dst_info, int64 nth, int64 dim);

	void m_array_copy(float* pDst, ArrayLoopAxisInfo<float>* dst_info, int64 nth, int64 dim, float val);
	void m_array_add(float* pSrc, float* pDst, ArrayLoopAxisInfo<float>* src_info, ArrayLoopAxisInfo<float>* dst_info, int64 nth, int64 dim);
	void m_array_sub(float* pSrc, float* pDst, ArrayLoopAxisInfo<float>* src_info, ArrayLoopAxisInfo<float>* dst_info, int64 nth, int64 dim);

	static float ms_bin_op_sigmoid_cross_entropy_with_logits(float a, float b);
};

extern HostMath hmath;
