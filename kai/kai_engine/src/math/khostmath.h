/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "kmath.h"

class KaiHostMath : public KaiMath {
public:
	KaiHostMath();
	virtual ~KaiHostMath();

	virtual void cudaErrorCheck();

	virtual KaiArray<KFloat> copy(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> zeros(KaiShape shape);
	virtual KaiArray<KFloat> ones(KaiShape shape, KFloat fill = 1.0f);
	virtual KaiArray<KFloat> random_uniform(KaiShape shape);
	virtual KaiArray<KFloat> random_normal(KaiShape shape, KFloat mean, KFloat std, KBool adapt = false);

	virtual KaiArray<KInt> to_cuda(KaiArray<KInt> arr);
	virtual KaiArray<KFloat> to_cuda(KaiArray<KFloat> arr);

	virtual KaiArray<KInt> to_host(KaiArray<KInt> arr);
	virtual KaiArray<KFloat> to_host(KaiArray<KFloat> arr);

	// Added by Hyung-jae, Son (2021-08-20)
	virtual void to_cuda(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount);
	virtual void to_cuda(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount);
	virtual void to_host(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount);
	virtual void to_host(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount);

	virtual KaiArray<KInt> arange(KInt nCount);
	virtual KaiArray<KInt> subrange(KaiArray<KInt> arr, KInt nStart, KInt nCount);

	virtual void shuffle(KaiArray<KInt> arr);
	virtual void shuffle(KaiArray<KInt> arr, KInt nStart, KInt nCount);

	virtual KaiArray<KFloat> matmul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> add_bias(KaiArray<KFloat> arr, KaiArray<KFloat> bias);

	virtual KaiArray<KFloat> transpose(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> sum_on_column(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> sum_on_row(KaiArray<KFloat> arr);

	virtual KaiArray<KFloat> acivate(KaiArray<KFloat> arr, KInt nActFuncID, KaiExecContext* pContext);
	virtual KaiArray<KFloat> acivate_backprop(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat> y, KInt nActFuncID, KaiExecContext* pContext);

	virtual KaiArray<KFloat> sign(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> square(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> sqrt(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> log(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> exp(KaiArray<KFloat> arr);

	virtual KaiArray<KFloat> sum(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> sum_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr);
	virtual KaiArray<KFloat> mean(KaiArray<KFloat> farr);

	virtual KaiArray<KFloat> minus(KaiArray<KFloat> arr);

	virtual KaiArray<KFloat> argmax(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> max(KaiArray<KFloat> arr);

	virtual KaiArray<KFloat> eval_binary_op(exp_op op_code, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> eval_binary_op(exp_op op_code, KaiArray<KFloat> arr, KFloat term);

	virtual KaiArray<KFloat> add(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> sub(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> mul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> div(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);

	virtual KaiArray<KFloat> gt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> lt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> equal(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);

	virtual KaiArray<KFloat> filter(KaiArray<KFloat> xarr, KaiArray<KInt> mask);

	virtual KaiArray<KFloat> sigmoid_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> softmax_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	
	virtual KaiArray<KFloat> softmax_cross_entropy_with_logits_idx(KaiArray<KFloat> arr1, KaiArray<KInt> arr2);
	virtual KaiArray<KFloat> softmax_cross_entropy_with_logits_idx_derv(KaiArray<KFloat> arr1, KaiArray<KInt> arr2);

	virtual KaiArray<KFloat> equal_col(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> max_col(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> max_col_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr);

	virtual void vstack(KaiArray<KFloat> arr_on, KaiArray<KFloat> arr, KInt nFrom);
	virtual KaiArray<KFloat> vstack_grad(KaiArray<KFloat> grad, KInt nStart, KInt nCount);

	virtual KaiArray<KFloat> iou_yolo(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual KaiArray<KFloat> iou_yolo_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2, KInt nth);

	virtual void add_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual void sub_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual void mul_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);
	virtual void div_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2);

	virtual KaiArray<KFloat> add(KaiArray<KFloat> arr, KFloat term);
	virtual KaiArray<KFloat> sub(KaiArray<KFloat> arr, KFloat term);
	virtual KaiArray<KFloat> mul(KaiArray<KFloat> arr, KFloat term);
	virtual KaiArray<KFloat> div(KaiArray<KFloat> arr, KFloat term);

	virtual void mul_on(KaiArray<KFloat> arr, KFloat term);

	virtual KaiArray<KFloat> eval_adam_delta(KaiArray<KFloat> grad, KaiArray<KFloat> s, KaiArray<KFloat> t, KFloat n, KFloat ro1, KFloat ro2, KFloat epsilon);
	virtual KaiArray<KFloat> apply_decay(KaiArray<KFloat> pm, KaiArray<KFloat> grad, KFloat l2_decay, KFloat l1_decay);

	virtual KInt fetch(KaiArray<KInt> arr, KInt nIndex = 0);
	virtual KInt fetch(KInt* arr, KInt nIndex = 0);
	virtual KFloat fetch(KaiArray<KFloat> arr, KInt nIndex = 0);
	virtual KFloat fetch(KFloat* arr, KInt nIndex = 0);

	virtual KaiArray<KFloat> sigmoid(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> sigmoid_derv_grad(KaiArray<KFloat> gsig, KaiArray<KFloat> sig);

	virtual KaiArray<KFloat> softmax(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> softmax_derv(KaiArray<KFloat> gyarr, KaiArray<KFloat> yarr);

public:
	virtual KaiArray<KFloat> relu(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> tanh(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> leaky_relu(KaiArray<KFloat> arr, KFloat leaky_alpha);
	virtual KaiArray<KFloat> gelu(KaiArray<KFloat> arr);

	virtual KaiArray<KFloat> relu_derv(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> sigmoid_derv(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> tanh_derv(KaiArray<KFloat> arr);
	virtual KaiArray<KFloat> leaky_relu_derv(KaiArray<KFloat> arr, KFloat leaky_alpha);
	virtual KaiArray<KFloat> gelu_derv(KaiArray<KFloat> arr);

	KInt getDevAllocSize() { return 0; }

	virtual KaiList to_host(KaiList list);
	virtual KaiDict to_host(KaiDict dict);

	virtual KFloat mean(KaiList list);
	virtual KFloat sum(KaiList list);

	virtual KaiArray<KFloat> convolution(KaiArray<KFloat> xarr, KaiArray<KFloat> kernel);
	virtual KaiArray<KFloat> convolution_derv_x(KaiArray<KFloat> gyarr, KaiArray<KFloat> kernel);
	virtual KaiArray<KFloat> convolution_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> xarr, KaiShape kshape);

	virtual KaiArray<KFloat> subrange(KaiArray<KFloat> xarr, KInt nth_ax, KInt nFrom, KInt nCount);
	virtual KaiArray<KFloat> subrange_derv(KaiArray<KFloat> gyarr, KInt nth_ax, KInt nFrom, KInt nCount);

	virtual KaiArray<KFloat> stride(KaiArray<KFloat> xarr, KaiShape stride);
	virtual KaiArray<KFloat> stride_derv(KaiArray<KFloat> gyarr, KaiShape stride, KaiShape xshape);

	virtual KaiArray<KFloat> max_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pMaxMap, KaiShape kernel);
	virtual KaiArray<KFloat> max_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> maxMap, KaiShape kernel);
	
	virtual KaiArray<KFloat> avg_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pAvgCnt, KaiShape kernel);
	virtual KaiArray<KFloat> avg_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> avgCnt, KaiShape kernel);

	virtual KaiArray<KFloat> globalavg(KaiArray<KFloat> xarr);
	virtual KaiArray<KFloat> globalavg_derv(KaiArray<KFloat> gyarr, KaiShape xshape);

	virtual KaiArray<KFloat> BNCollectNorm(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KaiArray<KFloat>& var, KFloat momentum, KFloat epsilon);
	virtual KaiArray<KFloat> BnNormalize(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KFloat epsilon);
	virtual KaiArray<KFloat> BnScale(KaiArray<KFloat> xarr, KaiArray<KFloat> scale, KaiArray<KFloat> shift);
	virtual KaiArray<KFloat> BnNormDerv(KaiArray<KFloat> garr, KaiArray<KFloat> var, KFloat epsilon);

	virtual void rescale_derv_pm(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat>* p_grad_scale, KaiArray<KFloat>* p_grad_shift);
	virtual KaiArray<KFloat> rescale_derv_x(KaiArray<KFloat> garr, KaiArray<KFloat> scale);

	virtual void CopyIntoSlice(KaiArray<KFloat> yarr, KaiArray<KFloat> barr, KInt& nChnPos);
	virtual KaiArray<KFloat> CopyFromSlice(KaiArray<KFloat> garr, KInt& nChnPos, KInt nChnCnt);

	virtual KaiArray<KFloat> random_bernoulli(KaiShape xshape, KFloat one_ratio);
	virtual KaiArray<KFloat> dropout(KaiArray<KFloat> xarr, KaiArray<KFloat> mask, KFloat keep_raio);
	virtual KaiArray<KFloat> dropout_derv(KaiArray<KFloat> xarr, KaiArray<KFloat> mask, KFloat keep_raio);

	virtual void residual_add(KaiArray<KFloat> yarr, KaiArray<KFloat> xarr);
	virtual KaiArray<KFloat> residual_add_derv(KaiArray<KFloat> gyarr, KInt bchn);

	virtual KaiArray<KFloat> CombineExtendedInput(KaiArray<KFloat> recurrent, KBool isSeq, KaiArray<KFloat> xarr, KInt nth);
	virtual KaiArray<KFloat> SplitExtendedInputGrad(KaiArray<KFloat> g_exp_input, KBool isSeq, KaiArray<KFloat> g_x, KInt nth);
	virtual void CopyIntoTimeSlice(KaiArray<KFloat> yarr, KaiArray<KFloat> recurrent, KInt nth);
	virtual void add_time_slice_on_dest(KaiArray<KFloat> dest, KaiArray<KFloat> whole, KInt nth);

	virtual KaiArray<KFloat> lstm_gates(KaiArray<KFloat> affine);
	virtual KaiArray<KFloat> lstm_proc(KaiArray<KFloat> gates, KaiArray<KFloat>& state, KBool use_state);

	virtual KaiArray<KFloat> lstm_gates_derv(KaiArray<KFloat> g_gates, KaiArray<KFloat> gates);
	virtual KaiArray<KFloat> lstm_proc_derv(KaiArray<KFloat>& g_state, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_state, KaiArray<KFloat> post_recur, KBool use_state);

	virtual KaiArray<KFloat> gru_combine_extra(KaiArray<KFloat> exp_input, KaiArray<KFloat> gates);
	virtual void gru_combine_extra_derv(KaiArray<KFloat> g_exp_input, KaiArray<KFloat> g_gates, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> exp_input);
	virtual KaiArray<KFloat> gru_proc(KaiArray<KFloat> gates, KaiArray<KFloat> recurrent, KaiArray<KFloat> extra_affine);
	virtual KaiArray<KFloat> gru_proc_derv(KaiArray<KFloat>& g_gates, KaiArray<KFloat>& g_new_rec, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_recur, KaiArray<KFloat> extra_affine);

	virtual void add_embed_dict(KaiArray<KFloat> yarr, KaiArray<KInt> tokens, KaiArray<KFloat> word_vecs, KInt axis);

	virtual KaiList split_array(KaiArray<KFloat> arr, KInt piece_cnt);
	virtual KaiArray<KFloat> merge_array(KaiList arrs);

	virtual KaiArray<KFloat> multi_head_matmul_qk(KaiArray<KFloat> query, KaiArray<KFloat> key, KInt head_cnt);
	virtual KaiArray<KFloat> multi_head_matmul_qk_derv_q(KaiArray<KFloat> gyarr, KaiArray<KFloat> key, KInt head_cnt);
	virtual KaiArray<KFloat> multi_head_matmul_qk_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> query, KInt head_cnt);

	virtual KaiArray<KFloat> multi_head_matmul_pv(KaiArray<KFloat> probs, KaiArray<KFloat> value);
	virtual KaiArray<KFloat> multi_head_matmul_pv_derv_p(KaiArray<KFloat> gyarr, KaiArray<KFloat> value, KInt head_cnt);
	virtual KaiArray<KFloat> multi_head_matmul_pv_derv_v(KaiArray<KFloat> gyarr, KaiArray<KFloat> probs);

	virtual KaiArray<KFloat> extract(KaiArray<KFloat> xarr, KInt axis, KInt index, KInt count, KBool reduce_seq);
	virtual KaiArray<KFloat> extract_derv(KaiArray<KFloat> gyarr, KaiShape xshape, KInt axis, KInt index, KInt count, KBool reduce_seq);

	virtual KaiArray<KFloat> select(KaiArray<KFloat> xarr, KaiArray<KInt> selector, KaiShape vector_shape);
	virtual KaiArray<KFloat> select_derv(KaiArray<KFloat> garr, KaiArray<KInt> selector_arr, KaiShape xshape, KaiShape vshape);

	virtual void update_dic_weight_sgd(KaiArray<KFloat> weight, KaiArray<KFloat> grad, KaiArray<KInt> tokens, KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);
	virtual void update_dic_weight_adam(KaiArray<KFloat> weight, KaiArray<KFloat> s, KaiArray<KFloat> t, KaiArray<KFloat> n, KaiArray<KFloat> grad, KaiArray<KInt> tokens, KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay, KFloat ro1, KFloat ro2, KFloat epsilon);

	virtual KaiArray<KFloat> expand(KaiArray<KFloat> xarr, KaiShape ratio);
	virtual KaiArray<KFloat> expand_derv(KaiArray<KFloat> gyarr, KaiShape ratio);

	virtual KInt stack_on(KaiArray<KFloat> dest, KaiArray<KFloat> src, KInt tail_size, KInt nFrom, KInt nTo);
	virtual KaiArray<KFloat> stack_on_grad(KaiArray<KFloat> gyarr, KaiShape shape, KInt tail_size, KInt& nFrom, KInt nTo);

	virtual KaiArray<KFloat> get_subvector(KaiArray<KFloat> arr, KInt nStart, KInt nCount);
	virtual void get_subvector_derv_acc(KaiArray<KFloat> grad, KaiArray<KFloat> grad_subvec, KInt nStart, KInt nCount);

	virtual void fft(KFloat* pWave, KFloat* pFTT, KInt mb_size, KInt fetch_width, KInt step_width, KInt step_cnt, KInt fft_width, KInt freq_cnt);

protected:
	std::default_random_engine m_randGen;
	KFloat m_eval_binary_op(exp_op op_code, KFloat elem1, KFloat elem2);
};

extern KaiHostMath hostmath;