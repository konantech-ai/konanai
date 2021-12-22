/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "karray.h"

class KaiModelInstance;
class KaiExecContext;

class KaiMath {
public:
	KaiMath() {}
	virtual ~KaiMath() {}

	static KaiMath* GetHostMath();
	static KaiMath* Allocate(KaiModelInstance* pModelContext);

	virtual void cudaErrorCheck() = 0;

	virtual KaiArray<KFloat> copy(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> zeros(KaiShape shape) = 0;
	virtual KaiArray<KFloat> ones(KaiShape shape, KFloat fill=1.0f) = 0;
	virtual KaiArray<KFloat> random_uniform(KaiShape shape) = 0;
	virtual KaiArray<KFloat> random_normal(KaiShape shape, KFloat mean, KFloat std, KBool adapt) = 0;

	virtual KaiArray<KInt> to_cuda(KaiArray<KInt> arr) = 0;
	virtual KaiArray<KFloat> to_cuda(KaiArray<KFloat> arr) = 0;

	virtual KaiArray<KInt> to_host(KaiArray<KInt> arr) = 0;
	virtual KaiArray<KFloat> to_host(KaiArray<KFloat> arr) = 0;

	// Added by Hyung-jae, Son (2021-08-20)
	virtual void to_cuda(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) = 0;
	virtual void to_cuda(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) = 0;
	virtual void to_host(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) = 0;
	virtual void to_host(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) = 0;

	virtual KaiArray<KInt> arange(KInt nCount) = 0;
	virtual KaiArray<KInt> subrange(KaiArray<KInt> arr, KInt nStart, KInt nCount) = 0;

	virtual void shuffle(KaiArray<KInt> arr) = 0;
	virtual void shuffle(KaiArray<KInt> arr, KInt nStart, KInt nCount) = 0;

	virtual KaiArray<KFloat> matmul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> add_bias(KaiArray<KFloat> arr, KaiArray<KFloat> bias) = 0;

	virtual KaiArray<KFloat> transpose(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> sum_on_column(KaiArray<KFloat> arr) = 0;

	virtual KaiArray<KFloat> acivate(KaiArray<KFloat> arr, KInt nActFuncI, KaiExecContext* pContextD) = 0;
	virtual KaiArray<KFloat> acivate_backprop(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat> y, KInt nActFuncID, KaiExecContext* pContext) = 0;

	virtual KaiArray<KFloat> sign(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> square(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> sqrt(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> log(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> exp(KaiArray<KFloat> arr) = 0;

	virtual KaiArray<KFloat> sum(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> sum_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr) = 0;

	virtual KaiArray<KFloat> mean(KaiArray<KFloat> arr) = 0;

	virtual KaiArray<KFloat> minus(KaiArray<KFloat> arr) = 0;

	virtual KaiArray<KFloat> argmax(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> max(KaiArray<KFloat> arr) = 0;

	virtual KaiArray<KFloat> eval_binary_op(exp_op op_code, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> eval_binary_op(exp_op op_code, KaiArray<KFloat> arr, KFloat term) = 0;

	virtual KaiArray<KFloat> add(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> sub(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> mul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> div(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;

	virtual KaiArray<KFloat> gt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> lt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> equal(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;

	virtual KaiArray<KFloat> filter(KaiArray<KFloat> xarr, KaiArray<KInt> mask) = 0;

	virtual KaiArray<KFloat> sigmoid_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> softmax_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;

	virtual KaiArray<KFloat> softmax_cross_entropy_with_logits_idx(KaiArray<KFloat> arr1, KaiArray<KInt> arr2) = 0;
	virtual KaiArray<KFloat> softmax_cross_entropy_with_logits_idx_derv(KaiArray<KFloat> arr1, KaiArray<KInt> arr2) = 0;

	virtual KaiArray<KFloat> equal_col(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> max_col(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> max_col_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr) = 0;

	virtual KaiArray<KFloat> iou_yolo(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual KaiArray<KFloat> iou_yolo_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2, KInt nth) = 0;
	
	virtual void vstack(KaiArray<KFloat> arr_on, KaiArray<KFloat> arr, KInt nFrom) = 0;
	virtual KaiArray<KFloat> vstack_grad(KaiArray<KFloat> grad, KInt nStart, KInt nCount) = 0;

	virtual void add_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual void sub_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual void mul_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;
	virtual void div_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) = 0;

	virtual KaiArray<KFloat> add(KaiArray<KFloat> arr, KFloat term) = 0;
	virtual KaiArray<KFloat> sub(KaiArray<KFloat> arr, KFloat term) = 0;
	virtual KaiArray<KFloat> mul(KaiArray<KFloat> arr, KFloat term) = 0;
	virtual KaiArray<KFloat> div(KaiArray<KFloat> arr, KFloat term) = 0;
	
	virtual void mul_on(KaiArray<KFloat> arr, KFloat term) = 0;

	virtual KaiArray<KFloat> eval_adam_delta(KaiArray<KFloat> grad, KaiArray<KFloat> s, KaiArray<KFloat> t, KFloat n, KFloat ro1, KFloat ro2, KFloat epsilon) = 0;
	virtual KaiArray<KFloat> apply_decay(KaiArray<KFloat> pm, KaiArray<KFloat> grad, KFloat l2_decay, KFloat l1_decay) = 0;

	virtual KInt fetch(KaiArray<KInt> arr, KInt nIndex = 0) = 0;
	virtual KInt fetch(KInt* arr, KInt nIndex = 0) = 0;
	virtual KFloat fetch(KaiArray<KFloat> arr, KInt nIndex = 0) = 0;
	virtual KFloat fetch(KFloat* arr, KInt nIndex = 0) = 0;

	virtual KaiArray<KFloat> sigmoid(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> sigmoid_derv_grad(KaiArray<KFloat> gsig, KaiArray<KFloat> sig) = 0;
	
	virtual KaiArray<KFloat> softmax(KaiArray<KFloat> arr) = 0;
	virtual KaiArray<KFloat> softmax_derv(KaiArray<KFloat> gyarr, KaiArray<KFloat> yarr) = 0;

	virtual KInt getDevAllocSize() = 0;

	virtual KaiList to_host(KaiList list) = 0;
	virtual KaiDict to_host(KaiDict dict) = 0;

	virtual KFloat mean(KaiList list) = 0;
	virtual KFloat sum(KaiList list) = 0;

	virtual KaiArray<KFloat> convolution(KaiArray<KFloat> xarr, KaiArray<KFloat> kernel) = 0;
	virtual KaiArray<KFloat> convolution_derv_x(KaiArray<KFloat> gyarr, KaiArray<KFloat> kernel) = 0;
	virtual KaiArray<KFloat> convolution_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> xarr, KaiShape kshape) = 0;

	virtual KaiArray<KFloat> subrange(KaiArray<KFloat> xarr, KInt nth_ax, KInt nFrom, KInt nCount) = 0;
	virtual KaiArray<KFloat> subrange_derv(KaiArray<KFloat> gyarr, KInt nth_ax, KInt nFrom, KInt nCount) = 0;

	virtual KaiArray<KFloat> stride(KaiArray<KFloat> xarr, KaiShape stride) = 0;
	virtual KaiArray<KFloat> stride_derv(KaiArray<KFloat> gyarr, KaiShape stride, KaiShape xshape) = 0;

	virtual KaiArray<KFloat> max_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pMaxMap, KaiShape kernel) = 0;
	virtual KaiArray<KFloat> max_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> maxMap, KaiShape kernel) = 0;

	virtual KaiArray<KFloat> avg_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pAvgCnt, KaiShape kernel) = 0;
	virtual KaiArray<KFloat> avg_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> avgCnt, KaiShape kernel) = 0;

	virtual KaiArray<KFloat> globalavg(KaiArray<KFloat> xarr) = 0;
	virtual KaiArray<KFloat> globalavg_derv(KaiArray<KFloat> gyarr, KaiShape xshape) = 0;

	virtual KaiArray<KFloat> BNCollectNorm(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KaiArray<KFloat>& var, KFloat momentum, KFloat epsilon) = 0;
	virtual KaiArray<KFloat> BnNormalize(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KFloat epsilon) = 0;
	virtual KaiArray<KFloat> BnScale(KaiArray<KFloat> xarr, KaiArray<KFloat> scale, KaiArray<KFloat> shift) = 0;
	virtual KaiArray<KFloat> BnNormDerv(KaiArray<KFloat> garr, KaiArray<KFloat> var, KFloat epsilon) = 0;

	virtual void rescale_derv_pm(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat>* p_grad_scale, KaiArray<KFloat>* p_grad_shift) = 0;
	virtual KaiArray<KFloat> rescale_derv_x(KaiArray<KFloat> garr, KaiArray<KFloat> scale) = 0;

	virtual void CopyIntoSlice(KaiArray<KFloat> yarr, KaiArray<KFloat> barr, KInt& nChnPos) = 0;
	virtual KaiArray<KFloat> CopyFromSlice(KaiArray<KFloat> garr, KInt& nChnPos, KInt nChnCnt) = 0;

	virtual KaiArray<KFloat> random_bernoulli(KaiShape xshape, KFloat one_ratio) = 0;
	virtual KaiArray<KFloat> dropout(KaiArray<KFloat> xarr, KaiArray<KFloat> mask, KFloat keep_raio) = 0;
	virtual KaiArray<KFloat> dropout_derv(KaiArray<KFloat> xarr, KaiArray<KFloat> mask, KFloat keep_raio) = 0;

	virtual void residual_add(KaiArray<KFloat> yarr, KaiArray<KFloat> xarr) = 0;
	virtual KaiArray<KFloat> residual_add_derv(KaiArray<KFloat> gyarr, KInt bchn) = 0;

	// 입력 기울기는 시간대 슬라이스에 복사 혹은 전체 배열에 덮어쓰기로 처리함
	// forward 처리 때 매 스텝 반복 입력 방식을 지원하게 되면 SplitExtendedInputGrad() 함수에 합산처리 방식의 확장 필요
	virtual KaiArray<KFloat> CombineExtendedInput(KaiArray<KFloat> recurrent, KBool isSeq, KaiArray<KFloat> xarr, KInt nth) = 0;
	virtual KaiArray<KFloat> SplitExtendedInputGrad(KaiArray<KFloat> g_exp_input, KBool isSeq, KaiArray<KFloat> g_x, KInt nth) = 0;
	virtual void CopyIntoTimeSlice(KaiArray<KFloat> yarr, KaiArray<KFloat> recurrent, KInt nth) = 0;
	virtual void add_time_slice_on_dest(KaiArray<KFloat> dest, KaiArray<KFloat> whole, KInt nth) = 0;

	virtual KaiArray<KFloat> lstm_gates(KaiArray<KFloat> affine) = 0;
	virtual KaiArray<KFloat> lstm_proc(KaiArray<KFloat> gates, KaiArray<KFloat>& state, KBool use_state) = 0;

	virtual KaiArray<KFloat> lstm_gates_derv(KaiArray<KFloat> g_gates, KaiArray<KFloat> gates) = 0;
	virtual KaiArray<KFloat> lstm_proc_derv(KaiArray<KFloat>& g_state, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_state, KaiArray<KFloat> post_recur, KBool use_state) = 0;

	virtual KaiArray<KFloat> gru_combine_extra(KaiArray<KFloat> exp_input, KaiArray<KFloat> gates) = 0;
	virtual void gru_combine_extra_derv(KaiArray<KFloat> g_exp_input, KaiArray<KFloat> g_gates, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> exp_input) = 0;
	virtual KaiArray<KFloat> gru_proc(KaiArray<KFloat> gates, KaiArray<KFloat> recurrent, KaiArray<KFloat> extra_affine) = 0;
	virtual KaiArray<KFloat> gru_proc_derv(KaiArray<KFloat>& g_gates, KaiArray<KFloat>& g_new_rec, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_recur, KaiArray<KFloat> extra_affine) = 0;

	virtual void add_embed_dict(KaiArray<KFloat> yarr, KaiArray<KInt> tokens, KaiArray<KFloat> word_vecs, KInt axis) = 0;

	virtual KaiList split_array(KaiArray<KFloat> arr, KInt piece_cnt) = 0;
	virtual KaiArray<KFloat> merge_array(KaiList arrs) = 0;

	virtual KaiArray<KFloat> multi_head_matmul_qk(KaiArray<KFloat> query, KaiArray<KFloat> key, KInt head_cnt) = 0;
	virtual KaiArray<KFloat> multi_head_matmul_qk_derv_q(KaiArray<KFloat> gyarr, KaiArray<KFloat> key, KInt head_cnt) = 0;
	virtual KaiArray<KFloat> multi_head_matmul_qk_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> query, KInt head_cnt) = 0;

	virtual KaiArray<KFloat> multi_head_matmul_pv(KaiArray<KFloat> probs, KaiArray<KFloat> value) = 0;
	virtual KaiArray<KFloat> multi_head_matmul_pv_derv_p(KaiArray<KFloat> gyarr, KaiArray<KFloat> value, KInt head_cnt) = 0;
	virtual KaiArray<KFloat> multi_head_matmul_pv_derv_v(KaiArray<KFloat> gyarr, KaiArray<KFloat> probs) = 0;

	virtual KaiArray<KFloat> extract(KaiArray<KFloat> xarr, KInt axis, KInt index, KInt count, KBool reduce_seq) = 0;
	virtual KaiArray<KFloat> extract_derv(KaiArray<KFloat> gyarr, KaiShape xshape, KInt axis, KInt index, KInt count, KBool reduce_seq) = 0;

	virtual KaiArray<KFloat> select(KaiArray<KFloat> xarr, KaiArray<KInt> selector, KaiShape vector_shape) = 0;
	virtual KaiArray<KFloat> select_derv(KaiArray<KFloat> garr, KaiArray<KInt> selector_arr, KaiShape xshape, KaiShape vshape) = 0;

	virtual void update_dic_weight_sgd(KaiArray<KFloat> weight, KaiArray<KFloat> grad, KaiArray<KInt> tokens, KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) = 0;
	virtual void update_dic_weight_adam(KaiArray<KFloat> weight, KaiArray<KFloat> s, KaiArray<KFloat> t, KaiArray<KFloat> n, KaiArray<KFloat> grad, KaiArray<KInt> tokens, KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay, KFloat ro1, KFloat ro2, KFloat epsilon) = 0;

	virtual KaiArray<KFloat> expand(KaiArray<KFloat> xarr, KaiShape ratio) = 0;
	virtual KaiArray<KFloat> expand_derv(KaiArray<KFloat> gyarr, KaiShape ratio) = 0;

	virtual KInt stack_on(KaiArray<KFloat> dest, KaiArray<KFloat> src, KInt tail_size, KInt nFrom, KInt nTo) = 0;
	virtual KaiArray<KFloat> stack_on_grad(KaiArray<KFloat> gyarr, KaiShape shape, KInt tail_size, KInt& nFrom, KInt nTo) = 0;

	virtual KaiArray<KFloat> get_subvector(KaiArray<KFloat> arr, KInt nStart, KInt nCount) = 0;
	virtual void get_subvector_derv_acc(KaiArray<KFloat> grad, KaiArray<KFloat> grad_subvec, KInt nStart, KInt nCount) = 0;

	virtual void fft(KFloat* pWave, KFloat* pFTT, KInt mb_size, KInt fetch_width, KInt step_width, KInt step_cnt, KInt fft_width, KInt freq_cnt) = 0;

protected:

};
