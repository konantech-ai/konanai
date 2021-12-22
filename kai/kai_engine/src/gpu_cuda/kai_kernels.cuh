/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <cuda_runtime.h>

#include "../session/kcommon.h"

#define KAI_ADD_RANGE       10000

__global__ void kai_ker_set(KInt size, KFloat* y, KFloat term);

__global__ void kai_ker_matmul(KInt size, KFloat* a, KFloat* h, KFloat* w, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_add_bias(KInt size, KFloat* y, KFloat* a, KFloat* b, KInt ncol);

__global__ void kai_ker_transpose(KInt size, KFloat* t, KFloat* a, KInt rows, KInt cols);
__global__ void kai_ker_sum_on_column(KInt size, KFloat* s, KFloat* a, KInt rows, KInt cols);

__global__ void kai_ker_activate(KInt size, KFloat* y, KFloat* x, KInt nFunc, KFloat leaky_alpha);
__global__ void kai_ker_activate_backprop(KInt size, KFloat* g_out, KFloat* g_in, KFloat* x, KFloat* y, KInt nFunc, KFloat leaky_alpha);

__global__ void kai_ker_sign(KInt size, KFloat* y, KFloat* a);
__global__ void kai_ker_square(KInt size, KFloat* y, KFloat* a);
__global__ void kai_ker_sqrt(KInt size, KFloat* y, KFloat* a);
__global__ void kai_ker_log(KInt size, KFloat* y, KFloat* a);
__global__ void kai_ker_exp(KInt size, KFloat* y, KFloat* a);
__global__ void kai_ker_sigmoid(KInt size, KFloat* y, KFloat* a);
__global__ void kai_ker_sigmoid_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* y_in);

__global__ void kai_ker_sum(KInt size, KFloat* x, KInt xsize, KInt range);
__global__ void kai_ker_mul_scalar_on(KInt size, KFloat* y, KFloat coef);

__global__ void kai_ker_minus(KInt size, KFloat* y, KFloat* a);

__global__ void kai_ker_binary_op(KInt size, KFloat* y, KFloat* a, KFloat* b, exp_op op_code, KInt vec_size1, KInt vec_size2);
__global__ void kai_ker_binary_op(KInt size, KFloat* y, KFloat* a, KFloat term, exp_op op_code);

__global__ void kai_ker_add(KInt size, KFloat* y, KFloat* a, KFloat* b);
__global__ void kai_ker_sub(KInt size, KFloat* y, KFloat* a, KFloat* b);
__global__ void kai_ker_mul(KInt size, KFloat* y, KFloat* a, KFloat* b);
__global__ void kai_ker_div(KInt size, KFloat* y, KFloat* a, KFloat* b);

__global__ void kai_ker_add(KInt size, KFloat* y, KFloat* a, KFloat term);
__global__ void kai_ker_sub(KInt size, KFloat* y, KFloat* a, KFloat term);
__global__ void kai_ker_mul(KInt size, KFloat* y, KFloat* a, KFloat term);
__global__ void kai_ker_div(KInt size, KFloat* y, KFloat* a, KFloat term);

__global__ void kai_ker_eval_adam_delta(KInt size, KFloat* d, KFloat* g, KFloat* s, KFloat* t, KFloat n, KFloat ro1, KFloat ro2, KFloat epsilon);
__global__ void kai_ker_apply_decay(KInt size, KFloat* d, KFloat* p, KFloat* g, KFloat l2_decay, KFloat l1_decay);

__global__ void kai_ker_softmax(KInt size, KFloat* y, KFloat* est, KInt vec);
__global__ void kai_ker_softmax_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* y_in, KInt nvec);

__global__ void kai_ker_softmax_cross_entropy_with_logits(KInt size, KFloat* y, KFloat* est, KFloat* ans, KInt nvec);

__global__ void kai_ker_softmax_cross_entropy_with_logits_idx(KInt size, KFloat* y_out, KFloat* est_in, KInt* ans_in, KInt nvec);
__global__ void kai_ker_softmax_cross_entropy_with_logits_idx_derv(KInt size, KFloat* y_out, KFloat* est_in, KInt* ans_in, KInt nvec);

__global__ void kai_ker_equal_row(KInt size, KFloat* y, KFloat* est, KFloat* ans, KInt nvec);
__global__ void kai_ker_max_row(KInt size, KFloat* y, KFloat* x, KInt nvec);
__global__ void kai_ker_max_row_derv(KInt size, KFloat* gx, KFloat* x, KFloat* gy, KInt nvec);

__global__ void kai_ker_vstack(KInt size, KFloat* dst, KFloat* src, KInt ncol, KInt nfrom, KInt nvec);
__global__ void kai_ker_vstack_derv(KInt size, KFloat* dst, KFloat* src, KInt ncol, KInt nfrom, KInt nvec);

__global__ void kai_ker_iou(KInt size, KFloat* piou, KFloat* pa1, KFloat* pa2, KInt col1, KInt col2);
//__global__ void kai_ker_iou_grad(KInt size, KFloat* pgx, KFloat* pgiou, KFloat* pa2, KInt col1, KInt col2);

__global__ void kai_ker_argmax(KInt size, KFloat* y, KFloat* a, KInt nvec);
__global__ void kai_ker_maxcol(KInt size, KFloat* y, KFloat* a, KInt nvec);

__global__ void kai_ker_conv_kernel(KInt size, KFloat* b_out, KFloat* x_in, KFloat* pm_in, KInt xh, KInt xw, KInt kh, KInt kw, KInt xchn, KInt ychn);
__global__ void kai_ker_conv_sum(KInt size, KFloat* y_out, KFloat* b_in, KInt xchn);
__global__ void kai_ker_conv_add_bias(KInt size, KFloat* y_out, KFloat* pm_in, KInt ychn);

__global__ void kai_ker_conv_derv_x_kernel(KInt size, KFloat* c_out, KFloat* gy_in, KFloat* k_in, KInt xh, KInt xw, KInt kh, KInt kw, KInt xchn, KInt ychn);
__global__ void kai_ker_conv_derv_x_sum(KInt size, KFloat* gx_out, KFloat* c_in, KInt ychn);

__global__ void kai_ker_conv_derv_kw_x(KInt size, KFloat* d_out, KFloat* gy_in, KFloat* x_in, KInt mb_size, KInt xh, KInt xw, KInt kh, KInt kw, KInt xchn, KInt ychn);
__global__ void kai_ker_conv_derv_kw_sum1(KInt size, KFloat* d_inout, KInt xh);
__global__ void kai_ker_conv_derv_kw_sum2(KInt size, KFloat* gw_out, KFloat* d_in, KInt mb_size, KInt xh);

__global__ void kai_ker_max(KInt size, KFloat* y_out, KInt* n_out, KFloat* x_in, KInt xh, KInt xw, KInt xchn, KInt kh, KInt kw);
__global__ void kai_ker_max_derv(KInt size, KFloat* gx_out, KInt* n_in, KFloat* gy_in, KInt xh, KInt xw, KInt xchn, KInt kh, KInt kw);

__global__ void kai_ker_avg(KInt size, KFloat* y_out, KInt* n_out, KFloat* x_in, KInt xh, KInt xw, KInt xchn, KInt kh, KInt kw);
__global__ void kai_ker_avg_derv(KInt size, KFloat* gx_out, KInt* n_in, KFloat* gy_in, KInt xh, KInt xw, KInt xchn, KInt kh, KInt kw);

__global__ void kai_ker_globalavg(KInt size, KFloat* y, KFloat* x, KInt xh, KInt xw, KInt chn);
__global__ void kai_ker_globalavg_derv(KInt size, KFloat* gx, KFloat* gy, KInt xh, KInt xw, KInt chn);

__global__ void kai_ker_stride(KInt size, KFloat* y, KFloat* x, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw);
__global__ void kai_ker_stride_derv(KInt size, KFloat* gx, KFloat* gy, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw);

__global__ void kai_ker_bn_collect(KInt bsize, KFloat* avg, KFloat* var, KFloat* mavg, KFloat* mvar, KFloat* x, KInt hsize, KFloat momentum);
__global__ void kai_ker_bn_normalize(KInt hsize, KFloat* y, KFloat* x, KFloat* avg, KFloat* var, KInt bsize, KFloat epsilon);
__global__ void kai_ker_bn_rescale(KInt hsize, KFloat* y, KFloat* x, KFloat* scale, KFloat* shift, KInt bsize);

__global__ void kai_ker_bn_rescale_derv_pm(KInt bsize, KFloat* gscale_out, KFloat* gshift_out, KFloat* gx_in, KFloat* x_in, KInt hsize);
__global__ void kai_ker_bn_rescale_derv_x(KInt hsize, KFloat* gx_out, KFloat* gy_in, KFloat* scale, KInt bsize);
__global__ void kai_ker_bn_norm_derv(KInt hsize, KFloat* gh_inout, KFloat* var_in, KInt bsize, KFloat epsilon);

__global__ void kai_ker_put_branch(KInt size, KFloat* y, KFloat* b, KInt ychn, KInt bchn, KInt chn_from);
__global__ void kai_ker_get_branch(KInt size, KFloat* gb, KFloat* gy, KInt ychn, KInt bchn, KInt chn_from);

__global__ void kai_ker_dropout(KInt size, KFloat* y_out, KFloat* x_in, KFloat* m_in, KFloat keep_ratio);
__global__ void kai_ker_dropout_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* m_in, KFloat keep_ratio);

__global__ void kai_ker_tile_chn(KInt size, KFloat* t_out, KFloat* x_in, KInt ychn, KInt xchn);
__global__ void kai_ker_untile_chn(KInt size, KFloat* gt_out, KFloat* gy_in, KInt ychn, KInt xchn);

__global__ void kai_ker_rnn_combine_ex_inp(KInt size, KFloat* ex_inp_out, KFloat* x_in, KFloat* rec_in, KInt timesteps, KInt timefeats, KInt recur_size, bool inseq, KInt tn);
__global__ void kai_ker_rnn_split_ex_inp(KInt size, KFloat* gx_out, KFloat* grec_out, KFloat* gex_inp_in, KInt timesteps, KInt timefeats, KInt recur_size, bool inseq, KInt tn);

__global__ void kai_ker_rnn_fill_output_slice(KInt size, KFloat* y_out, KFloat* x_in, KInt timesteps, KInt recur_size, KInt tn);
__global__ void kai_ker_rnn_add_time_slice(KInt size, KFloat* gy_out, KFloat* gx_in, KInt timesteps, KInt recur_size, KInt tn);

__global__ void kai_ker_lstm_gate(KInt size, KFloat* gates, KFloat* affine);
__global__ void kai_ker_lstm_proc(KInt size, KFloat* recur_out, KFloat* state_out, KFloat* state_in, KFloat* gates_in);
__global__ void kai_ker_lstm_gate_derv(KInt size, KFloat* g_affine, KFloat* g_gates, KFloat* gates);
__global__ void kai_ker_lstm_proc_derv(KInt size, KFloat* g_gates_out, KFloat* g_state_out, KFloat* g_state_in, KFloat* g_recur_in, KFloat* gates_in, KFloat* state_in, KFloat* rec_in);

__global__ void kai_ker_gru_combine_extra(KInt size, KFloat* cuda_x2, KFloat* cuda_x1, KFloat* cuda_gt, KInt ext_size, KInt inp_size, KInt rec_size);
__global__ void kai_ker_gru_combine_extra_derv(KInt size, KFloat* g_gates, KFloat* g_ext_inout, KFloat* ext_in, KFloat* gates, KInt ext_size, KInt inp_size, KInt rec_size);
__global__ void kai_ker_gru_proc(KInt size, KFloat* cuda_r2, KFloat* cuda_r1, KFloat* cuda_gt, KFloat* cuda_in);
__global__ void kai_ker_gru_proc_derv(KInt size, KFloat* g_inp, KFloat* g_gate, KFloat* g_rec_out, KFloat* g_rec_in, KFloat* pre_rec, KFloat* gate, KFloat* inp);

__global__ void kai_ker_wave_slices_to_complex(KInt size, KFloat* c_out, KFloat* w_in, KInt step_width, KInt step_cnt, KInt fft_width, KInt fetch_width);
__global__ void kai_ker_fft_step_split(KInt size, KFloat* dst_out, KFloat* src_in, KInt data_size, KInt step, KInt nd_base);
__global__ void kai_ker_complex_to_abs_mean(KInt size, KFloat* a_out, KFloat* c_in, KInt data_num, KInt freq_cnt);

__global__ void kai_ker_add_embed_dict(KInt size, float* y_out, float* dic_in, KInt* token_in, KInt vec_size, KInt dic_kind, KInt axis);
__global__ void kai_kernel_sgd_update_embed(KInt size, KFloat* w, KFloat* grads, KInt* tokens, KInt dic_cnt, KInt nth, KInt dic_size, KInt vec_size, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);
__global__ void kai_kernel_adam_update_embed(KInt size, KFloat* w, KFloat* s, KFloat* t, KFloat* n, KFloat* grads, KInt* tokens, KInt dic_cnt, KInt nth, KInt dic_size, KInt vec_size,
    KFloat learning_rate, KFloat l2_decay, KFloat l1_decay, KFloat ro1, KFloat ro2, KFloat epsilon);

__global__ void kai_ker_split_array(KInt size, KFloat* piece_0, KFloat* piece_1, KFloat* piece_2, KFloat* piece_3, KFloat* x_in, KInt piece_cnt);
__global__ void kai_ker_merge_array(KInt size, KFloat* y_out, KFloat* piece_0, KFloat* piece_1, KFloat* piece_2, KFloat* piece_3, KInt piece_cnt);

__global__ void kai_ker_multi_head_matmul_qk(KInt size, KFloat* r_out, KFloat* q_in, KFloat* k_in, KInt timesteps, KInt head_cnt, KInt vec_size);
__global__ void kai_ker_multi_head_matmul_qk_derv_q(KInt size, KFloat* gq_out, KFloat* gy_in, KFloat* k_in, KInt timesteps, KInt head_cnt, KInt vec_per_head);
__global__ void kai_ker_multi_head_matmul_qk_dev_k(KInt size, KFloat* gk_out, KFloat* gy_in, KFloat* q_in, KInt timesteps, KInt head_cnt, KInt vec_per_head);

__global__ void kai_ker_multi_head_matmul_pv(KInt size, KFloat* r_out, KFloat* p_in, KFloat* v_in, KInt timesteps, KInt head_cnt, KInt vec_size);
__global__ void kai_ker_multi_head_matmul_pv_derv_p(KInt size, KFloat* gp_out, KFloat* gy_in, KFloat* v_in, KInt timesteps, KInt head_cnt, KInt vec_per_head);
__global__ void kai_ker_multi_head_matmul_pv_derv_v(KInt size, KFloat* gv_out, KFloat* gy_in, KFloat* p_in, KInt timesteps, KInt head_cnt, KInt vec_per_head);

__global__ void kai_ker_extract(KInt size, KFloat* e_out, KFloat* x_in, KInt ax_size, KInt index, KInt nCount, KInt nProd);
__global__ void kai_ker_extract_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt ax_size, KInt index, KInt nCount, KInt nProd);

__global__ void kai_ker_mask_to_idx(KInt size, KInt* p_map, KInt* p_mask, KInt msize);
__global__ void kai_ker_filter(KInt size, KFloat* filtered_out, KFloat* x_in, KInt* map_in, KInt vec_size);
__global__ void kai_ker_filter_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt* map_in, KInt vec_size);

__global__ void kai_ker_expand(KInt size, KFloat* y_out, KFloat* x_in, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio);
__global__ void kai_ker_expand_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio);

__global__ void kai_ker_stack_on(KInt size, KFloat* cuda_d, KFloat* cuda_s, KInt block_size, KInt region_size, KInt tail_size, KInt nFrom);
__global__ void kai_ker_stack_on_grad(KInt size, KFloat* cuda_gx, KFloat* cuda_gy, KInt block_size, KInt region_size, KInt tail_size, KInt nFrom);

__global__ void kai_ker_get_subvector(KInt size, KFloat* cuda_s, KFloat* cuda_a, KInt vec_size, KInt nStart, KInt nCount);
__global__ void kai_ker_acc_grad_subvector(KInt size, KFloat* cuda_gy, KFloat* cuda_gs, KInt vec_size, KInt nStart, KInt nCount);

/*
__global__ void kai_ker_init(KInt size, KFloat* y_out, KFloat value);
__global__ void kai_ker_init_int(KInt size, KInt* y_out, KInt value);
__global__ void kai_ker_init_int64(KInt size, KInt* y_out, KInt value);
__global__ void kai_ker_copy_to(KInt size, KFloat* y_out, KFloat* x_in);
__global__ void kai_ker_add_on(KInt size, KFloat* y_out, KFloat* x_in);
__global__ void kai_ker_mult_to(KInt size, KFloat* y_out, KFloat* x1_in, KFloat* x2_in);
__global__ void kai_ker_mult_on(KInt size, KFloat* y_inout, KFloat* x_in);
__global__ void kai_ker_mult_scalar_to(KInt size, KFloat* y_inout, KFloat* x1_in, KFloat coef);
__global__ void kai_ker_mult_scalar_on(KInt size, KFloat* y_inout, KFloat coef);
__global__ void kai_ker_sigmoid_on(KInt size, KFloat* x_inout);
__global__ void kai_ker_tanh_on(KInt size, KFloat* x_inout);
__global__ void kai_ker_tanh_to(KInt size, KFloat* y_out, KFloat* x_in);
__global__ void kai_ker_sigmoid_derv_on(KInt size, KFloat* gx_inout, KFloat* y_in);
__global__ void kai_ker_tanh_derv_on(KInt size, KFloat* gx_inout, KFloat* y_in);
__global__ void kai_ker_abs(KInt size, KFloat* y_out, KFloat* x_in);
__global__ void kai_ker_near_zero(KInt size, KFloat* y_out, KFloat* x_in, KFloat threshold);
__global__ void kai_ker_rescale(KInt size, KFloat* x_inout, KFloat mean, KFloat std);
__global__ void kai_ker_sum(KInt size, KFloat* x_inout, KInt xsize, KInt range);
__global__ void kai_ker_sum_rows(KInt size, KFloat* x_inout, KInt xsize, KInt range, KInt ncol);
__global__ void kai_ker_binomial(KInt size, KFloat* x_inout, KFloat prob_threshod);
__global__ void kai_ker_argmax(KInt size, KInt* n_out, KFloat* arr_in, KInt nvec);
__global__ void kai_ker_get_hash_idx(KInt size, KInt* n_out, KFloat* arr_in, KInt nvec);
__global__ void kai_ker_get_hash_diff(KInt size, KInt* n_out, KFloat* h1_in, KFloat* h2_in, KInt nvec);

//__global__ void kai_ker_forward_postproc(KInt size, KFloat* cuda_res, KFloat* cuda_aux, KFloat* cuda_est, KFloat* cuda_ans, KInt enumMod, KInt nvec);
__global__ void kai_ker_mse(KInt size, KFloat* y_out, KFloat* est_in, KFloat* ans_in);
__global__ void kai_ker_sigmoid_cross_entropy_with_logits(KInt size, KFloat* y_out, KFloat* est_in, KFloat* ans_in);
__global__ void kai_ker_softmax_cross_entropy_with_logits_idx(KInt size, KFloat* y_out, KFloat* est_in, KInt* ans_in, KInt nvec);
__global__ void kai_ker_softmax_cross_entropy_with_logits_1st(KInt size, KFloat* y_out, KFloat* est_in, KInt nvec);

__global__ void kai_ker_mult_diff_coef(KInt size, KFloat* y_out, KFloat* est_in, KFloat* ans_in, KFloat coef);
__global__ void kai_ker_bin_acc(KInt size, KFloat* y_out, KFloat* x1_in, KFloat* x2_in);
__global__ void kai_ker_class_acc(KInt size, KFloat* y_out, KFloat* est_logit_in, KFloat* ans_prob_in, KInt nvec);
__global__ void kai_ker_class_idx_acc(KInt size, KFloat* y_out, KFloat* est_logit_in, KInt* ans_prob_in, KInt nvec);
__global__ void kai_ker_class_1st_acc(KInt size, KFloat* y_out, KFloat* est_logit_in, KInt nvec);
__global__ void kai_ker_mse_diff_sq(KInt size, KFloat* y_out, KFloat* x1_in, KFloat* x2_in);

__global__ void kai_ker_matmul(KInt size, KFloat* a_out, KFloat* h_in, KFloat* w_in, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_multi_matmul(KInt size, KFloat* a_out, KFloat* h_in, KFloat* w_in, KInt msize, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_add_bias(KInt size, KFloat* a_inout, KFloat* b_in, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_matmul_derv_x(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* w_in, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_matmul_derv_w(KInt size, KFloat* gw_out, KFloat* gy_in, KFloat* x_in, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_multi_matmul_derv_x(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* w_in, KInt mb_size, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_multi_matmul_derv_w(KInt size, KFloat* gw_out, KFloat* gy_in, KFloat* x_in, KInt mb_size, KInt nrow, KInt nvec, KInt ncol);
__global__ void kai_ker_add_bias_derv(KInt size, KFloat* gb_out, KFloat* gy_in, KInt nrow, KInt nvec, KInt ncol);

__global__ void kai_ker_update_param_sgd(KInt size, KFloat* pm_inout, KFloat* gpm_in, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);
__global__ void kai_ker_update_param_adam(KInt size, KFloat* pm_inout, KFloat* s_inout, KFloat* t_inout, KFloat* gpm_in, KFloat ro1, KFloat ro2, KInt nstep, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);

__global__ void kai_ker_update_param_sgd_select(KInt size, KFloat* pm_inout, KInt* wid_in, KFloat* gpm_in, KInt word_cnt, KInt vec_size, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);
__global__ void kai_ker_update_param_adam_select(KInt size, KFloat* pm_inout, KFloat* s_inout, KFloat* t_inout, KFloat* n_inout, KInt* wid_in, KFloat* gpm_in, KInt word_cnt, KInt vec_size, KFloat ro1, KFloat ro2, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);

__global__ void kai_ker_update_param_dup_count(KInt size, KFloat* delta_out, KFloat* count_out, KInt* wid_in, KFloat* gpm_in, KInt dic_count, KInt vec_size);
__global__ void kai_ker_update_param_sgd_select_multi_dic(KInt size, KFloat* pm_inout, KInt* wid_in, KFloat* gpm_in, KInt dic_count, KInt* voc_counts, KInt vec_size, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);
__global__ void kai_ker_update_param_adam_select_multi_dic(KInt size, KFloat* pm_inout, KFloat* s_inout, KFloat* t_inout, KFloat* n_inout, KInt* wid_in, KFloat* delta_in, KFloat* count_in, KInt dic_count, KInt* voc_counts, KInt vec_size, KFloat ro1, KFloat ro2, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay);

__global__ void kai_ker_update_affine_param(KInt size, KFloat* gpm_in, KFloat* wp_inout, KFloat* bp_inout, KInt nrows, KInt ncols);

__global__ void kai_ker_sigmoid(KInt size, KFloat* y_out, KFloat* x_in);
__global__ void kai_ker_sigmoid_cross_entropy_with_logits_derv(KInt hsize, KFloat* y_out, KFloat* est_logit_in, KFloat* ans_prob_in, KFloat coef);

__global__ void kai_ker_softmax(KInt size, KFloat* y_out, KFloat* x_in, KInt nvec);
__global__ void kai_ker_softmax_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* y_in, KInt nvec);
__global__ void kai_ker_softmax_cross_entropy_with_logits_derv(KInt hsize, KFloat* y_out, KFloat* est_logit_in, KFloat* ans_prob_in, KInt nvec, KFloat coef);
__global__ void kai_ker_softmax_cross_entropy_with_logits_idx_derv(KInt hsize, KFloat* y_out, KFloat* est_logit_in, KInt* ans_prob_in, KInt nvec, KFloat coef);
__global__ void kai_ker_softmax_cross_entropy_with_logits_1st_derv(KInt hsize, KFloat* y_out, KFloat* est_logit_in, KInt nvec, KFloat coef);

__global__ void kai_ker_dropout_old(KInt size, KFloat* x_in, KFloat* dm_in, KFloat* y_out, KInt dm_size, KFloat keep_ratio);
__global__ void kai_ker_backprop_dropout_old(KInt size, KFloat* gy_in, KFloat* dm_in, KFloat* gx_out, KInt dm_size, KFloat keep_ratio);

__global__ void kai_ker_activate(KInt size, KFloat* y_out, KFloat* x_in, KInt nFunc, KFloat alpha);
__global__ void kai_ker_activate_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* x_in, KFloat* y_in, KInt nFunc, KFloat alpha);


__global__ void kai_ker_conv_derv_kb_sum1(KInt size, KFloat* d_out, KFloat* gy_in, KInt mb_size, KInt xh, KInt xw, KInt ychn);
__global__ void kai_ker_conv_derv_kb_sum2(KInt size, KFloat* d_inout, KInt mb_size, KInt xh, KInt xw, KInt ychn);
__global__ void kai_ker_conv_derv_kb_sum3(KInt size, KFloat* gb_out, KFloat* d_in, KInt mb_size, KInt xh, KInt xw, KInt ychn);


__global__ void kai_ker_avg_exact(KInt size, KFloat* t_out, KFloat* x_in, KInt xh, KInt xw, KInt xchn, KInt sh, KInt sw);
__global__ void kai_ker_avg_exact_derv(KInt size, KFloat* gt_out, KFloat* gy_in, KInt xh, KInt xw, KInt xchn, KInt sh, KInt sw);

__global__ void kai_ker_stride(KInt size, KFloat* y_out, KFloat* x_in, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt kh, KInt kw, KInt sh, KInt sw, bool same_padding);
__global__ void kai_ker_stride_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt kh, KInt kw, KInt sh, KInt sw, bool same_padding);

__global__ void kai_ker_stride_expand(KInt size, KFloat* y_out, KFloat* x_in, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw);
__global__ void kai_ker_stride_expand_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw);

__global__ void kai_ker_batch_normal(KInt size, KFloat* y_out, KFloat* x_in, KInt ngroup);
__global__ void kai_ker_batch_normal_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* y_in, KInt ngroup);

__global__ void kai_ker_bn_rescale_derv_pm(KInt bsize, KFloat* gscale_out, KFloat* gshift_out, KFloat* gx_in, KFloat* x_in, KInt hsize);
__global__ void kai_ker_bn_rescale_derv_x(KInt hsize, KFloat* gx_out, KFloat* gy_in, KFloat* scale, KInt bsize);
__global__ void kai_ker_bn_norm_derv(KInt hsize, KFloat* gh_inout, KFloat* var_in, KInt bsize, KFloat epsilon);

__global__ void kai_ker_get_branch(KInt bsize, KFloat* y_out, KFloat* b_in, KInt ychn, KInt bchn, KInt chn_from);
__global__ void kai_ker_set_branch(KInt bsize, KFloat* gb_out, KFloat* gy_in, KInt ychn, KInt bchn, KInt chn_from);

__global__ void kai_ker_rnn_fill_output_slice(KInt size, KFloat* y_out, KFloat* x_in, KInt timesteps, KInt recur_size, KInt tn);
__global__ void kai_ker_rnn_add_time_slice(KInt size, KFloat* gy_out, KFloat* gx_in, KInt timesteps, KInt recur_size, KInt tn);
__global__ void kai_ker_rnn_copy_last_grad(KInt size, KFloat* grec_inout, KFloat* gx_in, KInt timesteps, KInt recur_size);

__global__ void kai_ker_lstm_split_affine(KInt size, KFloat* fgate_out, KFloat* igate_out, KFloat* ogate_out, KFloat* block_out, KFloat* affine_in, KInt recur_size);
__global__ void kai_ker_lstm_combine_affine(KInt size, KFloat* gaffine_out, KFloat* gfgate_in, KFloat* gigate_in, KFloat* gogate_in, KFloat* gblock_in, KInt recur_size);
__global__ void kai_ker_rnn_select_last_vecs(KInt size, KFloat* selected_out, KFloat* time_pool_in, KInt timesteps, KInt nvec);

__global__ void kai_ker_lstm_new_state(KInt size, KFloat* state_out, KFloat* state_in, KFloat* fgate_in, KFloat* block_in, KFloat* igate_in);
__global__ void kai_ker_lstm_state_derv(KInt size, KFloat* gstate_inout, KFloat* grec_in, KFloat* ogate_in, KFloat* rec_in);

__global__ void kai_ker_attention_split(KInt size, KFloat* q_out, KFloat* k_out, KFloat* v_out, KFloat* qkv_in, KInt L, KInt H, KInt R);
__global__ void kai_ker_attention_combine(KInt size, KFloat* gqkv_out, KFloat* gq_in, KFloat* gk_in, KFloat* gv_in, KInt L, KInt H, KInt R);

__global__ void kai_ker_attention_mask_future(KInt size, KFloat* score_inout, KInt L);

__global__ void kai_ker_attention_reshape_out(KInt size, KFloat* a_out, KFloat* m_in, KInt L, KInt H, KInt R);
__global__ void kai_ker_attention_reshape_mul(KInt size, KFloat* gm_out, KFloat* ga_in, KInt L, KInt H, KInt R);

__global__ void kai_ker_embedding_fetch(KInt size, KFloat* wiv_out, KFloat* wov_out, KInt* hint_in, KInt* noms_in, KFloat* iwdic_in, KFloat* owdic_in, KInt in_cnt, KInt out_cnt, KInt vec_size);

__global__ void kai_ker_embedding_dotmul(KInt size, KFloat* y_out, KFloat* sub_in, KFloat* wov_in, KInt vec_cnt, KInt vec_size);
__global__ void kai_ker_embedding_dotmul_derv(KInt size, KFloat* gsub_out, KFloat* gwov_out, KFloat* gy_in, KFloat* sub_in, KFloat* wov_in, KInt vec_cnt, KInt vec_size);

__global__ void kai_ker_merge_avg(KInt size, KFloat* m_out, KFloat* x_in, KInt vec_cnt, KInt vec_size);
__global__ void kai_ker_merge_avg_derv(KInt size, KFloat* gx_out, KFloat* gm_in, KInt vec_cnt, KInt vec_size);

__global__ void kai_ker_extract(KInt size, KFloat* e_out, KFloat* x_in, KInt ax_size, KInt index, KInt nprod);
__global__ void kai_ker_unextract(KInt size, KFloat* gx_out, KFloat* ge_in, KInt ax_size, KInt index, KInt nprod);

__global__ void kai_ker_embed_fetch_multi_dic(KInt size, KFloat* v_out, KInt* wids_in, KFloat* dics_in, KInt dic_count, KInt* voc_counts, KInt vec_size);

__global__ void kai_ker_vstack(KInt size, KFloat* vs_out, KFloat* x1_in, KFloat* x2_in, KInt vol1, KInt vol2, KInt rest);
__global__ void kai_ker_hstack(KInt size, KFloat* vs_out, KFloat* x1_in, KFloat* x2_in, KInt vec1, KInt vec2);

__global__ void kai_ker_hsplit(KInt size, KFloat* p1_out, KFloat* p2_out, KFloat* src_in, KInt vec_size, KInt p1_size);

__global__ void kai_ker_extract_selected_pickup(KInt size, KFloat* dst_out, KFloat* arr_in, KInt* map_in, KInt drest);
__global__ void kai_ker_extract_selected_pickup_int(KInt size, KInt* dst_out, KInt* arr_in, KInt* map_in, KInt drest);
__global__ void kai_ker_extract_selected_fill(KInt size, KFloat* arr_inout, KFloat* slice_in, KInt* map_in, KInt drest);

__global__ void kai_ker_expand(KInt size, KFloat* y_out, KFloat* x_in, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio);
__global__ void kai_ker_expand_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio);

// (ndata, image_id, width, height) for images
#define IMG_INFO_SIZE       4

__device__ inline KInt img_ndata(KInt* np, KInt n) { return np[n * IMG_INFO_SIZE + 0]; }
__device__ inline KInt img_imageid(KInt* np, KInt n) { return np[n * IMG_INFO_SIZE + 1]; }
__device__ inline KInt img_width(KInt* np, KInt n) { return np[n * IMG_INFO_SIZE + 2]; }
__device__ inline KInt img_height(KInt* np, KInt n) { return np[n * IMG_INFO_SIZE + 3]; }

// (nimginfo, cat_id, center_x, center_y, width, height, mixed)
#define BOX_INFO_SIZE   7

__device__ inline KInt box_nimg(KFloat* fp, KInt n) { return (KInt)fp[n * BOX_INFO_SIZE + 0]; }
__device__ inline KInt box_catid(KFloat* fp, KInt n) { return (KInt)fp[n * BOX_INFO_SIZE + 1]; }

__device__ inline KFloat box_cx(KFloat* fp, KInt n) { return fp[n * BOX_INFO_SIZE + 2]; }
__device__ inline KFloat box_cy(KFloat* fp, KInt n) { return fp[n * BOX_INFO_SIZE + 3]; }
__device__ inline KFloat box_wd(KFloat* fp, KInt n) { return fp[n * BOX_INFO_SIZE + 4]; }
__device__ inline KFloat box_ht(KFloat* fp, KInt n) { return fp[n * BOX_INFO_SIZE + 5]; }
__device__ inline KFloat box_mixed(KFloat* fp, KInt n) { return fp[n * BOX_INFO_SIZE + 6]; }

__device__ inline KFloat box_rect(KFloat* fp, KInt n, KInt nc) { return fp[n * BOX_INFO_SIZE + nc + 2]; }

// (box_idx, nanchor, cx, cy)
#define SCALE_BOX_SIZE  4

__device__ inline KInt scale_nbox(KInt* np, KInt n) { return np[n * SCALE_BOX_SIZE + 0]; }
__device__ inline KInt scale_nx(KInt* np, KInt n) { return np[n * SCALE_BOX_SIZE + 1]; }
__device__ inline KInt scale_ny(KInt* np, KInt n) { return np[n * SCALE_BOX_SIZE + 2]; }
__device__ inline KInt scale_nanchor(KInt* np, KInt n) { return np[n * SCALE_BOX_SIZE + 3]; }

__device__ inline void set_scale_nbox(KInt* np, KInt n, KInt v) { np[n * SCALE_BOX_SIZE + 0] = (KInt)v; }
__device__ inline void set_scale_nx(KInt* np, KInt n, KInt v) { np[n * SCALE_BOX_SIZE + 1] = (KInt)v; }
__device__ inline void set_scale_ny(KInt* np, KInt n, KInt v) { np[n * SCALE_BOX_SIZE + 2] = (KInt)v; }
__device__ inline void set_scale_nanchor(KInt* np, KInt n, KInt v) { np[n * SCALE_BOX_SIZE + 3] = (KInt)v; }

// (cx, cy, wd, ht, conf, class0, ..., class79)
#define FMAP_SIZE  85

__device__ inline KFloat fmap_cx(KFloat* fp, KInt n) { return fp[n * FMAP_SIZE + 0]; }
__device__ inline KFloat fmap_cy(KFloat* fp, KInt n) { return fp[n * FMAP_SIZE + 1]; }
__device__ inline KFloat fmap_wd(KFloat* fp, KInt n) { return fp[n * FMAP_SIZE + 2]; }
__device__ inline KFloat fmap_ht(KFloat* fp, KInt n) { return fp[n * FMAP_SIZE + 3]; }
__device__ inline KFloat fmap_conf(KFloat* fp, KInt n) { return fp[n * FMAP_SIZE + 4]; }
__device__ inline KFloat fmap_class(KFloat* fp, KInt n, KInt cls) { return fp[n * FMAP_SIZE + 5 + cls]; }
__device__ inline KFloat fmap_elem(KFloat* fp, KInt n, KInt nc) { return fp[n * FMAP_SIZE + nc]; }

// (cx, cy, wd, ht, conf, class0, ..., class79)
#define PRED_SIZE  85

__device__ inline KFloat pred_xmin(KFloat* fp, KInt n) { return fp[n * PRED_SIZE + 0]; }
__device__ inline KFloat pred_ymin(KFloat* fp, KInt n) { return fp[n * PRED_SIZE + 1]; }
__device__ inline KFloat pred_xmax(KFloat* fp, KInt n) { return fp[n * PRED_SIZE + 2]; }
__device__ inline KFloat pred_ymax(KFloat* fp, KInt n) { return fp[n * PRED_SIZE + 3]; }
__device__ inline KFloat pred_conf(KFloat* fp, KInt n) { return fp[n * PRED_SIZE + 4]; }
__device__ inline KFloat pred_class(KFloat* fp, KInt n, KInt cls) { return fp[n * PRED_SIZE + 5 + cls]; }

__device__ inline void set_pred_xmin(KFloat* fp, KInt n, KFloat v) { fp[n * PRED_SIZE + 0] = v; }
__device__ inline void set_pred_ymin(KFloat* fp, KInt n, KFloat v) { fp[n * PRED_SIZE + 1] = v; }
__device__ inline void set_pred_xmax(KFloat* fp, KInt n, KFloat v) { fp[n * PRED_SIZE + 2] = v; }
__device__ inline void set_pred_ymax(KFloat* fp, KInt n, KFloat v) { fp[n * PRED_SIZE + 3] = v; }
__device__ inline void set_pred_conf(KFloat* fp, KInt n, KFloat v) { fp[n * PRED_SIZE + 4] = v; }
__device__ inline void set_pred_class(KFloat* fp, KInt n, KInt cls, KFloat v) { fp[n * PRED_SIZE + 5 + cls] = v; }

// [0:4] box rect, 4: score, 5: class_num, 6: ndata, 7:flag
#define RES_BOX_SIZE   8

__device__ inline KFloat res_xmin(KFloat* fp, KInt n) { return fp[n * RES_BOX_SIZE + 0]; }
__device__ inline KFloat res_ymin(KFloat* fp, KInt n) { return fp[n * RES_BOX_SIZE + 1]; }
__device__ inline KFloat res_xmax(KFloat* fp, KInt n) { return fp[n * RES_BOX_SIZE + 2]; }
__device__ inline KFloat res_ymax(KFloat* fp, KInt n) { return fp[n * RES_BOX_SIZE + 3]; }
__device__ inline KFloat res_score(KFloat* fp, KInt n) { return fp[n * RES_BOX_SIZE + 4]; }

__device__ inline KInt res_class(KFloat* fp, KInt n) { return (KInt)fp[n * RES_BOX_SIZE + 5]; }
__device__ inline KInt res_ndata(KFloat* fp, KInt n) { return (KInt)fp[n * RES_BOX_SIZE + 6]; }
__device__ inline KInt res_flag(KFloat* fp, KInt n) { return (KInt)fp[n * RES_BOX_SIZE + 7]; }

__device__ inline void set_res_xmin(KFloat* fp, KInt n, KFloat v) { fp[n * RES_BOX_SIZE + 0] = v; }
__device__ inline void set_res_ymin(KFloat* fp, KInt n, KFloat v) { fp[n * RES_BOX_SIZE + 1] = v; }
__device__ inline void set_res_xmax(KFloat* fp, KInt n, KFloat v) { fp[n * RES_BOX_SIZE + 2] = v; }
__device__ inline void set_res_ymax(KFloat* fp, KInt n, KFloat v) { fp[n * RES_BOX_SIZE + 3] = v; }
__device__ inline void set_res_score(KFloat* fp, KInt n, KFloat v) { fp[n * RES_BOX_SIZE + 4] = v; }

__device__ inline void set_res_class(KFloat* fp, KInt n, KInt v) { fp[n * RES_BOX_SIZE + 5] = (KFloat)v; }
__device__ inline void set_res_ndata(KFloat* fp, KInt n, KInt v) { fp[n * RES_BOX_SIZE + 6] = (KFloat)v; }
__device__ inline void set_res_flag(KFloat* fp, KInt n, KInt v) { fp[n * RES_BOX_SIZE + 7] = (KFloat)v; }

__device__ inline KFloat myfmax(KFloat x1, KFloat x2) { return (x1 > x2) ? x1 : x2; }
__device__ inline KFloat myfmin(KFloat x1, KFloat x2) { return (x1 < x2) ? x1 : x2; }

#define clip(x, lm, um) (((x) < um) ? (((x) > lm) ? (x) : lm) : um)

__global__ void kai_ker_yolo_eval_true_box_score(KInt size, KFloat* score_out, KFloat* boxes_in, KInt* anchors_in, KInt num_scales, KInt anchor_per_scale);
__global__ void kai_ker_yolo_eval_true_box_select(KInt size, KFloat* selected_out, KFloat* score_in, KInt num_scales, KInt anchor_per_scale);
__global__ void kai_ker_yolo_eval_true_count_selected(KInt size, KInt* selected_cnt_out, KFloat* selected_in, KInt dat_size, KInt num_scales, KInt anchor_per_scale);
__global__ void kai_ker_yolo_eval_true_lookup_scale_box(KInt size, KInt* box_scale_cood_out, KFloat* selected_in, KInt nscale, KInt dat_size, KInt num_scales, KInt anchor_per_scale);
__global__ void kai_ker_yolo_eval_true_eval_box_cood(KInt size, KInt* box_scale_cood_inout, KFloat* box_info_in, KInt grid_size);

__global__ void kai_ker_yolo_conv_fmap(KInt size, KFloat* pred_out, KFloat* fmap_in, KInt* anchors_in, KInt img_size, KInt grid_cnt, KInt anchors_cnt, KInt class_num);
__global__ void kai_ker_yolo_eval_iou(KInt size, KFloat* iou_out, KInt* boxid_out, KFloat* fmap_in, KInt* img_info_in, KFloat* box_info_in, KInt* scale_boxes_in, KInt* anchors_in,
    KInt nanchor, KInt img_size, KInt grid_cnt, KInt box_cnt);
__global__ void kai_ker_yolo_select_best_iou(KInt size, KInt* best_box_out, KFloat* best_iou_out, KFloat* iou_in, KInt* boxid_in, KInt box_cnt);

__global__ void kai_ker_yolo_eval_losses(KInt size, KFloat* loss_out, KFloat* fmap_in, KFloat* box_info_in, KInt* best_box_in, KFloat* best_iou_in, KInt* anchors_in,
    KInt mb_size, KInt img_size, KInt grid_cnt, KInt nanchor, KInt class_num, bool use_focal, bool smooth_onehot);
__global__ void kai_ker_yolo_eval_grads(KInt size, KFloat* grad_out, KFloat* fmap_in, KFloat* box_info_in, KInt* best_box_in, KFloat* best_iou_in, KInt* anchors_in,
    KInt mb_size, KInt img_size, KInt grid_cnt, KInt nanchor, KInt class_num, bool use_focal, bool smooth_onehot);

__global__ void kai_ker_yolo_select_pred_boxes(KInt size, unsigned char* flag_out, KInt* ndata_out, KFloat* conf_rects_out, KFloat* fmap_in, KInt* anchors_in, KFloat pred_conf_thr, KInt out_base, KInt img_size, KInt grid_cnt, KInt anchor_per_scale, KInt class_num);

__global__ void kai_ker_yolo_eval_predict_score(KInt size, KFloat* score_out, KFloat* pred_in, KInt class_num, KFloat score_thresh);
__global__ void kai_ker_yolo_get_boxes(KInt size, KFloat* boxes_out, KFloat* old_boxes_in, KInt* idxs_in, KFloat* score_in, KFloat* pred_in, KInt old_count, KInt grid_cnt, KInt anchors_cnt, KInt class_num);
__global__ void kai_ker_yolo_non_max_suppression(KInt size, KFloat* boxes_inout, KInt box_cnt, KInt max_boxes, KFloat iou_thr);

__global__ void kai_ker_real_to_complex(KInt size, KFloat* c_out, KFloat* f_in);
__global__ void kai_ker_short_to_complex(KInt size, KFloat* c_out, short* f_in);

__global__ void kai_ker_fft_step(KInt size, KFloat* dst_out, KFloat* src_in, KInt data_size, KInt step);

__global__ void kai_ker_complex_to_abs(KInt size, KFloat* a_out, KFloat* c_in);

__global__ void kai_ker_eveal_hash_match_point(KInt size, KFloat* p_out, KFloat* c1_in, KFloat* c2_in, KInt nrow, KInt ncol, KInt nvec);
__global__ void kai_ker_eveal_vector_dist(KInt size, KFloat* d_out, KFloat* c1_in, KFloat* c2_in, KInt nrow, KInt ncol, KInt nvec);
*/
