/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "cuda_conn.cuh"

#define ADD_RANGE       10000

__global__ void ker_init(int64 size, float* y_out, float value);
__global__ void ker_init_int(int64 size, int64* y_out, int64 value);
__global__ void ker_init_int64(int64 size, int64* y_out, int64 value);
__global__ void ker_copy_to(int64 size, float* y_out, float* x_in);
__global__ void ker_add_on(int64 size, float* y_out, float* x_in);
__global__ void ker_mult_to(int64 size, float* y_out, float* x1_in, float* x2_in);
__global__ void ker_mult_on(int64 size, float* y_inout, float* x_in);
__global__ void ker_mult_scalar_to(int64 size, float* y_inout, float* x1_in, float coef);
__global__ void ker_mult_scalar_on(int64 size, float* y_inout, float coef);
__global__ void ker_sigmoid_on(int64 size, float* x_inout);
__global__ void ker_tanh_on(int64 size, float* x_inout);
__global__ void ker_tanh_to(int64 size, float* y_out, float* x_in);
__global__ void ker_sigmoid_derv_on(int64 size, float* gx_inout, float* y_in);
__global__ void ker_tanh_derv_on(int64 size, float* gx_inout, float* y_in);
__global__ void ker_abs(int64 size, float* y_out, float* x_in);
__global__ void ker_sqr(int64 size, float* y_out, float* x_in);
__global__ void ker_near_zero(int64 size, float* y_out, float* x_in, float threshold);
__global__ void ker_rescale(int64 size, float* x_inout, float mean, float std);
__global__ void ker_sum(int64 size, float* x_inout, int64 xsize, int64 range);
__global__ void ker_sum_rows(int64 size, float* x_inout, int64 xsize, int64 range, int64 ncol);
__global__ void ker_binomial(int64 size, float* x_inout, float prob_threshod);
__global__ void ker_argmax(int64 size, int64* n_out, float* arr_in, int64 nvec);
__global__ void ker_get_hash_idx(int64 size, int64* n_out, float* arr_in, int64 nvec);
__global__ void ker_get_hash_diff(int64 size, int64* n_out, float* h1_in, float* h2_in, int64 nvec);

//__global__ void ker_forward_postproc(int64 size, float* cuda_res, float* cuda_aux, float* cuda_est, float* cuda_ans, int64 enumMod, int64 nvec);
__global__ void ker_mse(int64 size, float* y_out, float* est_in, float* ans_in);
__global__ void ker_sigmoid_cross_entropy_with_logits(int64 size, float* y_out, float* est_in, float* ans_in);
__global__ void ker_softmax_cross_entropy_with_logits(int64 size, float* y_out, float* est_in, float* ans_in, int64 nvec);
__global__ void ker_softmax_cross_entropy_with_logits_idx(int64 size, float* y_out, float* est_in, int64* ans_in, int64 nvec);
__global__ void ker_softmax_cross_entropy_with_logits_1st(int64 size, float* y_out, float* est_in, int64 nvec);

__global__ void ker_mult_diff_coef(int64 size, float* y_out, float* est_in, float* ans_in, float coef);
__global__ void ker_bin_acc(int64 size, float* y_out, float* x1_in, float* x2_in);
__global__ void ker_class_acc(int64 size, float* y_out, float* est_logit_in, float* ans_prob_in, int64 nvec);
__global__ void ker_class_idx_acc(int64 size, float* y_out, float* est_logit_in, int64* ans_prob_in, int64 nvec);
__global__ void ker_class_1st_acc(int64 size, float* y_out, float* est_logit_in, int64 nvec);
__global__ void ker_mse_diff_sq(int64 size, float* y_out, float* x1_in, float* x2_in);

__global__ void ker_matmul(int64 size, float* a_out, float* h_in, float* w_in, int64 nrow, int64 nvec, int64 ncol);
__global__ void ker_multi_matmul(int64 size, float* a_out, float* h_in, float* w_in, int64 msize, int64 nrow, int64 nvec, int64 ncol);
__global__ void ker_add_bias(int64 size, float* a_inout, float* b_in, int64 nrow, int64 nvec, int64 ncol);
__global__ void ker_matmul_derv_x(int64 size, float* gx_out, float* gy_in, float* w_in, int64 nrow, int64 nvec, int64 ncol);
__global__ void ker_matmul_derv_w(int64 size, float* gw_out, float* gy_in, float* x_in, int64 nrow, int64 nvec, int64 ncol);
__global__ void ker_multi_matmul_derv_x(int64 size, float* gx_out, float* gy_in, float* w_in, int64 mb_size, int64 nrow, int64 nvec, int64 ncol);
__global__ void ker_multi_matmul_derv_w(int64 size, float* gw_out, float* gy_in, float* x_in, int64 mb_size, int64 nrow, int64 nvec, int64 ncol);
__global__ void ker_add_bias_derv(int64 size, float* gb_out, float* gy_in, int64 nrow, int64 nvec, int64 ncol);

__global__ void ker_update_param_sgd(int64 size, float* pm_inout, float* gpm_in, float learning_rate, float l2_decay, float l1_decay);
__global__ void ker_update_param_adam(int64 size, float* pm_inout, float* s_inout, float* t_inout, float* gpm_in, float ro1, float ro2, int64 nstep, float epsilon, float learning_rate, float l2_decay, float l1_decay);

__global__ void ker_update_param_sgd_select(int64 size, float* pm_inout, int64* wid_in, float* gpm_in, int64 word_cnt, int64 vec_size, float learning_rate, float l2_decay, float l1_decay);
__global__ void ker_update_param_adam_select(int64 size, float* pm_inout, float* s_inout, float* t_inout, float* n_inout, int64* wid_in, float* gpm_in, int64 word_cnt, int64 vec_size, float ro1, float ro2, float epsilon, float learning_rate, float l2_decay, float l1_decay);

__global__ void ker_update_param_dup_count(int64 size, float* delta_out, float* count_out, int64* wid_in, float* gpm_in, int64 dic_count, int64 vec_size);
__global__ void ker_update_param_sgd_select_multi_dic(int64 size, float* pm_inout, int64* wid_in, float* gpm_in, int64 dic_count, int64* voc_counts, int64 vec_size, float learning_rate, float l2_decay, float l1_decay);
__global__ void ker_update_param_adam_select_multi_dic(int64 size, float* pm_inout, float* s_inout, float* t_inout, float* n_inout, int64* wid_in, float* delta_in, float* count_in, int64 dic_count, int64* voc_counts, int64 vec_size, float ro1, float ro2, float epsilon, float learning_rate, float l2_decay, float l1_decay);

__global__ void ker_update_affine_param(int64 size, float* gpm_in, float* wp_inout, float* bp_inout, int64 nrows, int64 ncols);

__global__ void ker_sigmoid(int64 size, float* y_out, float* x_in);
__global__ void ker_sigmoid_cross_entropy_with_logits_derv(int64 hsize, float* y_out, float* est_logit_in, float* ans_prob_in, float coef);

__global__ void ker_softmax(int64 size, float* y_out, float* x_in, int64 nvec);
__global__ void ker_softmax_derv(int64 size, float* gx_out, float* gy_in, float* y_in, int64 nvec);
__global__ void ker_softmax_cross_entropy_with_logits_derv(int64 hsize, float* y_out, float* est_logit_in, float* ans_prob_in, int64 nvec, float coef);
__global__ void ker_softmax_cross_entropy_with_logits_idx_derv(int64 hsize, float* y_out, float* est_logit_in, int64* ans_prob_in, int64 nvec, float coef);
__global__ void ker_softmax_cross_entropy_with_logits_1st_derv(int64 hsize, float* y_out, float* est_logit_in, int64 nvec, float coef);

__global__ void ker_dropout_old(int64 size, float* x_in, float* dm_in, float* y_out, int64 dm_size, float keep_ratio);
__global__ void ker_backprop_dropout_old(int64 size, float* gy_in, float* dm_in, float* gx_out, int64 dm_size, float keep_ratio);

__global__ void ker_activate(int64 size, float* y_out, float* x_in, int64 nFunc, float alpha);
__global__ void ker_activate_derv(int64 size, float* gx_out, float* gy_in, float* x_in, float* y_in, int64 nFunc, float alpha);

__global__ void ker_conv_kernel(int64 size, float* b_out, float* x_in, float* pm_in, int64 xh, int64 xw, int64 kh, int64 kw, int64 xchn, int64 ychn);
__global__ void ker_conv_sum(int64 size, float* y_out, float* b_in, int64 xchn);
__global__ void ker_conv_add_bias(int64 size, float* y_out, float* pm_in, int64 ychn);

__global__ void ker_conv_derv_x_kernel(int64 size, float* c_out, float* gy_in, float* k_in, int64 xh, int64 xw, int64 kh, int64 kw, int64 xchn, int64 ychn);
__global__ void ker_conv_derv_x_sum(int64 size, float* gx_out, float* c_in, int64 ychn);

__global__ void ker_conv_derv_kw_x(int64 size, float* d_out, float* gy_in, float* x_in, int64 mb_size, int64 xh, int64 xw, int64 kh, int64 kw, int64 xchn, int64 ychn);
__global__ void ker_conv_derv_kw_sum1(int64 size, float* d_inout, int64 xh);
__global__ void ker_conv_derv_kw_sum2(int64 size, float* gw_out, float* d_in, int64 mb_size, int64 xh);

__global__ void ker_conv_derv_kb_sum1(int64 size, float* d_out, float* gy_in, int64 mb_size, int64 xh, int64 xw, int64 ychn);
__global__ void ker_conv_derv_kb_sum2(int64 size, float* d_inout, int64 mb_size, int64 xh, int64 xw, int64 ychn);
__global__ void ker_conv_derv_kb_sum3(int64 size, float* gb_out, float* d_in, int64 mb_size, int64 xh, int64 xw, int64 ychn);

__global__ void ker_max(int64 size, float* y_out, int64* n_out, float* x_in, int64 xh, int64 xw, int64 xchn, int64 kh, int64 kw);
__global__ void ker_max_derv(int64 size, float* gx_out, int64* n_in, float* gy_in, int64 xh, int64 xw, int64 xchn, int64 kh, int64 kw);

__global__ void ker_avg(int64 size, float* y_out, int64* n_out, float* x_in, int64 xh, int64 xw, int64 xchn, int64 kh, int64 kw);
__global__ void ker_avg_derv(int64 size, float* gx_out, int64* n_in, float* gy_in, int64 xh, int64 xw, int64 xchn, int64 kh, int64 kw);

__global__ void ker_avg_exact(int64 size, float* t_out, float* x_in, int64 xh, int64 xw, int64 xchn, int64 sh, int64 sw);
__global__ void ker_avg_exact_derv(int64 size, float* gt_out, float* gy_in, int64 xh, int64 xw, int64 xchn, int64 sh, int64 sw);

__global__ void ker_tile_chn(int64 size, float* t_out, float* x_in, int64 ychn, int64 xchn);
__global__ void ker_untile_chn(int64 size, float* gt_out, float* gy_in, int64 ychn, int64 xchn);

__global__ void ker_stride(int64 size, float* y_out, float* x_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 kh, int64 kw, int64 sh, int64 sw, bool same_padding);
__global__ void ker_stride_derv(int64 size, float* gx_out, float* gy_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 kh, int64 kw, int64 sh, int64 sw, bool same_padding);

__global__ void ker_stride_expand(int64 size, float* y_out, float* x_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 sh, int64 sw);
__global__ void ker_stride_expand_derv(int64 size, float* gx_out, float* gy_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 sh, int64 sw);

__global__ void ker_batch_normal(int64 size, float* y_out, float* x_in, int64 ngroup);
__global__ void ker_batch_normal_derv(int64 size, float* gx_out, float* gy_in, float* y_in, int64 ngroup);

__global__ void ker_bn_collect(int64 bsize, float* avg_inout, float* var_inout, float* mavg_inout, float* mvar_inout, float* x_in, int64 hsize, float momentum);
__global__ void ker_bn_normalize(int64 hsize, float* h_inout, float* avg_in, float* var_in, int64 bsize, float epsilon);
__global__ void ker_bn_rescale(int64 hsize, float* h_inout, float* scale_in, float* shift_in, int64 bsize);

__global__ void ker_dropout(int64 size, float* y_out, float* x_in, float* m_in, float keep_ratio);
__global__ void ker_dropout_derv(int64 size, float* gx_out, float* gy_in, float* m_in, float keep_ratio);

__global__ void ker_bn_rescale_derv_pm(int64 bsize, float* gscale_out, float* gshift_out, float* gx_in, float* x_in, int64 hsize);
__global__ void ker_bn_rescale_derv_x(int64 hsize, float* gh_inout, float* scale_in, int64 bsize);
__global__ void ker_bn_norm_derv(int64 hsize, float* gh_inout, float* var_in, int64 bsize, float epsilon);

__global__ void ker_get_branch(int64 bsize, float* y_out, float* b_in, int64 ychn, int64 bchn, int64 chn_from);
__global__ void ker_set_branch(int64 bsize, float* gb_out, float* gy_in, int64 ychn, int64 bchn, int64 chn_from);

__global__ void ker_rnn_combine_ex_inp(int64 size, float* ex_inp_out, float* x_in, float* rec_in, int64 timesteps, int64 timefeats, int64 recur_size, bool inseq, int64 tn);
__global__ void ker_rnn_split_ex_inp(int64 size, float* gx_out, float* grec_out, float* gex_inp_in, int64 timesteps, int64 timefeats, int64 recur_size, bool inseq, int64 tn);

__global__ void ker_rnn_fill_output_slice(int64 size, float* y_out, float* x_in, int64 timesteps, int64 recur_size, int64 tn);
__global__ void ker_rnn_add_time_slice(int64 size, float* gy_out, float* gx_in, int64 timesteps, int64 recur_size, int64 tn);
__global__ void ker_rnn_copy_last_grad(int64 size, float* grec_inout, float* gx_in, int64 timesteps, int64 recur_size);

__global__ void ker_lstm_split_affine(int64 size, float* fgate_out, float* igate_out, float* ogate_out, float* block_out, float* affine_in, int64 recur_size);
__global__ void ker_lstm_combine_affine(int64 size, float* gaffine_out, float* gfgate_in, float* gigate_in, float* gogate_in, float* gblock_in, int64 recur_size);
__global__ void ker_rnn_select_last_vecs(int64 size, float* selected_out, float* time_pool_in, int64 timesteps, int64 nvec);

__global__ void ker_lstm_new_state(int64 size, float* state_out, float* state_in, float* fgate_in, float* block_in, float* igate_in);
__global__ void ker_lstm_state_derv(int64 size, float* gstate_inout, float* grec_in, float* ogate_in, float* rec_in);

__global__ void ker_attention_split(int64 size, float* q_out, float* k_out, float* v_out, float* qkv_in, int64 L, int64 H, int64 R);
__global__ void ker_attention_combine(int64 size, float* gqkv_out, float* gq_in, float* gk_in, float* gv_in, int64 L, int64 H, int64 R);

__global__ void ker_attention_mask_future(int64 size, float* score_inout, int64 L);

__global__ void ker_attention_reshape_out(int64 size, float* a_out, float* m_in, int64 L, int64 H, int64 R);
__global__ void ker_attention_reshape_mul(int64 size, float* gm_out, float* ga_in, int64 L, int64 H, int64 R);
/*
__global__ void ker_attention_forward_mult_val(int64 size, float* att_in, float* qkv_in, float* buf_out, int64 B, int64 L, int64 V, int64 H, int64 R);
__global__ void ker_attention_backprop_matmul_probs(int64 size, float* gy_in, float* qkv_in, float* gprobs_out, int64 B, int64 L, int64 V, int64 H, int64 R);
__global__ void ker_attention_backprop_matmul_value(int64 size, float* gy_in, float* probs_in, float* gqkv_out, int64 B, int64 L, int64 V, int64 H, int64 R);

__global__ void ker_attention_forward_qk_mult(int64 size, float* qkv_in, float* att_out, int64 B, int64 L, int64 V, int64 H, int64 R, float coef);
__global__ void ker_attention_backprop_mult_kv(int64 size, float* gscore_in, float* qkv_in, float* gqkv_out, int64 B, int64 L, int64 V, int64 H, int64 R, float coef);
*/

__global__ void ker_embedding_fetch(int64 size, float* wiv_out, float* wov_out, int64* hint_in, int64* noms_in, float* iwdic_in, float* owdic_in, int64 in_cnt, int64 out_cnt, int64 vec_size);

__global__ void ker_embedding_dotmul(int64 size, float* y_out, float* sub_in, float* wov_in, int64 vec_cnt, int64 vec_size);
__global__ void ker_embedding_dotmul_derv(int64 size, float* gsub_out, float* gwov_out, float* gy_in, float* sub_in, float* wov_in, int64 vec_cnt, int64 vec_size);

__global__ void ker_merge_avg(int64 size, float* m_out, float* x_in, int64 vec_cnt, int64 vec_size);
__global__ void ker_merge_avg_derv(int64 size, float* gx_out, float* gm_in, int64 vec_cnt, int64 vec_size);

__global__ void ker_extract(int64 size, float* e_out, float* x_in, int64 ax_size, int64 index, int64 nprod);
__global__ void ker_unextract(int64 size, float* gx_out, float* ge_in, int64 ax_size, int64 index, int64 nprod);

__global__ void ker_embed_fetch_multi_dic(int64 size, float* v_out, int64* wids_in, float* dics_in, int64 dic_count, int64* voc_counts, int64 vec_size);

__global__ void ker_vstack(int64 size, float* vs_out, float* x1_in, float* x2_in, int64 vol1, int64 vol2, int64 rest);
__global__ void ker_hstack(int64 size, float* vs_out, float* x1_in, float* x2_in, int64 vec1, int64 vec2);

__global__ void ker_hsplit(int64 size, float* p1_out, float* p2_out, float* src_in, int64 vec_size, int64 p1_size);

__global__ void ker_extract_selected_pickup(int64 size, float* dst_out, float* arr_in, int64* map_in, int64 drest);
__global__ void ker_extract_selected_pickup_int(int64 size, int64* dst_out, int64* arr_in, int64* map_in, int64 drest);
__global__ void ker_extract_selected_fill(int64 size, float* arr_inout, float* slice_in, int64* map_in, int64 drest);

__global__ void ker_expand(int64 size, float* y_out, float* x_in, int64 heights, int64 widths, int64 chns, int64 hratio, int64 wratio);
__global__ void ker_expand_derv(int64 size, float* gx_out, float* gy_in, int64 heights, int64 widths, int64 chns, int64 hratio, int64 wratio);

// (ndata, image_id, width, height) for images
#define IMG_INFO_SIZE       4

__device__ inline int64 img_ndata(int64* np, int64 n) { return np[n * IMG_INFO_SIZE + 0]; }
__device__ inline int64 img_imageid(int64* np, int64 n) { return np[n * IMG_INFO_SIZE + 1]; }
__device__ inline int64 img_width(int64* np, int64 n) { return np[n * IMG_INFO_SIZE + 2]; }
__device__ inline int64 img_height(int64* np, int64 n) { return np[n * IMG_INFO_SIZE + 3]; }

// (nimginfo, cat_id, center_x, center_y, width, height, mixed)
#define BOX_INFO_SIZE   7

__device__ inline int64 box_nimg(float* fp, int64 n) { return (int64)fp[n * BOX_INFO_SIZE + 0]; }
__device__ inline int64 box_catid(float* fp, int64 n) { return (int64)fp[n * BOX_INFO_SIZE + 1]; }

__device__ inline float box_cx(float* fp, int64 n) { return fp[n * BOX_INFO_SIZE + 2]; }
__device__ inline float box_cy(float* fp, int64 n) { return fp[n * BOX_INFO_SIZE + 3]; }
__device__ inline float box_wd(float* fp, int64 n) { return fp[n * BOX_INFO_SIZE + 4]; }
__device__ inline float box_ht(float* fp, int64 n) { return fp[n * BOX_INFO_SIZE + 5]; }
__device__ inline float box_mixed(float* fp, int64 n) { return fp[n * BOX_INFO_SIZE + 6]; }

__device__ inline float box_rect(float* fp, int64 n, int64 nc) { return fp[n * BOX_INFO_SIZE + nc + 2]; }

// (box_idx, nanchor, cx, cy)
#define SCALE_BOX_SIZE  4

__device__ inline int64 scale_nbox(int64* np, int64 n) { return np[n * SCALE_BOX_SIZE + 0]; }
__device__ inline int64 scale_nx(int64* np, int64 n) { return np[n * SCALE_BOX_SIZE + 1]; }
__device__ inline int64 scale_ny(int64* np, int64 n) { return np[n * SCALE_BOX_SIZE + 2]; }
__device__ inline int64 scale_nanchor(int64* np, int64 n) { return np[n * SCALE_BOX_SIZE + 3]; }

__device__ inline void set_scale_nbox(int64* np, int64 n, int64 v) { np[n * SCALE_BOX_SIZE + 0] = (int64)v; }
__device__ inline void set_scale_nx(int64* np, int64 n, int64 v) { np[n * SCALE_BOX_SIZE + 1] = (int64)v; }
__device__ inline void set_scale_ny(int64* np, int64 n, int64 v) { np[n * SCALE_BOX_SIZE + 2] = (int64)v; }
__device__ inline void set_scale_nanchor(int64* np, int64 n, int64 v) { np[n * SCALE_BOX_SIZE + 3] = (int64)v; }

// (cx, cy, wd, ht, conf, class0, ..., class79)
#define FMAP_SIZE  85

__device__ inline float fmap_cx(float* fp, int64 n) { return fp[n * FMAP_SIZE + 0]; }
__device__ inline float fmap_cy(float* fp, int64 n) { return fp[n * FMAP_SIZE + 1]; }
__device__ inline float fmap_wd(float* fp, int64 n) { return fp[n * FMAP_SIZE + 2]; }
__device__ inline float fmap_ht(float* fp, int64 n) { return fp[n * FMAP_SIZE + 3]; }
__device__ inline float fmap_conf(float* fp, int64 n) { return fp[n * FMAP_SIZE + 4]; }
__device__ inline float fmap_class(float* fp, int64 n, int64 cls) { return fp[n * FMAP_SIZE + 5 + cls]; }
__device__ inline float fmap_elem(float* fp, int64 n, int64 nc) { return fp[n * FMAP_SIZE + nc]; }

// (cx, cy, wd, ht, conf, class0, ..., class79)
#define PRED_SIZE  85

__device__ inline float pred_xmin(float* fp, int64 n) { return fp[n * PRED_SIZE + 0]; }
__device__ inline float pred_ymin(float* fp, int64 n) { return fp[n * PRED_SIZE + 1]; }
__device__ inline float pred_xmax(float* fp, int64 n) { return fp[n * PRED_SIZE + 2]; }
__device__ inline float pred_ymax(float* fp, int64 n) { return fp[n * PRED_SIZE + 3]; }
__device__ inline float pred_conf(float* fp, int64 n) { return fp[n * PRED_SIZE + 4]; }
__device__ inline float pred_class(float* fp, int64 n, int64 cls) { return fp[n * PRED_SIZE + 5 + cls]; }

__device__ inline void set_pred_xmin(float* fp, int64 n, float v) { fp[n * PRED_SIZE + 0] = v; }
__device__ inline void set_pred_ymin(float* fp, int64 n, float v) { fp[n * PRED_SIZE + 1] = v; }
__device__ inline void set_pred_xmax(float* fp, int64 n, float v) { fp[n * PRED_SIZE + 2] = v; }
__device__ inline void set_pred_ymax(float* fp, int64 n, float v) { fp[n * PRED_SIZE + 3] = v; }
__device__ inline void set_pred_conf(float* fp, int64 n, float v) { fp[n * PRED_SIZE + 4] = v; }
__device__ inline void set_pred_class(float* fp, int64 n, int64 cls, float v) { fp[n * PRED_SIZE + 5 + cls] = v; }

// [0:4] box rect, 4: score, 5: class_num, 6: ndata, 7:flag
#define RES_BOX_SIZE   8

__device__ inline float res_xmin(float* fp, int64 n) { return fp[n * RES_BOX_SIZE + 0]; }
__device__ inline float res_ymin(float* fp, int64 n) { return fp[n * RES_BOX_SIZE + 1]; }
__device__ inline float res_xmax(float* fp, int64 n) { return fp[n * RES_BOX_SIZE + 2]; }
__device__ inline float res_ymax(float* fp, int64 n) { return fp[n * RES_BOX_SIZE + 3]; }
__device__ inline float res_score(float* fp, int64 n) { return fp[n * RES_BOX_SIZE + 4]; }

__device__ inline int64 res_class(float* fp, int64 n) { return (int64)fp[n * RES_BOX_SIZE + 5]; }
__device__ inline int64 res_ndata(float* fp, int64 n) { return (int64)fp[n * RES_BOX_SIZE + 6]; }
__device__ inline int64 res_flag(float* fp, int64 n) { return (int64)fp[n * RES_BOX_SIZE + 7]; }

__device__ inline void set_res_xmin(float* fp, int64 n, float v) { fp[n * RES_BOX_SIZE + 0] = v; }
__device__ inline void set_res_ymin(float* fp, int64 n, float v) { fp[n * RES_BOX_SIZE + 1] = v; }
__device__ inline void set_res_xmax(float* fp, int64 n, float v) { fp[n * RES_BOX_SIZE + 2] = v; }
__device__ inline void set_res_ymax(float* fp, int64 n, float v) { fp[n * RES_BOX_SIZE + 3] = v; }
__device__ inline void set_res_score(float* fp, int64 n, float v) { fp[n * RES_BOX_SIZE + 4] = v; }

__device__ inline void set_res_class(float* fp, int64 n, int64 v) { fp[n * RES_BOX_SIZE + 5] = (float)v; }
__device__ inline void set_res_ndata(float* fp, int64 n, int64 v) { fp[n * RES_BOX_SIZE + 6] = (float)v; }
__device__ inline void set_res_flag(float* fp, int64 n, int64 v) { fp[n * RES_BOX_SIZE + 7] = (float)v; }

__device__ inline float myfmax(float x1, float x2) { return (x1 > x2) ? x1 : x2; }
__device__ inline float myfmin(float x1, float x2) { return (x1 < x2) ? x1 : x2; }

#define clip(x, lm, um) (((x) < um) ? (((x) > lm) ? (x) : lm) : um)

__global__ void ker_yolo_eval_true_box_score(int64 size, float* score_out, float* boxes_in, int64* anchors_in, int64 num_scales, int64 anchor_per_scale);
__global__ void ker_yolo_eval_true_box_select(int64 size, float* selected_out, float* score_in, int64 num_scales, int64 anchor_per_scale);
__global__ void ker_yolo_eval_true_count_selected(int64 size, int64* selected_cnt_out, float* selected_in, int64 dat_size, int64 num_scales, int64 anchor_per_scale);
__global__ void ker_yolo_eval_true_lookup_scale_box(int64 size, int64* box_scale_cood_out, float* selected_in, int64 nscale, int64 dat_size, int64 num_scales, int64 anchor_per_scale);
__global__ void ker_yolo_eval_true_eval_box_cood(int64 size, int64* box_scale_cood_inout, float* box_info_in, int64 grid_size);

/*

__global__ void ker_yolo_set_scaled_true_box(int64 size, float* coods_out, float* sizes_out, float* confs_out, int64* class_out, float* selected_in, float* boxes_in, int64* catid_in,
                                             int64 nd, int64 nscale, int64 img_size, int64 grid_cnt, int64 num_scales, int64 anchor_per_scale);
*/

__global__ void ker_yolo_conv_fmap(int64 size, float* pred_out, float* fmap_in, int64* anchors_in, int64 img_size, int64 grid_cnt, int64 anchors_cnt, int64 class_num);
__global__ void ker_yolo_eval_iou(int64 size, float* iou_out, int64* boxid_out, float* fmap_in, int64* img_info_in, float* box_info_in, int64* scale_boxes_in, int64* anchors_in,
                                  int64 nanchor, int64 img_size, int64 grid_cnt, int64 box_cnt);
__global__ void ker_yolo_select_best_iou(int64 size, int64* best_box_out, float* best_iou_out, float* iou_in, int64* boxid_in, int64 box_cnt);

__global__ void ker_yolo_eval_losses(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, float* best_iou_in, int64* anchors_in,
    int64 mb_size, int64 img_size, int64 grid_cnt, int64 nanchor, int64 class_num, bool use_focal, bool smooth_onehot);
__global__ void ker_yolo_eval_grads(int64 size, float* grad_out, float* fmap_in, float* box_info_in, int64* best_box_in, float* best_iou_in, int64* anchors_in,
    int64 mb_size, int64 img_size, int64 grid_cnt, int64 nanchor, int64 class_num, bool use_focal, bool smooth_onehot);

/*
__global__ void ker_yolo_loss_cood(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, int64 img_size, int64 nanchor, int64 grid_cnt);
__global__ void ker_yolo_loss_size(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, int64* anchors_in, int64 img_size, int64 nanchor, int64 grid_cnt);
__global__ void ker_yolo_loss_conf(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, float* best_iou_in, bool use_focal);
__global__ void ker_yolo_loss_class(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, int64 class_num, bool smooth_onehot);
*/

/*
__global__ void ker_yolo_coods_derv(int64 size, float* gcoods_out, float* gmixed_inout, float* pcood_in, float* pmixed_in, float* box_rect_in, int64* best_box_in, int64 img_size, int64 nanchor, int64 grid_cnt);
__global__ void ker_yolo_sizes_derv(int64 size, float* gsizes_out, float* gmixed_inout, float* psize_in, float* pmixed_in, float* box_rect_in, int64* best_box_in, int64* anchors_in, int64 img_size, int64 nanchor, int64 grid_cnt);
__global__ void ker_yolo_confs_derv(int64 size, float* gconfs_out, float* gmixed_inout, float* pconfs_in, float* pmixed_in, int64* best_box_in, float* best_iou_in, bool use_focal);
__global__ void ker_yolo_probs_derv(int64 size, float* gprobs_out, float* gmbufs_out, float* pprobs_in, float* pmixed_in, int64* best_box_in, int64* box_info_in, int64 class_num, bool smooth_onehot);
__global__ void ker_yolo_acc_gmbufs(int64 size, float* gmixed_inout, float* gmbufs_in, int64 class_num);
__global__ void ker_yolo_fmap_derv(int64 size, float* grad_fmap_out, float* gcoods_in, float* gsizes_in, float* gconfs_in, float* gprobs_in, float* gmixed_in, float* coods_in, float* sizes_in, float* confs_in, float* probs_in, float* mixed,
    int64* anchors_in, int64 img_size, int64 grid_cnt, int64 anchors_cnt, int64 class_num, bool use_mixed);
*/

__global__ void ker_yolo_select_pred_boxes(int64 size, unsigned char* flag_out, int64* ndata_out, float* conf_rects_out, float* fmap_in, int64* anchors_in, float pred_conf_thr, int64 out_base, int64 img_size, int64 grid_cnt, int64 anchor_per_scale, int64 class_num);

/*
__global__ void ker_yolo_eval_box_pair_ious(int64 size, int64* pinfo_out, float* ious_out, int64* idxs_in, int64* cats_in, float* rects_in, int64* box_info_in, float* box_rect_in, int64 tbox_cnt);
__global__ void ker_yolo_count_true_boxes(int64 size, int64* tbox_cnt_out, int64* box_info_in, int64 tbox_cnt);
__global__ void ker_yolo_count_pred_boxes(int64 size, int64* pbox_cnt_out, int64* pair_info_in, int64 pbox_cnt, int64 tbox_cnt);
__global__ void ker_yolo_count_matched_box_pairs(int64 size, int64* match_cnt_out, int64* pair_info_in, float* ious_in, int64 iou_thr_cnt, float iou_thr_from, float iou_thr_step, int64 pbox_cnt, int64 tbox_cnt);
__global__ void ker_yolo_eval_prec_recall(int64 size, float* precision_out, float* recall_out, int64* tbox_cnt_in, int64* pbox_cnt_in, int64* match_cnt_in, int64 iou_thr_cnt);
*/

__global__ void ker_yolo_eval_predict_score(int64 size, float* score_out, float* pred_in, int64 class_num, float score_thresh);
__global__ void ker_yolo_get_boxes(int64 size, float* boxes_out, float* old_boxes_in, int64* idxs_in, float* score_in, float* pred_in, int64 old_count, int64 grid_cnt, int64 anchors_cnt, int64 class_num);
__global__ void ker_yolo_non_max_suppression(int64 size, float* boxes_inout, int64 box_cnt, int64 max_boxes, float iou_thr);

__global__ void ker_real_to_complex(int64 size, float* c_out, float* f_in);
__global__ void ker_short_to_complex(int64 size, float* c_out, short* f_in);
__global__ void ker_wave_slices_to_complex(int64 size, float* c_out, float* w_in, int64 step_width, int64 step_cnt, int64 fft_width, int64 fetch_width);

__global__ void ker_fft_step(int64 size, float* dst_out, float* src_in, int64 data_size, int64 step);
__global__ void ker_fft_step_split(int64 size, float* dst_out, float* src_in, int64 data_size, int64 step, int64 nd_base);

__global__ void ker_complex_to_abs(int64 size, float* a_out, float* c_in);
__global__ void ker_complex_to_abs_mean(int64 size, float* a_out, float* c_in, int64 data_num, int64 freq_cnt);

__global__ void ker_eveal_hash_match_point(int64 size, float* p_out, float* c1_in, float* c2_in, int64 nrow, int64 ncol, int64 nvec);
__global__ void ker_eveal_vector_dist(int64 size, float* d_out, float* c1_in, float* c2_in, int64 nrow, int64 ncol, int64 nvec);
