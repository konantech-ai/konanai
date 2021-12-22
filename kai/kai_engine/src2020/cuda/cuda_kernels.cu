/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "cuda_kernels.h"
#include "cuda_note.h"
#include "../int_plugin/layer.cuh"
#include "math_constants.h"

/*****************************************************************************
       device functions
*****************************************************************************/

__device__ float dev_sigmoid(float x) {
    if (x > 0) return __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-x)));
    else       return __fdiv_rn(__expf(x), __fadd_rn(1.0f, __expf(x)));
}

__device__ float dev_sigmoid_derv(float x) {
    float y = dev_sigmoid(x);
    return __fmul_rn(y, __fsub_rn(1.0f, y));
}

__device__ float dev_sigmoid_derv(float x, float y) {
    return __fmul_rn(y, __fsub_rn(1.0f, y));
}

__device__ float dev_tanh(float x) {
    return 2 * dev_sigmoid(2 * x) - 1;
}

__device__ float dev_tanh_derv(float x, float y) {
    return __fsub_rn(1, __fmul_rn(y, y));
}

__device__ float dev_sigmoid_cross_entropy_with_logits(float x, float z) {
    float p_inv = __fadd_rn(1.0f, __expf(-x));
    float term1 = __fmul_rn(__logf(p_inv), z);
    float term2 = __fmul_rn(__fadd_rn(x, __logf(p_inv)), __fsub_rn(1.0f, z));
    float ent = __fadd_rn(term1, term2);
    return ent;

    //return __fsub_rn(__logf(__fadd_rn(1.0f, __expf(x))), __fmul_rn(x, z));

    /*
    if (x > 0) return __fsub_rn(__fadd_rn(x, __logf(__fadd_rn(1.0f, __expf(-x)))), __fmul_rn(x, z));
    else       return __fsub_rn(__logf(__fadd_rn(1.0f, __expf(x))), __fmul_rn(x, z));
    */

    /*
    float p = 1.0f / (1.0f + __expf(-x));
    float ent = -__logf(p) * z - __logf(1.0f - p) * (1.0f - z);
    return ent;
    */

    /*
    float p = __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-x)));
    float term1 = __fmul_rn(__logf(p), z);
    float term2 = __fmul_rn(__logf(__fsub_rn(1.0f, p)), __fsub_rn(1.0f, z));
    float ent = -__fadd_rn(term1, term2);
    return ent;
    */
}

__device__ void dev_get_max_sum_for_softmax(float* logits, int64 nvec, float* pmax, float* psum) {
    float max_term = logits[0];
    float sum_exp = 0;

    for (int64 n = 1; n < nvec; n++) {
        if (logits[n] > max_term) max_term = logits[n];
    }

    for (int64 n = 0; n < nvec; n++) {
        sum_exp = __fadd_rn(sum_exp, __expf(__fsub_rn(logits[n], max_term)));
    }

    *pmax = max_term;
    *psum = sum_exp;
}

__device__ float dev_sigmoid_cross_entropy_with_logits_derv(float x, float z) {
    return __fsub_rn(dev_sigmoid(x), z);
}

/*****************************************************************************
       basic operation
*****************************************************************************/
__global__ void ker_init(int64 size, float* y_out, float value) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = value;
    }
}

__global__ void ker_init_int(int64 size, int* y_out, int value) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = value;
    }
}

__global__ void ker_init_int64(int64 size, int64* y_out, int64 value) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = value;
    }
}

__global__ void ker_add_on(int64 size, float* x_inout, float* p_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x_inout[idx] = __fadd_rn(x_inout[idx], p_in[idx]);
    }
}

__global__ void ker_copy_to(int64 size, float* y_out, float* x_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = x_in[idx];
    }
}

__global__ void ker_mult_to(int64 size, float* y_out, float* x1_in, float* x2_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = __fmul_rn(x1_in[idx], x2_in[idx]);
    }
}

__global__ void ker_mult_on(int64 size, float* y_inout, float* x_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_inout[idx] = __fmul_rn(y_inout[idx], x_in[idx]);
    }
}

__global__ void ker_mult_scalar_to(int64 size, float* y_inout, float* x1_in, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_inout[idx] = x1_in[idx] * coef;
    }
}

__global__ void ker_mult_scalar_on(int64 size, float* y_inout, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_inout[idx] = __fmul_rn(y_inout[idx], coef);
    }
}

__global__ void ker_sigmoid_on(int64 size, float* x_inout) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x_inout[idx] = dev_sigmoid(x_inout[idx]);
    }
}

__global__ void ker_tanh_on(int64 size, float* x_inout) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x_inout[idx] = dev_tanh(x_inout[idx]);
    }
}

__global__ void ker_tanh_to(int64 size, float* y_out, float* x_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = dev_tanh(x_in[idx]);
    }
}

__global__ void ker_sigmoid_derv_on(int64 size, float* gx_inout, float* y_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx_inout[idx] = __fmul_rn(dev_sigmoid_derv(0, y_in[idx]), gx_inout[idx]);
    }
}

__global__ void ker_tanh_derv_on(int64 size, float* gx_inout, float* y_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx_inout[idx] = __fmul_rn(dev_tanh_derv(0, y_in[idx]), gx_inout[idx]);
    }
}

__global__ void ker_abs(int64 size, float* y_out, float* x_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = x_in[idx];;
        y_out[idx] = (x > 0) ? x : -x;
    }
}

__global__ void ker_sqr(int64 size, float* y_out, float* x_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = x_in[idx];;
        y_out[idx] = __fmul_rn(x, x);
    }
}

__global__ void ker_binomial(int64 size, float* x_inout, float prob_threshod) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x_inout[idx] = (x_inout[idx] < prob_threshod) ? 1.0f : 0.0f;
    }
}

__global__ void ker_argmax(int64 size, int64* n_out, float* arr_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 from = idx * nvec;
        int64 argmax = 0;
        for (int64 n = 1; n < nvec; n++) {
            if (arr_in[from+n] > arr_in[from + argmax]) argmax = n;
        }
        n_out[idx] = argmax;
    }
}

__global__ void ker_get_hash_idx(int64 size, int64* n_out, float* arr_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 from = idx * nvec;
        int64 hash_idx = 0;
        for (int64 n = 0; n < nvec; n++) {
            hash_idx *= 2;
            if (arr_in[from + n] > 0) hash_idx++; // tanh 활성화 함수를 이용해 0을 기준으로 대칭 분포가 이루어지도록 하고 있음
        }
        n_out[idx] = hash_idx;
    }
}

__global__ void ker_get_hash_diff(int64 size, int64* n_out, float* h1_in, float* h2_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 hash1 = h1_in[idx];
        int64 hash2 = h2_in[idx];

        int64 bit_mask = 1;
        int64 match_cnt = 0;

        for (int64 n = 0; n < nvec; n++) {
            if ((hash1 & bit_mask) == (hash2 & bit_mask)) match_cnt++;
            bit_mask = bit_mask << 1;
        }
        n_out[idx] = match_cnt;
    }
}

__global__ void ker_near_zero(int64 size, float* y_out, float* x_in, float threshold) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = x_in[idx];;
        y_out[idx] = (-threshold < x && x < threshold) ? 1.0f : 0.0f;
    }
}

__global__ void ker_rescale(int64 size, float* x_inout, float mean, float std) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x_inout[idx] = __fmaf_rn(x_inout[idx], std, mean);
    }
}

__global__ void ker_sum(int64 size, float* x_inout, int64 xsize, int64 range) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 dest = idx * range;
        int64 start = dest;
        int64 end = MIN(dest + range, xsize);
        int64 step = range / ADD_RANGE;

        float sum = 0;
        for (int64 n = start; n < end; n += step) {
            sum = __fadd_rn(sum, x_inout[n]);
        }
        x_inout[dest] = sum;
    }
}

__global__ void ker_sum_rows(int64 size, float* x_inout, int64 xsize, int64 range, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / ncol;
        int64 nc = idx % ncol;

        int64 dest = nr * range * ncol + nc;
        int64 start = dest;
        int64 end = MIN(dest + range * ncol, xsize * ncol);
        int64 step = (range / ADD_RANGE) * ncol;

        float sum = 0;

        for (int64 n = start; n < end; n += step) {
            sum = __fadd_rn(sum, x_inout[n]);
        }
        
        x_inout[dest] = sum;
    }
}

/*****************************************************************************
       dataset builtin postproc kernels
*****************************************************************************/
__global__ void ker_mse(int64 size, float* y_out, float* est_in, float* ans_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float diff = __fsub_rn(est_in[idx], ans_in[idx]);
        y_out[idx] = __fmul_rn(diff, diff);
    }
}

__global__ void ker_sigmoid_cross_entropy_with_logits(int64 size, float* y_out, float* est_in, float* ans_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = dev_sigmoid_cross_entropy_with_logits(est_in[idx], ans_in[idx]);
    }
}

__global__ void ker_softmax_cross_entropy_with_logits(int64 size, float* y_out, float* est_in, float* ans_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float* logits = est_in + idx * nvec;
        float* answer = ans_in + idx * nvec;

        float max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        float entropy = 0;

        for (int64 n = 0; n < nvec; n++) {
            float prob_term = __fdiv_rn(__expf(__fsub_rn(logits[n], max_term)), sum_exp);
            float log_term = __logf(__fadd_rn(prob_term, 1.0e-10f));
            entropy = __fadd_rn(entropy, __fmul_rn(answer[n], log_term));
        }

        y_out[idx] = -entropy;
    }
}

__global__ void ker_softmax_cross_entropy_with_logits_idx(int64 size, float* y_out, float* est_in, int64* ans_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float* logits = est_in + idx * nvec;

        float max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        int64 nth = ans_in[idx];
        
        float prob_term = __fdiv_rn(__expf(__fsub_rn(logits[nth], max_term)), sum_exp);
        float log_term = __logf(__fadd_rn(prob_term, 1.0e-10f));
        float entropy = log_term;

        y_out[idx] = -entropy;
    }
}

__global__ void ker_softmax_cross_entropy_with_logits_1st(int64 size, float* y_out, float* est_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float* logits = est_in + idx * nvec;

        float max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        float prob_term = __fdiv_rn(__expf(__fsub_rn(logits[0], max_term)), sum_exp);
        float log_term = __logf(__fadd_rn(prob_term, 1.0e-10f));
        float entropy = log_term;

        y_out[idx] = -entropy;
    }
}

__global__ void ker_mult_diff_coef(int64 size, float* y_out, float* est_in, float* ans_in, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = __fmul_rn(__fsub_rn(est_in[idx], ans_in[idx]), coef);
    }
}

__global__ void ker_bin_acc(int64 size, float* y_out, float* est_logit_in, float* ans_prob_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        bool estimate = est_logit_in[idx] > 0;
        bool answer = ans_prob_in[idx] > 0.5;
        y_out[idx] = (estimate == answer) ? 1.0f : 0;
    }
}

__global__ void ker_class_acc(int64 size, float* y_out, float* est_logit_in, float* ans_probs_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 pos = idx * nvec;
        
        float est_max = est_logit_in[pos];
        float ans_max = ans_probs_in[pos];

        int64 est_arg = pos;
        int64 ans_arg = pos++;

        for (int64 n = 1; n < nvec; n++, pos++) {
            if (est_max < est_logit_in[pos]) est_max = est_logit_in[pos], est_arg = pos;
            if (ans_max < ans_probs_in[pos]) ans_max = ans_probs_in[pos], ans_arg = pos;
        }

        y_out[idx] = (est_arg == ans_arg) ? 1.0f : 0;
    }
}

__global__ void ker_class_idx_acc(int64 size, float* y_out, float* est_logit_in, int64* ans_probs_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 pos = idx * nvec;

        int64 ans_pos = pos + (int64)ans_probs_in[idx];

        float est_max = est_logit_in[pos];
        int64 est_pos = pos++;

        for (int64 n = 1; n < nvec; n++, pos++) {
            if (est_max < est_logit_in[pos]) est_max = est_logit_in[pos], est_pos = pos;
        }

        y_out[idx] = (est_pos == ans_pos) ? 1.0f : 0;
    }
}

__global__ void ker_class_1st_acc(int64 size, float* y_out, float* est_logit_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 pos = idx * nvec;

        float est_max = est_logit_in[pos];
        int64 est_arg = pos++;

        for (int64 n = 1; n < nvec; n++, pos++) {
            if (est_max < est_logit_in[pos]) est_max = est_logit_in[pos], est_arg = pos;
        }

        y_out[idx] = (est_arg == 0) ? 1.0f : 0;
    }
}

__global__ void ker_mse_diff_sq(int64 size, float* y_out, float* x1_in, float* x2_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float diff = __fsub_rn(x1_in[idx] , x2_in[idx]);
        y_out[idx] =__fmul_rn( diff, diff);
    }
}

/*****************************************************************************
       affine kernels
*****************************************************************************/
__global__ void ker_matmul(int64 size, float* a_out, float* h_in, float* w_in, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(nrow * ncol == size);

        int64 nr = idx / ncol;
        int64 nc = idx % ncol;

        int64 xpos = nr * nvec;
        int64 wpos = nc;

        float sum = 0;

        for (int64 nv = 0; nv < nvec; nv++) {
            sum = __fadd_rn(sum, __fmul_rn(h_in[xpos], w_in[wpos]));
            xpos++, wpos += ncol;
        }

        a_out[idx] = sum;
    }
}

__global__ void ker_multi_matmul(int64 size, float* a_out, float* h_in, float* w_in, int64 mb_size, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(mb_size * nrow * ncol == size);

        int64 nd = idx / (nrow * ncol);
        int64 nr = (idx / ncol) % nrow;
        int64 nc = idx % ncol;

        int64 xpos = nd * (nrow * nvec) + nr * nvec;
        int64 wpos = nd * (nvec * ncol) + nc;

        float sum = 0;

        for (int64 nv = 0; nv < nvec; nv++) {
            sum = __fadd_rn(sum, __fmul_rn(h_in[xpos], w_in[wpos]));
            xpos++, wpos += ncol;
        }

        a_out[idx] = sum;
    }
}

__global__ void ker_add_bias(int64 size, float* a_inout, float* b_in, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nc = idx % ncol;
        a_inout[idx] = __fadd_rn(a_inout[idx], b_in[nc]);
    }
}

__global__ void ker_matmul_derv_x(int64 size, float* gx_out, float* gy_in, float* w_in, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(nrow * nvec == size);

        int64 nr = idx / nvec;
        int64 nv = idx % nvec;

        int64 wpos = nv * ncol;
        int64 ypos = nr * ncol;

        float sum = 0;

        for (int64 n = 0; n < ncol; n++) {
            sum = __fadd_rn(sum, __fmul_rn(w_in[wpos++], gy_in[ypos++]));
        }

        gx_out[idx] = sum;
    }
}

__global__ void ker_multi_matmul_derv_x(int64 size, float* gx_out, float* gy_in, float* w_in, int64 mb_size, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(mb_size * nrow * nvec == size);

        int64 nd = idx / (nrow * nvec);
        int64 nr = (idx / nvec) % nrow;
        int64 nv = idx % nvec;

        int64 wpos = nd * (nvec * ncol) + nv * ncol;
        int64 ypos = nd * (nrow * ncol) + nr * ncol;

        float sum = 0;

        for (int64 n = 0; n < ncol; n++) {
            sum = __fadd_rn(sum, __fmul_rn(w_in[wpos++], gy_in[ypos++]));
        }

        gx_out[idx] = sum;
    }
}

__global__ void ker_matmul_derv_w(int64 size, float* gw_out, float* gy_in, float* x_in, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(nvec * ncol == size);

        int64 nv = idx / ncol;  // 여기에 nd 관련 내용 추가
        int64 nc = idx % ncol;

        int64 xpos = nv;
        int64 ypos = nc;

        float wsum = 0;

        for (int64 n = 0; n < nrow; n++) {
            wsum = __fadd_rn(wsum, __fmul_rn(x_in[xpos], gy_in[ypos]));
            xpos += nvec, ypos += ncol;
        }

        gw_out[idx] = wsum;
    }
}

__global__ void ker_multi_matmul_derv_w(int64 size, float* gw_out, float* gy_in, float* x_in, int64 mb_size, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(mb_size * nvec * ncol == size);

        int64 nd = idx / (nvec * ncol);
        int64 nv = (idx / ncol) % nvec;
        int64 nc = idx % ncol;

        int64 xpos = nd * nrow * nvec + nv;
        int64 ypos = nd * nrow * ncol + nc;

        float wsum = 0;

        for (int64 n = 0; n < nrow; n++) {
            wsum = __fadd_rn(wsum, __fmul_rn(x_in[xpos], gy_in[ypos]));
            xpos += nvec, ypos += ncol;
        }

        gw_out[idx] = wsum;
    }
}

__global__ void ker_add_bias_derv(int64 size, float* gb_out, float* gy_in, int64 nrow, int64 nvec, int64 ncol) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ypos = idx;
        float bsum = 0;

        for (int64 n = 0; n < nrow; n++) {
            bsum = __fadd_rn(bsum, gy_in[ypos]);
            ypos += ncol;
        }

        gb_out[idx] = bsum;
    }
}

__global__ void ker_update_affine_param(int64 size, float* gpm_in, float* wp_inout, float* bp_inout, int64 nrows, int64 ncols) {
    assert(0);
}
;
/*****************************************************************************
       parameter update kernels
*****************************************************************************/
__global__ void ker_update_param_sgd(int64 size, float* pm_inout, float* gpm_in, float learning_rate, float l2_decay, float l1_decay) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float delta = gpm_in[idx];

        if (l2_decay > 0) delta = __fmaf_rn(pm_inout[idx], l2_decay, delta);
        if (l1_decay > 0) {
            if (pm_inout[idx] >= -l1_decay && pm_inout[idx] <= l1_decay) {
                pm_inout[idx] = 0;
                return;
            }
            else if (pm_inout[idx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pm_inout[idx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pm_inout[idx] = __fsub_rn(pm_inout[idx], __fmul_rn(delta, learning_rate));
    }
}

__global__ void ker_update_param_adam(int64 size, float* pm_inout, float* s_inout, float* t_inout, float* gpm_in, float ro1, float ro2, int64 nstep, float epsilon, float learning_rate, float l2_decay, float l1_decay) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float delta = gpm_in[idx];

        s_inout[idx] = __fadd_rn(__fmul_rn(s_inout[idx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
        t_inout[idx] = __fadd_rn(__fmul_rn(t_inout[idx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

        float sm = __fdiv_rn(s_inout[idx], __fsub_rn(1.0f, __powf(ro1, (float)nstep)));
        float tm = __fdiv_rn(t_inout[idx], __fsub_rn(1.0f, __powf(ro2, (float)nstep)));

        delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

        if (l2_decay) delta += pm_inout[idx] * l2_decay;
        if (l1_decay) {
            if (pm_inout[idx] >= -l1_decay && pm_inout[idx] <= l1_decay) {
                pm_inout[idx] = 0;
                return;
            }
            else if (pm_inout[idx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pm_inout[idx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pm_inout[idx] = __fsub_rn(pm_inout[idx], __fmul_rn(delta, learning_rate));
    }
}

__global__ void ker_update_param_sgd_select(int64 size, float* pm_inout, int64* wid_in, float* gpm_in, int64 word_cnt, int64 vec_size, float learning_rate, float l2_decay, float l1_decay) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) {
    int64 nw = idx / vec_size;
    int64 nv = idx % vec_size;

    int64 pmidx = (int64)wid_in[nw] * vec_size + nv;

    float delta = gpm_in[idx];

    if (l2_decay > 0) delta = __fmaf_rn(pm_inout[pmidx], l2_decay, delta);
    if (l1_decay > 0) {
        if (pm_inout[pmidx] >= -l1_decay && pm_inout[pmidx] <= l1_decay) {
            pm_inout[pmidx] = 0;
            return;
        }
        else if (pm_inout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
        else if (pm_inout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
    }

    pm_inout[pmidx] = __fsub_rn(pm_inout[pmidx], __fmul_rn(delta, learning_rate));
}
}

__global__ void ker_update_param_adam_select(int64 size, float* pm_inout, float* s_inout, float* t_inout, float* n_inout, int64* wid_in, float* gpm_in, int64 word_cnt, int64 vec_size, float ro1, float ro2, float epsilon, float learning_rate, float l2_decay, float l1_decay) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64 nw = idx / vec_size;
        int64 nv = idx % vec_size;

        for (int64 n = 0; n < word_cnt; n++) {
            if (wid_in[n] != nw) continue;

            int64 pmidx = (int64)wid_in[n] * vec_size + nv;

            float delta = gpm_in[idx];

            s_inout[idx] = __fadd_rn(__fmul_rn(s_inout[idx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
            t_inout[idx] = __fadd_rn(__fmul_rn(t_inout[idx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

            float step = n_inout[nw] = __fadd_rn(n_inout[nw], 1.0f);

            float sm = __fdiv_rn(s_inout[idx], __fsub_rn(1.0f, __powf(ro1, step)));
            float tm = __fdiv_rn(t_inout[idx], __fsub_rn(1.0f, __powf(ro2, step)));

            delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

            if (l2_decay) delta += pm_inout[pmidx] * l2_decay;
            if (l1_decay) {
                if (pm_inout[idx] >= -l1_decay && pm_inout[idx] <= l1_decay) {
                    pm_inout[idx] = 0;
                    return;
                }
                else if (pm_inout[idx] > 0) delta = __fadd_rn(delta, l1_decay);
                else if (pm_inout[idx] < 0) delta = __fsub_rn(delta, l1_decay);
            }

            pm_inout[idx] = __fsub_rn(pm_inout[idx], __fmul_rn(delta, learning_rate));
        }
    }
}

__global__ void ker_update_param_sgd_select_multi_dic(int64 size, float* pm_inout, int64* wid_in, float* gpm_in, int64 dic_count, int64* voc_counts, int64 vec_size, float learning_rate, float l2_decay, float l1_decay) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64 nw = idx / (dic_count * vec_size);
        int64 nd = (idx / vec_size) % dic_count;
        int64 nv = idx % vec_size;

        int64 gpmidx = nw * vec_size + nv;
        float delta = gpm_in[gpmidx];

        int64 ndic = (int64)wid_in[nw * dic_count + nd];

        for (int64 n = 0; n < nd; n++) ndic += voc_counts[n];

        int64 pmidx = ndic * vec_size + nv;

        if (l2_decay > 0) delta = __fmaf_rn(pm_inout[pmidx], l2_decay, delta);
        if (l1_decay > 0) {
            if (pm_inout[pmidx] >= -l1_decay && pm_inout[pmidx] <= l1_decay) {
                pm_inout[pmidx] = 0;
                return;
            }
            else if (pm_inout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pm_inout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pm_inout[pmidx] = __fsub_rn(pm_inout[pmidx], __fmul_rn(delta, learning_rate));
    }
}

__global__ void ker_update_param_dup_count(int64 size, float* delta_out, float* count_out, int64* wid_in, float* gpm_in, int64 dic_count, int64 vec_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64 nw = idx / vec_size;
        int64 nv = idx % vec_size;

        //int64 nt = nw / dic_count;
        int64 nd = nw % dic_count;

        delta_out[idx] = 0;
        if (nv == 0) count_out[nw] = -1;

        int64 wid = (int64)wid_in[nw];

        //if (wid == 0 && nd_batch == 0) return;

        int64 count = 0;
        int64 terms_in_batch = size / vec_size;

        for (int64 n = nd; n < terms_in_batch; n += dic_count) {
            int64 wid_nom = (int64)wid_in[n];
            if (wid_nom != wid) continue;
            if (n < nw) return;

            int64 pmidx = (n / dic_count) * vec_size + nv;

            delta_out[idx] += gpm_in[pmidx];
            count++;
        }
        if (nv == 0) count_out[nw] = (float)count;
    }
}

__global__ void ker_update_param_adam_select_multi_dic(int64 size, float* pm_inout, float* s_inout, float* t_inout, float* n_inout, int64* wid_in, float* delta_in, float* count_in, int64 dic_count, int64* voc_counts, int64 vec_size, float ro1, float ro2, float epsilon, float learning_rate, float l2_decay, float l1_decay) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64 nw = idx / vec_size;
        int64 nv = idx % vec_size;

        //int64 nt = nw / dic_count;
        int64 nd = nw % dic_count;

        if (count_in[nw] <= 0) return;

        int64 wid = (int64)wid_in[nw];

        //if (wid == 0 && nd == 0) return;

        int64 wid_in_dics = wid;
        for (int64 n = 0; n < nd; n++) wid_in_dics += voc_counts[n];

        int64 pmidx = wid_in_dics * vec_size + nv;

        float delta = delta_in[idx];

        s_inout[pmidx] = __fadd_rn(__fmul_rn(s_inout[pmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
        t_inout[pmidx] = __fadd_rn(__fmul_rn(t_inout[pmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

        float step = __fadd_rn(n_inout[wid_in_dics], 1.0f);
        if (nv == 0) n_inout[wid_in_dics] = step;

        float sm = __fdiv_rn(s_inout[pmidx], __fsub_rn(1.0f, __powf(ro1, step)));
        float tm = __fdiv_rn(t_inout[pmidx], __fsub_rn(1.0f, __powf(ro2, step)));

        delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

        if (l2_decay) delta += pm_inout[pmidx] * l2_decay;
        if (l1_decay) {
            if (pm_inout[pmidx] >= -l1_decay && pm_inout[pmidx] <= l1_decay) {
                pm_inout[pmidx] = 0;
                return;
            }
            else if (pm_inout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pm_inout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pm_inout[pmidx] = __fsub_rn(pm_inout[pmidx], __fmul_rn(delta, learning_rate));
        /*
        int64 terms_in_batch = size / vec_size;

        for (int64 n = nd_batch; n < terms_in_batch; n += dic_count) {
            int64 wid_nom = (int64) wid_in[n];
            if (wid_nom != wid) continue;
            if (n < wid_pos) return;

            int64 ipmidx = n * vec_size + nv_batch;

            float delta = gpm_in[ipmidx];

            s_inout[dpmidx] = __fadd_rn(__fmul_rn(s_inout[dpmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
            t_inout[dpmidx] = __fadd_rn(__fmul_rn(t_inout[dpmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

            float step = __fadd_rn(n_inout[wid_in_dics], 1.0f);
            if (nv_batch == 0) n_inout[wid_in_dics] = step;

            //s_inout[dpmidx] = delta; // __fadd_rn(__fmul_rn(s_inout[dpmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
            t_inout[dpmidx] = __fadd_rn(__fmul_rn(t_inout[dpmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

            float step = __fadd_rn(n_inout[wid_in_dics], 1.0f);
            if (nv_batch == 0) n_inout[wid_in_dics] = step;

            float sm = __fdiv_rn(s_inout[dpmidx], __fsub_rn(1.0f, __powf(ro1, step)));
            float tm = __fdiv_rn(t_inout[dpmidx], __fsub_rn(1.0f, __powf(ro2, step)));

            delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

            if (l2_decay) delta += pm_inout[dpmidx] * l2_decay;
            if (l1_decay) {
                if (pm_inout[dpmidx] >= -l1_decay && pm_inout[dpmidx] <= l1_decay) {
                    pm_inout[dpmidx] = 0;
                    return;
                }
                else if (pm_inout[dpmidx] > 0) delta = __fadd_rn(delta, l1_decay);
                else if (pm_inout[dpmidx] < 0) delta = __fsub_rn(delta, l1_decay);
            }

            pm_inout[dpmidx] = __fsub_rn(pm_inout[dpmidx], __fmul_rn(delta, learning_rate));
        }
        */

        /*
        s_inout[pmidx] = __fadd_rn(__fmul_rn(s_inout[pmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
        t_inout[pmidx] = __fadd_rn(__fmul_rn(t_inout[pmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

        float step = n_inout[nw] = __fadd_rn(n_inout[nw], 1.0f);

        float sm = __fdiv_rn(s_inout[pmidx], __fsub_rn(1.0f, __powf(ro1, step)));
        float tm = __fdiv_rn(t_inout[pmidx], __fsub_rn(1.0f, __powf(ro2, step)));

        delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

        if (l2_decay) delta += pm_inout[pmidx] * l2_decay;
        if (l1_decay) {
            if (pm_inout[pmidx] >= -l1_decay && pm_inout[pmidx] <= l1_decay) {
                pm_inout[pmidx] = 0;
                return;
            }
            else if (pm_inout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pm_inout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pm_inout[pmidx] = __fsub_rn(pm_inout[pmidx], __fmul_rn(delta, learning_rate));
        */
    }
}

// depreciated
__global__
void ker_param_update(int64 size, float* p, float* g, float* s, float* t,
    float ro1, float ro2, float ro1_pow, float ro2_pow, float epsilon, float learning_rate, float l2_decay, float l1_decay)
{
    int64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float delta = g[i];

        if (s != NULL) {
            s[i] = s[i] * ro1 + (1.0f - ro1) * delta;
            t[i] = t[i] * ro2 + (1.0f - ro2) * delta * delta;

            float sm = s[i] / (1.0f - ro1_pow);
            float tm = t[i] / (1.0f - ro2_pow);

            delta = sm / ((float)::__fsqrt_rn(tm) + epsilon);
        }

        if (l2_decay) delta += p[i] * l2_decay;
        if (l1_decay) {
            if (p[i] > 0) delta += l1_decay;
            else if (p[i] < 0) delta -= l1_decay;
        }

        p[i] -= delta * learning_rate;
    }
}

/*****************************************************************************
       sigmoid kernels
*****************************************************************************/
__global__ void ker_sigmoid(int64 size, float* y_out, float* x_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = dev_sigmoid(x_in[idx]);
    }
}

__global__ void ker_sigmoid_cross_entropy_with_logits_derv(int64 size, float* y_out, float* est_logit_in, float* ans_prob_in, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float derv = dev_sigmoid_cross_entropy_with_logits_derv(est_logit_in[idx], ans_prob_in[idx]);
        y_out[idx] = __fmul_rn(derv, coef);
        /*
        float x = est_logit_in[idx];
        float term1 = (x > 0) ? 1 : __expf(x);
        float term2 = __fadd_rn(1.0f, __expf((x > 0) ? -x : x));
        y_out[idx] = __fmul_rn(__fsub_rn(__fdiv_rn(term1, term2), ans_prob_in[idx]), coef);
        */
    }
}

/*****************************************************************************
       softmax kernels
*****************************************************************************/
__global__ void ker_softmax(int64 size, float* y_out, float* x_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nth = idx % nvec;
        int64 pos = idx - nth;

        float* logits = x_in + pos;
        float max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        float idx_term = __expf(__fsub_rn(logits[nth], max_term));
        y_out[idx] = __fdiv_rn(idx_term, sum_exp);

        /*
        int64 pos = idx - idx % nvec;

        float* xp = x_in + pos;

        float max_term = xp[0];
        float sum_exp = 0;
        float idx_term = 0;

        for (int64 n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (int64 n = 0; n < nvec; n++) {
            float exp_term = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, exp_term);
            if (pos + n == idx) idx_term = exp_term;
        }

        y_out[idx] = __fdiv_rn(idx_term, sum_exp);
        */
    }
}

/*
__global__ void ker_softmax(int64 size, float* y_out, float* x_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float* xp = x_in + idx * nvec;
        float* yp = y_out + idx * nvec;

        float max_term = xp[0];
        float sum_exp = 0;

        for (int64 n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (int64 n = 0; n < nvec; n++) {
            yp[n] = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, yp[n]);
        }

        for (int64 n = 0; n < nvec; n++) {
            yp[n] = __fdiv_rn(yp[n], sum_exp);
        }
    }
}
*/

__global__ void ker_softmax_derv(int64 size, float* gx_out, float* gy_in, float* y_in, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 pos = idx - idx % nvec;

        float gsum = 0;

        for (int64 n = 0; n < nvec; n++, pos++) {
            float yac = __fmul_rn(y_in[pos], -y_in[idx]);
            if (pos == idx) yac = __fadd_rn(yac, y_in[idx]);

            gsum = __fadd_rn(gsum, __fmul_rn(yac, gy_in[pos]));
        }

        gx_out[idx] = gsum;
    }
}

__global__ void ker_softmax_cross_entropy_with_logits_derv(int64 size, float* y_out, float* est_logit_in, float* ans_probs_in, int64 nvec, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nth = idx % nvec;
        int64 pos = idx - nth;

        float* logits = est_logit_in + pos;
        float max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        float idx_term = __expf(__fsub_rn(logits[nth], max_term));
        float prob_term = __fdiv_rn(idx_term, sum_exp);

        y_out[idx] = __fmul_rn(__fsub_rn(prob_term, ans_probs_in[idx]), coef);
    }
}

__global__ void ker_softmax_cross_entropy_with_logits_idx_derv(int64 size, float* y_out, float* est_logit_in, int64* ans_probs_in, int64 nvec, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 wid = idx / nvec;
        int64 nth = idx % nvec;
        int64 pos = idx - nth;

        float* logits = est_logit_in + pos;
        float max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        float idx_term = __expf(__fsub_rn(logits[nth], max_term));
        float prob_term = __fdiv_rn(idx_term, sum_exp);

        float ans = (ans_probs_in[wid] == nth) ? 1.0f : 0;

        y_out[idx] = __fmul_rn(__fsub_rn(prob_term, ans), coef);

        /*
        int64 nw = idx / nvec;
        int64 nv = idx % nvec;

        float* xp = est_logit_in + idx - nv;

        float max_term = xp[0];
        float sum_exp = 0;

        for (int64 n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (int64 n = 0; n < nvec; n++) {
            float exp_term = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, exp_term);
            if (n == nv) y_out[idx] = exp_term;
        }

        float ans = (ans_probs_in[nw] == nv) ? 1.0f : 0;

        y_out[idx] = __fmul_rn(__fsub_rn(__fdiv_rn(y_out[idx], sum_exp), ans), coef);
        */
    }
}

__global__ void ker_softmax_cross_entropy_with_logits_1st_derv(int64 size, float* y_out, float* est_logit_in, int64 nvec, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nth = idx % nvec;
        int64 pos = idx - nth;

        float* logits = est_logit_in + pos;
        float max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        float idx_term = __expf(__fsub_rn(logits[nth], max_term));
        float prob_term = __fdiv_rn(idx_term, sum_exp);

        float ans = (nth == 0) ? 1.0f : 0;

        y_out[idx] = __fmul_rn(__fsub_rn(prob_term, ans), coef);

        /*
        int64 nv = idx % nvec;

        float* xp = est_logit_in + idx - nv;

        float max_term = xp[0];
        float sum_exp = 0;

        for (int64 n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (int64 n = 0; n < nvec; n++) {
            float exp_term = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, exp_term);
            if (n == nv) y_out[idx] = exp_term;
        }

        float ans = (nv == 0) ? 1.0f : 0;

        y_out[idx] = __fmul_rn(__fsub_rn(__fdiv_rn(y_out[idx], sum_exp), ans), coef);
        */
    }
}

/*****************************************************************************
       dropout kernels
*****************************************************************************/
__global__ void ker_dropout_old(int64 size, float* x_in, float* dm_in, float* y_out, int64 dm_size, float keep_ratio) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = __fdiv_rn(__fmul_rn(x_in[idx], dm_in[idx % dm_size]), keep_ratio);
    }
}

__global__ void ker_backprop_dropout_old(int64 size, float* gy_in, float* dm_in, float* gx_out, int64 dm_size, float keep_ratio) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx_out[idx] = __fdiv_rn(__fmul_rn(gy_in[idx], dm_in[idx % dm_size]), keep_ratio);
    }
}

/*****************************************************************************
       activate function kernels
*****************************************************************************/
__global__ void ker_activate(int64 size, float* y_out, float* x_in, int64 nFunc, float alpha) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = x_in[idx];
        float y = x;

        switch (nFunc) {
        case ACTFUNC_NONE:
            break;
        case ACTFUNC_RELU:
            if (x < 0) y = 0;
            break;
        case ACTFUNC_SIGMOID:
            y = dev_sigmoid(x);
            break;
        case ACTFUNC_TANH:
            y = dev_tanh(x);
            break;
        case ACTFUNC_LEAKY_RELU:
            if (x < 0) y = __fmul_rn(x, alpha);
            break;
        case ACTFUNC_GELU:
            {
                float x3 = __fmul_rn(x, __fmul_rn(x, x));
                float xt = __fmul_rn(2.0f, __fadd_rn(__fmul_rn(x, 0.797885f), __fmul_rn(0.035677f, x3)));
                float term1 = (xt > 0) ? 1 : __expf(xt);
                float term2 = __fadd_rn(1.0f, __expf((xt > 0) ? -xt : xt));
                float tanh = __fsub_rn(__fmul_rn(__fdiv_rn(term1, term2), 2.0f), 1.0f);
                float coef = __fadd_rn(tanh, 1.0f);
                y = __fmul_rn(__fmul_rn(x, 0.5f), coef);
            }
            break;
        default:
            break;
        }

        y_out[idx] = y;
    }
}

__global__ void ker_activate_derv(int64 size, float* gx_out, float* gy_in, float* x_in, float* y_in, int64 nFunc, float alpha) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = x_in[idx];
        float y = y_in[idx];
        float gy = gy_in[idx];
        float gx = 1;

        switch (nFunc) {
        case ACTFUNC_NONE:
            break;
        case ACTFUNC_RELU:
            if (x < 0) gx = 0;
            break;
        case ACTFUNC_SIGMOID:
            gx = dev_sigmoid_derv(x, y);
            break;
        case ACTFUNC_TANH:
            gx = dev_tanh_derv(x, y);
            break;
        case ACTFUNC_LEAKY_RELU:
            if (x < 0) gx = alpha;
            break;
        case ACTFUNC_GELU:
            {
                float x3 = __fmul_rn(x, __fmul_rn(x, x));
                float tan_x = __fadd_rn(__fmul_rn(x, 0.797885f), __fmul_rn(x3, 0.035677f));
                float sig_x = __fmul_rn(2.0f, tan_x);
                float term1 = (sig_x > 0) ? 1 : __expf(sig_x);
                float term2 = __fadd_rn(1.0f, __expf((sig_x > 0) ? -sig_x : sig_x));

                float gelu_a = __fsub_rn(__fmul_rn(__fdiv_rn(term1, term2), 2.0f), 1.0f);
                float gelu_b = __fadd_rn(__fmul_rn(__fmul_rn(x, x), 0.071356f), 0.797885f);

                gx = __fmul_rn(__fmul_rn(__fadd_rn(gelu_a, 1.0f), -0.5f), __fadd_rn(__fmul_rn(__fmul_rn(x, __fsub_rn(gelu_a, 1.0f)), gelu_b), 1.0f));
            }
            break;
        default:
            break;
        }

        gx_out[idx] = __fmul_rn(gx, gy);
    }
}

/*****************************************************************************
       convolution kernels
*****************************************************************************/

__global__ void ker_conv_kernel(int64 size, float* b_out, float* x_in, float* pm_in, int64 xh, int64 xw, int64 kh, int64 kw, int64 xchn, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (xh * xw * ychn * xchn);
        int64 xrow = (idx / (xw * ychn * xchn)) % xh;
        int64 xcol = (idx / (ychn * xchn)) % xw;
        int64 yn = (idx / xchn) % ychn;
        int64 xn = idx % xchn;

        float sum = 0;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        int64 xpos1 = ndat * xh * xw * xchn + xn;
        int64 kpos1 = xn * ychn + yn;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;

            int64 xpos2 = xpos1 + row * xw * xchn;
            int64 kpos2 = kpos1 + kr * kw * xchn * ychn;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;

                int64 xpos3 = xpos2 + col * xchn;
                int64 kpos3 = kpos2 + kc * xchn * ychn;

                sum = __fadd_rn(sum, __fmul_rn(x_in[xpos3], pm_in[kpos3]));
            }
        }

        b_out[idx] = sum;
    }
}

__global__ void ker_conv_sum(int64 size, float* y_out, float* b_in, int64 xchn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float sum = 0;
        for (int64 n = 0; n < xchn; n++) sum = __fadd_rn(sum, b_in[idx * xchn + n]);

        y_out[idx] = sum;
    }
}

__global__ void ker_conv_add_bias(int64 size, float* y_out, float* pm_in, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = __fadd_rn(y_out[idx], pm_in[idx % ychn]);
    }
}

__global__ void ker_conv_derv_x_kernel(int64 size, float* c_out, float* gy_in, float* k_in, int64 xh, int64 xw, int64 kh, int64 kw, int64 xchn, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (xh * xw * xchn * ychn);
        int64 xrow = (idx / (xw * xchn * ychn)) % xh;
        int64 xcol = (idx / (xchn * ychn)) % xw;
        int64 xn = (idx / ychn) % xchn;
        int64 yn = idx % ychn;

        float sum = 0;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        int64 ypos1 = ndat * xh * xw * ychn + yn;
        int64 kpos1 = xn * ychn + yn;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 yrow = xrow + bh - kr;
            if (yrow < 0 || yrow >= xh) continue;

            int64 ypos2 = ypos1 + yrow * xw * ychn;
            int64 kpos2 = kpos1 + kr * kw * xchn * ychn;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 ycol = xcol + bw - kc;
                if (ycol < 0 || ycol >= xw) continue;

                int64 ypos3 = ypos2 + ycol * ychn;
                int64 kpos3 = kpos2 + kc * xchn * ychn;

                sum = __fadd_rn(sum, __fmul_rn(gy_in[ypos3], k_in[kpos3]));
            }
        }

        c_out[idx] = sum;
    }
}

__global__ void ker_conv_derv_x_sum(int64 size, float* gx_out, float* c_in, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float sum = 0;
        for (int64 n = 0; n < ychn; n++) sum = __fadd_rn(sum, c_in[idx * ychn + n]);

        gx_out[idx] = sum;
    }
}

__global__ void ker_conv_derv_kw_x(int64 size, float* d_out, float* gy_in, float* x_in, int64 mb_size, int64 xh, int64 xw, int64 kh, int64 kw, int64 xchn, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 kr = (idx / (kw * xchn * ychn * mb_size * xh)) % kh;
        int64 kc = (idx / (xchn * ychn * mb_size * xh)) % kw;
        int64 xn = (idx / (ychn * mb_size * xh)) % xchn;
        int64 yn = (idx / (mb_size * xh)) % ychn;
        int64 ndat = (idx / xh) % mb_size;
        int64 xrow = idx % xh;

        float sum = 0;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        for (int64 xcol = 0; xcol < xw; xcol++) {
            int64 yrow = xrow - kr + bh;
            int64 ycol = xcol - kc + bw;
            
            if (yrow < 0 || yrow >= xh) continue;
            if (ycol < 0 || ycol >= xw) continue;

            int64 xpos = ((ndat * xh + xrow) * xw + xcol) * xchn + xn;
            int64 ypos = ((ndat * xh + yrow) * xw + ycol) * ychn + yn;

            sum = __fadd_rn(sum, __fmul_rn(gy_in[ypos], x_in[xpos]));
        }

        d_out[idx] = sum;
    }
}

__global__ void ker_conv_derv_kw_sum1(int64 size, float* d_inout, int64 xh) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 spos = idx * xh;

        float sum = 0;

        for (int64 n = 0; n < xh; n++) {
            sum = __fadd_rn(sum, d_inout[spos + n]);
        }

        d_inout[spos] = sum;
    }
}

__global__ void ker_conv_derv_kw_sum2(int64 size, float* gw_out, float* d_in, int64 mb_size, int64 xh) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 spos = idx * mb_size * xh;

        float sum = 0;

        for (int64 n = 0; n < mb_size; n++) {
            sum = __fadd_rn(sum, d_in[spos + n * xh]);
        }

        gw_out[idx] = sum;
    }
}

__global__ void ker_conv_derv_kb_sum1(int64 size, float* d_out, float* gy_in, int64 mb_size, int64 xh, int64 xw, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 yn = idx / (mb_size * xh);
        int64 nd = (idx / xh) % mb_size;
        int64 xr = idx % xh;

        float sum = 0;

        int64 ypos = (nd * xh + xr) * xw * ychn + yn;

        for (int64 xc = 0; xc < xw; xc++) {
            sum = __fadd_rn(sum, gy_in[ypos + xc * ychn]);
        }

        d_out[idx] = sum;
    }
}

__global__ void ker_conv_derv_kb_sum2(int64 size, float* d_inout, int64 mb_size, int64 xh, int64 xw, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 yn = idx / mb_size;
        int64 nd = idx % mb_size;

        float sum = 0;

        int64 bpos = (yn * mb_size + nd) * xh;

        for (int64 xr = 0; xr < xh; xr++) {
            sum = __fadd_rn(sum, d_inout[bpos + xr]);
        }

        d_inout[bpos] = sum;
    }
}

__global__ void ker_conv_derv_kb_sum3(int64 size, float* gb_out, float* d_in, int64 mb_size, int64 xh, int64 xw, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 yn = idx;

        float sum = 0;

        int64 bpos = yn * mb_size * xh;

        for (int64 nd = 0; nd < mb_size; nd++) {
            sum = __fadd_rn(sum, d_in[bpos + nd * xh]);
        }

        gb_out[idx] = sum;
    }
}

/*
__global__ void ker_conv_derv_x(int64 size, float* gx_out, float* c_buf, float* gy_in, float* k_in, int64 mb_size, int64 xh, int64 xw, int64 kh, int64 kw, int64 xchn, int64 ychn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (xh * xw * xchn);
        int64 xrow = (idx / (xw * xchn)) % xh;
        int64 xcol = (idx / xchn) % xw;
        int64 xn = idx % xchn;

        float sum = 0;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        int64 aidx0 = ndat * xh;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 row = xrow - kr + bh;
            if (row < 0 || row >= xh) continue;
            int64 aidx1 = (aidx0 + row) * xw, kidx1 = kr * kw;
            for (int64 kc = 0; kc < kw; kc++) {
                int64 col = xcol - kc + bw;
                if (col < 0 || col >= xw) continue;
                int64 aidx2 = (aidx1 + col) * ychn, kidx2 = ((kidx1 + kc) * xchn + xn) * ychn;
                for (int64 n = 0; n < ychn; n++) {
                    int64 aidx = aidx2 + n, kidx = kidx2 + n; // (kidx2 + n)* xchn + xn;
                    sum += gyp[aidx] * kp[kidx];
                }
            }
        }

        gxp[xidx] = sum;

        if (idx == 0) assert(0);
        __global__
            void ker_backprop_conv(int64 xsize, int64 ksize, int64 bsize,
                float* gyp, float* xp, float* kp, float* gxp, float* gkp, float* gbp,
                int64 mb_size, int64 xh, int64 xw, int64 xchn, int64 kh, int64 kw, int64 ychn) {
            int64 cidx = blockIdx.x * blockDim.x + threadIdx.x;
            if (cidx < xsize) {
            }
            else if (cidx < xsize + ksize) {
                int64 kidx = cidx - xsize;

                int64 krow = kidx / (kw * xchn * ychn);
                int64 kcol = (kidx / (xchn * ychn)) % kw;
                int64 xn = (kidx / ychn) % xchn;
                int64 yn = kidx % ychn;

                float sum = 0;

                int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;
                int64 xidx0 = xn, yidx0 = yn;

                for (int64 yrow = 0; yrow < xh; yrow++) {
                    int64 xrow = yrow + krow - bh;
                    if (xrow < 0 || xrow >= xh) continue;
                    int64 xidx1 = xidx0 + xrow * xw * xchn;
                    int64 yidx1 = yidx0 + yrow * xw * ychn;
                    for (int64 ycol = 0; ycol < xw; ycol++) {
                        int64 xcol = ycol + kcol - bw;
                        if (xcol < 0 || xcol >= xw) continue;
                        int64 xidx2 = xidx1 + xcol * xchn;
                        int64 yidx2 = yidx1 + ycol * ychn;
                        for (int64 n = 0; n < mb_size; n++) {
                            int64 xidx = xidx2 + n * xh * xw * xchn;
                            int64 yidx = yidx2 + n * xh * xw * ychn;
                            sum += xp[xidx] * gyp[yidx];
                        }
                    }
                }

                gkp[kidx] = sum;
            }
            else if (cidx < xsize + ksize + bsize) {
                int64 bidx = cidx - xsize - ksize;
                int64 yn = bidx % ychn;
                int64 ysize = mb_size * xh * xw * ychn;

                float sum = 0;

                for (int64 n = yn; n < ysize; n += ychn) {
                    sum += gyp[n];
                }

                gbp[bidx] = sum;
            }
        }
    }
}

__global__ void ker_conv_derv_k(int64 size, float* gk_out, float* c_buf, float* gy_in, float* x_in, int64 mb_size, int64 xh, int64 xw, int64 kh, int64 kw, int64 ychn, bool use_bias) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(0);
    }
}
*/

/*****************************************************************************
       max pool kernels
*****************************************************************************/
__global__ void ker_max(int64 size, float* y_out, int64* n_out, float* x_in, int64 xh, int64 xw, int64 chn, int64 kh, int64 kw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (xh * xw * chn);
        int64 xrow = (idx / (xw * chn)) % xh;
        int64 xcol = (idx / chn) % xw;
        int64 xn = idx % chn;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        int64 xpos_base = ndat * xh * xw * chn + xn;
        //int64 xpos_base = ((ndat * xh + xrow - bh) * xw + xcol - bw) * xchn + xn;

        float maxval = 0;
        int64 argmax = -1;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;
            for (int64 kc = 0; kc < kw; kc++) {
                int64 col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;
                int64 xpos = xpos_base + row * xw * chn + col * chn;
                float x = x_in[xpos];
                if (argmax < 0 || maxval < x) {
                    maxval = x;
                    argmax = (int64) (kr *kw + kc);
                }
            }
        }

        y_out[idx] = maxval;
        n_out[idx] = argmax;
    }
}

__global__ void ker_max_derv(int64 size, float* gx_out, int64* n_in, float* gy_in, int64 xh, int64 xw, int64 chn, int64 kh, int64 kw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (xh * xw * chn);
        int64 xrow = (idx / (xw * chn)) % xh;
        int64 xcol = (idx / chn) % xw;
        int64 xn = idx % chn;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        int64 ypos_base = ndat * xh * xw * chn + xn;

        float sum = 0;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 row = xrow - kr + bh;
            if (row < 0 || row >= xh) continue;
            for (int64 kc = 0; kc < kw; kc++) {
                int64 col = xcol - kc + bw;
                if (col < 0 || col >= xw) continue;
                int64 ypos = ypos_base + row * xw * chn + col * chn;
                int64 argmax = (int64) (kr * kw + kc);
                if (n_in[ypos] != argmax) continue;
                sum = __fadd_rn(sum, gy_in[ypos]);
            }
        }

        gx_out[idx] = sum;
    }
}

/*****************************************************************************
       avg pool kernels
*****************************************************************************/
__global__ void ker_avg(int64 size, float* y_out, int64* n_out, float* x_in, int64 xh, int64 xw, int64 xchn, int64 kh, int64 kw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (xh * xw * xchn);
        int64 xrow = (idx / (xw * xchn)) % xh;
        int64 xcol = (idx / xchn) % xw;
        int64 xn = idx % xchn;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        int64 xpos = ((ndat * xh + xrow - bh) * xw + xcol - bw) * xchn + xn;

        float sum = 0;
        int64 cnt = 0;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;
            for (int64 kc = 0; kc < kw; kc++) {
                int64 col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;
                float x = x_in[xpos + (kr * xw + kc) * xchn];
                sum = __fadd_rn(sum, x);
                cnt++;
            }
        }

        y_out[idx] = __fdiv_rn(sum, (float) cnt);
        n_out[idx] = cnt;
    }
}

__global__ void ker_avg_derv(int64 size, float* gx_out, int64* n_in, float* gy_in, int64 xh, int64 xw, int64 chn, int64 kh, int64 kw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (xh * xw * chn);
        int64 xrow = (idx / (xw * chn)) % xh;
        int64 xcol = (idx / chn) % xw;
        int64 xn = idx % chn;

        int64 bh = kh / 2, bw = kw / 2;
        int64 ypos_base = ((ndat * xh + xrow - bh) * xw + xcol - bw) * chn + xn;

        float sum = 0;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;
            for (int64 kc = 0; kc < kw; kc++) {
                int64 col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;
                int64 ypos = ypos_base + (kr * xw + kc) * chn;
                sum = __fadd_rn(sum, __fdiv_rn(gy_in[ypos], (float) n_in[ypos]));
            }
        }

        gx_out[idx] = sum;
    }
}

/*****************************************************************************
       add kernels
*****************************************************************************/
__global__ void ker_avg_exact(int64 size, float* t_out, float* x_in, int64 xh, int64 xw, int64 xchn, int64 sh, int64 sw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 th = xh / sh;
        int64 tw = xw / sw;

        int64 ndat = idx / (th * tw * xchn);
        int64 trow = (idx / (tw * xchn)) % th;
        int64 tcol = (idx / xchn) % tw;
        int64 xn = idx % xchn;

        float sum = 0;

        for (int64 r = 0; r < sh; r++) {
            int64 xrow = trow * sh + r;
            for (int64 c = 0; c < sw; c++) {
                int64 xcol = tcol * sw + c;
                int64 xidx = ((ndat * xh + xrow) * xw + xcol) * xchn + xn;
                sum = __fadd_rn(sum, x_in[xidx]);
            }
        }

        t_out[idx] = __fdiv_rn(sum, (float)(sh*sw));
    }
}

__global__ void ker_avg_exact_derv(int64 size, float* gt_out, float* gy_in, int64 xh, int64 xw, int64 xchn, int64 sh, int64 sw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 yh = xh / sh;
        int64 yw = xw / sw;

        int64 ndat = idx / (xh * xw * xchn);
        int64 xrow = (idx / (xw * xchn)) % xh;
        int64 xcol = (idx / xchn) % xw;
        int64 xn = idx % xchn;

        int64 yrow = xrow / sh;
        int64 ycol = xcol / sh;

        int64 yidx = ((ndat * yh + yrow) * yw + ycol) * xchn + xn;

        gt_out[idx] = __fdiv_rn(gy_in[yidx], (float)(sh * sw));
    }
}

__global__ void ker_tile_chn(int64 size, float* t_out, float* x_in, int64 ychn, int64 xchn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / ychn;
        int64 tn = idx % ychn;

        int64 ratio = ychn / xchn;
        int64 xn = tn / ratio;

        int64 xidx = row * xchn + xn;

        t_out[idx] = x_in[xidx];
    }
}

__global__ void ker_untile_chn(int64 size, float* gt_out, float* gy_in, int64 ychn, int64 xchn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / xchn;
        int64 tn = idx % xchn;

        int64 ratio = ychn / xchn;

        float sum = 0;

        for (int64 n = 0; n < ratio; n++) {
            int64 yidx = row * ychn + tn * ratio + n;
            sum = __fadd_rn(sum, gy_in[yidx]);
        }

        gt_out[idx] = sum;
    }
}

/*****************************************************************************
       stride kernels
*****************************************************************************/
__global__ void ker_stride(int64 size, float* y_out, float* x_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 kh, int64 kw, int64 sh, int64 sw, bool valid_padding) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (yh * yw * chn);
        int64 row = (idx / (yw * chn)) % yh;
        int64 col = (idx / chn) % yw;
        int64 xn = idx % chn;

        int64 bh = (sh - 1) / 2, bw = (sw - 1) / 2;
        if (valid_padding) bh += (kh - 1) / 2, bw += (kw - 1) / 2;

        int64 rpos = row * sh + bh;
        int64 cpos = col * sw + bw;

        int64 xpos = ((ndat * xh + rpos) * xw + cpos) * chn + xn;

        y_out[idx] = x_in[xpos];
    }
}

__global__ void ker_stride_derv(int64 size, float* gx_out, float* gy_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 kh, int64 kw, int64 sh, int64 sw, bool valid_padding) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx_out[idx] = 0;

        int64 ndat = idx / (xh * xw * chn);
        int64 row = (idx / (xw * chn)) % xh;
        int64 col = (idx / chn) % xw;
        int64 xn = idx % chn;

        int64 bh = (sh - 1) / 2, bw = (sw - 1) / 2;

        if (valid_padding) {
            bh += (kh - 1) / 2, bw += (kw - 1) / 2;
        }

        if ((row - bh) % sh != 0) return;
        if ((col - bw) % sw != 0) return;

        int64 rpos = (row - bh) / sh;
        int64 cpos = (col - bw) / sw;

        if (rpos < 0 | rpos >= yh) return;
        if (cpos < 0 | cpos >= yw) return;

        int64 spos = ((ndat * yh + rpos) * yw + cpos) * chn + xn;

        gx_out[idx] = gy_in[spos];
    }
}

__global__ void ker_stride_expand(int64 size, float* y_out, float* x_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 sh, int64 sw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ndat = idx / (yh * yw * chn);
        int64 row = (idx / (yw * chn)) % yh;
        int64 col = (idx / chn) % yw;
        int64 xn = idx % chn;

        int64 rpos = row / sh;
        int64 cpos = col / sw;

        int64 xpos = ((ndat * xh + rpos) * xw + cpos) * chn + xn;

        y_out[idx] = x_in[xpos];
    }
}

__global__ void ker_stride_expand_derv(int64 size, float* gx_out, float* gy_in, int64 xh, int64 xw, int64 yh, int64 yw, int64 chn, int64 sh, int64 sw) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx_out[idx] = 0;

        int64 ndat = idx / (xh * xw * chn);
        int64 row = (idx / (xw * chn)) % xh;
        int64 col = (idx / chn) % xw;
        int64 xn = idx % chn;

        int64 rpos = row * sh;
        int64 cpos = col * sw;

        int64 spos = ((ndat * yh + rpos) * yw + cpos) * chn + xn;

        gx_out[idx] = 0;

        for (int64 n = 0; n < sh; n++) {
            for (int64 m = 0; m < sw; m++) {
                gx_out[idx] += gy_in[spos + (n * yw + m) * chn];
            }
        }
    }
}

/*****************************************************************************
       batch normal kernels
*****************************************************************************/
__global__ void ker_batch_normal(int64 size, float* y_out, float* x_in, int64 ngroup) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(0);
    }
}

__global__ void ker_batch_normal_derv(int64 size, float* gx_out, float* gy_in, float* y_in, int64 ngroup) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(0);
    }
}

__global__ void ker_bn_collect(int64 size, float* avg_inout, float* var_inout, float* mavg_inout, float* mvar_inout, float* x_in, int64 hsize, float momentum) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nrest = hsize / size;

        float sum = 0, sqsum = 0;

        for (int64 n = idx; n < hsize; n+=size) {
            float x = x_in[n];
            sum += x;
            sqsum += x * x;
        }

        float avg = sum / (float) nrest;
        float var = sqsum / (float)nrest - avg * avg;

        avg_inout[idx] = avg;
        var_inout[idx] = var;

        mavg_inout[idx] = mavg_inout[idx] * momentum + avg * (1 - momentum);
        mvar_inout[idx] = mvar_inout[idx] * momentum + var * (1 - momentum);
    }
}

__global__ void ker_bn_normalize(int64 size, float* h_inout, float* avg_in, float* var_in, int64 bsize, float epsilon) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 bidx = idx % bsize;
        float std = __fsqrt_rn(__fadd_rn(var_in[bidx], epsilon));

        h_inout[idx] = __fdiv_rn(__fsub_rn(h_inout[idx], avg_in[bidx]), std);
    }
}

__global__ void ker_bn_rescale(int64 size, float* h_inout, float* scale_in, float* shift_in, int64 bsize) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 bidx = idx % bsize;

        h_inout[idx] = __fmaf_rn(h_inout[idx], scale_in[bidx], shift_in[bidx]);
    }
}

__global__ void ker_bn_rescale_derv_pm(int64 size, float* gscale_out, float* gshift_out, float* gx_in, float* x_in, int64 hsize) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float scale_sum = 0, shift_sum = 0;

        for (int64 n = idx; n < hsize; n += size) {
            scale_sum = __fadd_rn(scale_sum, __fmul_rn(gx_in[n], x_in[n]));
            shift_sum = __fadd_rn(shift_sum, gx_in[n]);
        }

        gscale_out[idx] = scale_sum;
        gshift_out[idx] = shift_sum;
    }
}

__global__ void ker_bn_rescale_derv_x(int64 size, float* gh_inout, float* scale_in, int64 bsize) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 bidx = idx % bsize;

        gh_inout[idx] = __fmul_rn(gh_inout[idx], scale_in[bidx]);
    }
}

__global__ void ker_bn_norm_derv(int64 size, float* gh_inout, float* var_in, int64 bsize, float epsilon) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 bidx = idx % bsize;
        float std = __fsqrt_rn(__fadd_rn(var_in[bidx], epsilon));

        gh_inout[idx] = __fdiv_rn(gh_inout[idx], std);
    }
}

/*****************************************************************************
       dropout kernels
*****************************************************************************/
__global__ void ker_dropout(int64 size, float* y_out, float* x_in, float* m_in, float keep_ratio) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y_out[idx] = __fdiv_rn(__fmul_rn(x_in[idx], m_in[idx]), keep_ratio);
    }
}

__global__ void ker_dropout_derv(int64 size, float* gx_out, float* gy_in, float* m_in, float keep_ratio) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx_out[idx] = __fdiv_rn(__fmul_rn(gy_in[idx], m_in[idx]), keep_ratio);
    }
}

/*****************************************************************************
       parallel layer
*****************************************************************************/
__global__ void ker_get_branch(int64 size, float* y_out, float* b_in, int64 ychn, int64 bchn, int64 chn_from) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / bchn;
        int64 bcol = idx % bchn;
        
        int64 ypos = row * ychn + chn_from + bcol;

        y_out[ypos] = b_in[idx];
    }
}
__global__ void ker_set_branch(int64 size, float* gb_out, float* gy_in, int64 ychn, int64 bchn, int64 chn_from) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / bchn;
        int64 bcol = idx % bchn;

        int64 ypos = row * ychn + chn_from + bcol;

        gb_out[idx] = gy_in[ypos];
    }
}

/*****************************************************************************
       rnn/lstm layer
*****************************************************************************/
__global__ void ker_rnn_combine_ex_inp(int64 size, float* ex_inp_out, float* x_in, float* rec_in, int64 timesteps, int64 timefeats, int64 recur_size, bool inseq, int64 tn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / (timefeats + recur_size);
        int64 col = idx % (timefeats + recur_size);

        if (col < timefeats) {
            if (inseq) 
                ex_inp_out[idx] = x_in[((row * timesteps) + tn) * timefeats + col];
            else
                ex_inp_out[idx] = x_in[row * timefeats + col];
        }
        else {
            ex_inp_out[idx] = rec_in[row * recur_size + (col - timefeats)];
        }
    }
}

__global__ void ker_rnn_split_ex_inp(int64 size, float* gx_out, float* grec_out, float* gex_inp_in, int64 timesteps, int64 timefeats, int64 recur_size, bool inseq, int64 tn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / (timefeats + recur_size);
        int64 col = idx % (timefeats + recur_size);

        if (col < timefeats) {
            if (inseq) {
                int64 xidx = ((row * timesteps) + tn) * timefeats + col;
                gx_out[xidx] = gex_inp_in[idx];
            }
            else {
                int64 xidx = row * timefeats + col;
                gx_out[xidx] = __fadd_rn(gx_out[xidx], gex_inp_in[idx]);
            }
        }
        else {
            int64 ridx = row * recur_size + (col - timefeats);
            grec_out[ridx] = gex_inp_in[idx];
        }
    }
}

__global__ void ker_rnn_fill_output_slice(int64 size, float* y_out, float* x_in, int64 timesteps, int64 recur_size, int64 tn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / recur_size;
        int64 col = idx % recur_size;

        y_out[((row * timesteps) + tn) * recur_size + col] = x_in[idx];
    }
}

__global__ void ker_rnn_add_time_slice(int64 size, float* gy_out, float* gx_in, int64 timesteps, int64 recur_size, int64 tn) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / recur_size;
        int64 col = idx % recur_size;

        gy_out[idx] = __fadd_rn(gy_out[idx], gx_in[((row * timesteps) + tn) * recur_size + col]);
    }
}

__global__ void ker_rnn_copy_last_grad(int64 size, float* grec_inout, float* gx_in, int64 timesteps, int64 recur_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        //int64 row = idx / recur_size;

        grec_inout[idx] = gx_in[idx];
    }
}

__global__ void ker_lstm_split_affine(int64 size, float* fgate_out, float* igate_out, float* ogate_out, float* block_out, float* affine_in, int64 recur_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / (4 * recur_size);
        int64 col = idx % (4 * recur_size);
        int64 slice_idx = row * recur_size + col % recur_size;

        if (col < recur_size)
            fgate_out[slice_idx] = affine_in[idx];
        else if (col < 2 * recur_size)
            igate_out[slice_idx] = affine_in[idx];
        else if (col < 3 * recur_size)
            ogate_out[slice_idx] = affine_in[idx];
        else if (col < 4 * recur_size)
            block_out[slice_idx] = affine_in[idx];
    }
}

__global__ void ker_lstm_combine_affine(int64 size, float* gaffine_out, float* gfgate_in, float* gigate_in, float* gogate_in, float* gblock_in, int64 recur_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / (4 * recur_size);
        int64 col = idx % (4 * recur_size);
        int64 slice_idx = row * recur_size + col % recur_size;

        if (col < recur_size)
            gaffine_out[idx] = gfgate_in[slice_idx];
        else if (col < 2 * recur_size)
            gaffine_out[idx] = gigate_in[slice_idx];
        else if (col < 3 * recur_size)
            gaffine_out[idx] = gogate_in[slice_idx];
        else if (col < 4 * recur_size)
            gaffine_out[idx] = gblock_in[slice_idx];

        /*
        if (col < recur_size)
            gaffine_out[idx] = gfgate_in[slice_idx];
        else if (col < 2 * recur_size)
            gaffine_out[idx] = gigate_in[slice_idx];
        else if (col < 3 * recur_size)
            gaffine_out[idx] = gogate_in[slice_idx];
        else if (col < 4 * recur_size)
            gaffine_out[idx] = gblock_in[slice_idx];
        */
    }
}

__global__ void ker_rnn_select_last_vecs(int64 size, float* selected_out, float* time_pool_in, int64 timesteps, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 row = idx / nvec;
        int64 col = idx % nvec;

        int64 tn = timesteps - 1;
        int64 pidx = (row * timesteps + tn) * nvec + col;

        selected_out[idx] = time_pool_in[pidx];
    }
}

__global__ void ker_lstm_new_state(int64 size, float* state_out, float* state_in, float* fgate_in, float* block_in, float* igate_in) { // state = state * forget_gate + block_input * input_gate;
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        state_out[idx] = __fadd_rn(__fmul_rn(state_in[idx], fgate_in[idx]), __fmul_rn(block_in[idx], igate_in[idx]));
    }
}

__global__ void ker_lstm_state_derv(int64 size, float* gstate_inout, float* grec_in, float* ogate_in, float* rec_in) { // G_recur_tmp = G_recurrent * output_gate; G_state += kmath->tanh_derv(recur_tmp) * G_recur_tmp;
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float grec1 = grec_in[idx] * ogate_in[idx];
        float gtanh = __fmul_rn(__fadd_rn(1.0f, rec_in[idx]), __fsub_rn(1.0f, rec_in[idx]));

        gstate_inout[idx] = __fadd_rn(gstate_inout[idx], __fmul_rn(grec1, gtanh));
    }
}

/*****************************************************************************
       attention layer: forward
*****************************************************************************/
__global__ void ker_attention_split(int64 size, float* q_out, float* k_out, float* v_out, float* qkv_in, int64 L, int64 H, int64 R) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (L * 3 * H * R);
        int64 nt = (idx / (3 * H * R)) % L;
        int64 n3 = (idx / (H * R)) % 3;
        int64 nh = (idx / R) % H;
        int64 nr = idx % R;

        if (n3 == 0) {
            int64 qpos = ((nd * H + nh)* L + nt)* R + nr;
            q_out[qpos] = qkv_in[idx];
        }
        else if (n3 == 1) {
            int64 kpos = ((nd * H + nh) * R + nr) * L + nt;
            k_out[kpos] = qkv_in[idx];
        }
        else {
            int64 vpos = ((nd * H + nh) * L + nt) * R + nr;
            v_out[vpos] = qkv_in[idx];
        }
    }
}

__global__ void ker_attention_combine(int64 size, float* gqkv_out, float* gq_in, float* gk_in, float* gv_in, int64 L, int64 H, int64 R) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (L * 3 * H * R);
        int64 nt = (idx / (3 * H * R)) % L;
        int64 n3 = (idx / (H * R)) % 3;
        int64 nh = (idx / R) % H;
        int64 nr = idx % R;

        if (n3 == 0) {
            int64 qpos = ((nd * H + nh) * L + nt) * R + nr;
            gqkv_out[idx] = gq_in[qpos];
        }
        else if (n3 == 1) {
            int64 kpos = ((nd * H + nh) * R + nr) * L + nt;
            gqkv_out[idx] = gk_in[kpos];
        }
        else {
            int64 vpos = ((nd * H + nh) * L + nt) * R + nr;
            gqkv_out[idx] = gv_in[vpos];
        }
    }
}

__global__ void ker_attention_mask_future(int64 size, float* score_inout, int64 L) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nt1 = (idx / L) % L;
        int64 nt2 = idx % L;
        
        if (nt2 > nt1) score_inout[idx] -= 10000.0f;
    }
}

__global__ void ker_attention_reshape_out(int64 size, float* a_out, float* m_in, int64 L, int64 H, int64 R) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (L * H * R);
        int64 nt = (idx / (H * R)) % L;
        int64 nh = (idx / R) % H;
        int64 nr = idx % R;
        
        int64 mpos = ((nd * H + nh) * L + nt) * R + nr;

        a_out[idx] = m_in[mpos];
    }
}

__global__ void ker_attention_reshape_mul(int64 size, float* gm_out, float* ga_in, int64 L, int64 H, int64 R) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (H * L * R);
        int64 nh = (idx / (L * R)) % H;
        int64 nt = (idx / R) % L;
        int64 nr = idx % R;

        int64 apos = ((nd * L + nt) * H + nh) * R + nr;

        gm_out[idx] = ga_in[apos];
    }
}

/*
__global__ void ker_attention_forward_mult_val(int64 size, float* att_in, float* qkv_in, float* buf_out, int64 B, int64 L, int64 V, int64 H, int64 R) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (L * V);
        int64 nt = idx / V % L;
        int64 nv = idx % V;

        int64 nh = nv / R;

        int64 apos = nd * (H * L * L) + nh * (L * L) + nt * L;
        int64 vpos = nd * (L * 3 * V) + nv + 2 * V;

        float sum = 0;

        for (int64 nt1 = 0; nt1 < L; nt1++) {
            sum = __fadd_rn(sum, __fmul_rn(att_in[apos], qkv_in[vpos]));
            apos++, vpos += 3 * V;
        }

        buf_out[idx] = sum;
    }
}

__global__ void ker_attention_backprop_matmul_probs(int64 size, float* gy_in, float* qkv_in, float* gprobs_out, int64 B, int64 L, int64 V, int64 H, int64 R) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (H * L * L);
        int64 nh = idx / (L * L) % H;
        int64 nt1 = idx / L % L;
        int64 nt2 = idx % L;

        int64 vpos = nd * (3 * L * V) + nt2 * (3 * V) + nh * R + V + V;
        int64 ypos = nd * (L * V) + nh * R + nt1 * V;

        float gsum = 0;

        for (int64 nr = 0; nr < R; nr++) {
            gsum = __fadd_rn(gsum, __fmul_rn(qkv_in[vpos], gy_in[ypos]));
            vpos += 1, ypos += 1;
        }

        gprobs_out[idx] = gsum;
    }
}

__global__ void ker_attention_backprop_matmul_value(int64 size, float* gy_in, float* probs_in, float* gqkv_out, int64 B, int64 L, int64 V, int64 H, int64 R) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (L * V);
        int64 nh = idx / R % H;
        int64 nt = idx / V % L;
        int64 nr = idx % R;

        int64 ppos = nd * (H * L * L) + nh * (L * L) + nt;
        int64 ypos = nd * (H * L * R) + nh * R + nr;
        int64 gpos = nd * (3 * L * V) + nt * (3 * V) + nh * R + nr + V + V;

        float gsum = 0;

        for (int64 nt2 = 0; nt2 < L; nt2++) {
            gsum = __fadd_rn(gsum, __fmul_rn(probs_in[ppos], gy_in[ypos]));
            ppos += L, ypos += V;
        }

        gqkv_out[gpos] = gsum;
    }
}
*/

/*****************************************************************************
       attention layer: [Q(in qkv)] matmul [K(in qkv)]
*****************************************************************************/
/*
__global__ void ker_attention_forward_qk_mult(int64 size, float* qkv_in, float* att_out, int64 B, int64 L, int64 V, int64 H, int64 R, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (H * L * L);
        int64 nh = idx / (L * L) % H;
        int64 nt1 = idx / L % L;
        int64 nt2 = idx % L;

        int64 qpos = nd * (3 * L * V) + nh * R + nt1 * (3 * V);
        int64 kpos = nd * (3 * L * V) + nh * R + nt2 * (3 * V) + V;

        float sum = 0;

        for (int64 nv = 0; nv < R; nv++) {
            sum = __fadd_rn(sum, __fmul_rn(qkv_in[qpos++], qkv_in[kpos++]));
        }

        att_out[idx] = sum * coef;
    }
}

__global__ void ker_attention_backprop_mult_kv(int64 size, float* gscore_in, float* qkv_in, float* gqkv_out, int64 B, int64 L, int64 V, int64 H, int64 R, float coef) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (L * V);
        int64 nh = idx / R % H;
        int64 nt = idx / V % L;
        int64 nr = idx % R;

        int64 qspos = nd * (H * L * L) + nh * (L * L) + nt * L;
        int64 kspos = nd * (H * L * L) + nh * (L * L) + nt;

        int64 qkpos = nd * (L * 3 * V) + nh * R + nr + V;
        int64 kqpos = nd * (L * 3 * V) + nh * R + nr;

        float gqsum = 0, gksum = 0;

        for (int64 n = 0; n < L; n++) {
            gqsum = __fadd_rn(gqsum, __fmul_rn(gscore_in[qspos], qkv_in[qkpos]));
            gksum = __fadd_rn(gksum, __fmul_rn(gscore_in[kspos], qkv_in[kqpos]));

            qspos++, qkpos += 3 * V;
            kspos += L, kqpos += 3 * V;
        }

        int64 qpos = nd * (L * 3 * V) + nh * R + nt * (3 * V) + nr;
        int64 kpos = nd * (L * 3 * V) + nh * R + nt * (3 * V) + nr + V;

        gqkv_out[qpos] = __fmul_rn(gqsum, coef);
        gqkv_out[kpos] = __fmul_rn(gksum, coef);
    }
}
*/

/*****************************************************************************
       embed layer
*****************************************************************************/
__global__ void ker_embedding_fetch(int64 size, float* wiv_out, float* wov_out, int64* hint_in, int64* noms_in, float* iwdic_in, float* owdic_in, int64 in_cnt, int64 out_cnt, int64 vec_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 vec_cnt = in_cnt + out_cnt;

        int64 nr = idx / (vec_cnt * vec_size);
        int64 nv = (idx / vec_size) % vec_cnt;
        int64 nc = idx % vec_size;

        if (nv < in_cnt) {
            int64 word_id = (int64)hint_in[nr * in_cnt + nv];
            int64 didx = word_id * vec_size + nc;
            int64 widx = (nr * in_cnt + nv) * vec_size + nc;
            wiv_out[widx] = iwdic_in[didx];
        }
        else {
            nv -= in_cnt;
            int64 word_id = (int64)noms_in[nr * out_cnt + nv];
            int64 didx = word_id * vec_size + nc;
            int64 widx = (nr * out_cnt + nv) * vec_size + nc;
            wov_out[widx] = owdic_in[didx];
        }
    }
}

__global__ void ker_embedding_dotmul(int64 size, float* y_out, float* sub_in, float* wov_in, int64 vec_cnt, int64 vec_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / vec_cnt;
        //int64 nc = idx % vec_cnt;

        int64 sidx = nr * vec_size;
        int64 xidx = idx * vec_size;

        float sum = 0;

        for (int64 n = 0; n < vec_size; n++, xidx++) {
            sum = __fadd_rn(sum, __fmul_rn(sub_in[sidx], wov_in[xidx]));
        }

        y_out[idx] = sum;
    }
}

__global__ void ker_embedding_dotmul_derv(int64 size, float* gsub_out, float* gwov_out, float* gy_in, float* sub_in, float* wov_in, int64 vec_cnt, int64 vec_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / (vec_cnt * vec_size);
        int64 nv = (idx / vec_size) % vec_cnt;
        int64 nc = idx % vec_size;

        int64 sidx = nr * vec_size + nc;
        int64 yidx = nr * vec_cnt + nv;

        gwov_out[idx] = __fmul_rn(sub_in[sidx], gy_in[yidx]);

        if (nv == 0) {
            float sum = 0;

            for (int64 n = 0; n < vec_cnt; n++) {
                int64 widx = (nr * vec_cnt + n) * vec_size + nc;
                int64 yidx = nr * vec_cnt + n;

                sum = __fadd_rn(sum, __fmul_rn(wov_in[widx], gy_in[yidx]));
            }
            int64 gidx = nr * vec_size + nc;
            gsub_out[gidx] = sum;
        }
    }
}

/*****************************************************************************
       merge layer
*****************************************************************************/
__global__ void ker_merge_avg(int64 size, float* m_out, float* x_in, int64 vec_cnt, int64 vec_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / vec_size;
        int64 nc = idx % vec_size;

        int64 xidx = nr * vec_cnt * vec_size + nc;
            
        float sum = 0;
        
        for (int64 n = 0; n < vec_cnt; n++, xidx += vec_size) {
            sum = __fadd_rn(sum, x_in[xidx]);
        }
        
        m_out[idx] = __fdiv_rn(sum, (float)vec_cnt);
    }
}

__global__ void ker_merge_avg_derv(int64 size, float* gx_out, float* gm_in, int64 vec_cnt, int64 vec_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / (vec_cnt * vec_size);
        int64 nc = idx % vec_size;

        int64 midx = nr * vec_size + nc;

        gx_out[idx] = __fdiv_rn(gm_in[midx], (float)vec_cnt);
    }
}

/*****************************************************************************
       extract layer
*****************************************************************************/
__global__ void ker_extract(int64 size, float* e_out, float* x_in, int64 ax_size, int64 index, int64 nprod) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / nprod;
        int64 nc = idx % nprod;

        int64 xpos = (nr * ax_size + index) * nprod + nc;

        e_out[idx] = x_in[xpos];
    }
}

__global__ void ker_unextract(int64 size, float* gx_out, float* ge_in, int64 ax_size, int64 index, int64 nprod) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / (ax_size * nprod);
        int64 nx = (idx / nprod) % ax_size;
        int64 nc = idx % nprod;

        int64 epos = nr * nprod + nc;

        if (nx == index)
            gx_out[idx] = ge_in[epos];
        else
            gx_out[idx] = 0;
    }
}

/*****************************************************************************
       math
*****************************************************************************/
__global__ void ker_vstack(int64 size, float* vs_out, float* x1_in, float* x2_in, int64 vol1, int64 vol2, int64 rest) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / rest;
        int64 nc = idx % rest;

        if (nr < vol1)
            vs_out[idx] = x1_in[nr * rest + nc];
        else
            vs_out[idx] = x2_in[(nr - vol1) * rest + nc];
    }
}

__global__ void ker_hstack(int64 size, float* vs_out, float* x1_in, float* x2_in, int64 vec1, int64 vec2) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 vec_size = vec1 + vec2;

        int64 nr = idx / vec_size;
        int64 nc = idx % vec_size;

        if (nc < vec1)
            vs_out[idx] = x1_in[nr * vec1 + nc];
        else
            vs_out[idx] = x2_in[nr * vec2 + (nc - vec1)];
    }
}

__global__ void ker_hsplit(int64 size, float* p1_out, float* p2_out, float* src_in, int64 vec_size, int64 p1_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nrow = idx / vec_size;
        int64 ncol = idx % vec_size;

        if (ncol < p1_size) {
            int64 p1_idx = nrow * p1_size + ncol;
            p1_out[p1_idx] = src_in[idx];
        }
        else {
            int64 p2_size = vec_size - p1_size;
            int64 p2_col = ncol - p1_size;
            int64 p2_idx = nrow * p2_size + p2_col;
            p2_out[p2_idx] = src_in[idx];
        }
    }
}

__global__ void ker_embed_fetch_multi_dic(int64 size, float* v_out, int64* wids_in, float* dics_in, int64 dic_count, int64* voc_counts, int64 vec_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nw = idx / vec_size;
        int64 nv = idx % vec_size;

        int64 wpos = nw * dic_count;
        int64 wbase = 0;

        float sum = 0;

        for (int64 n = 0; n < dic_count; n++) {
            int64 wid = (int64) wids_in[wpos++] + wbase;
            int64 dpos = wid * vec_size + nv;
            wbase += voc_counts[n];
            sum = __fadd_rn(sum, dics_in[dpos]);
        }

        v_out[idx] = sum;
    }
}

__global__ void ker_set_row(int64 size, float* p_inout, int64 nrest, float value) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        p_inout[idx * nrest] = value;
    }
}

__global__ void ker_extract_selected_pickup(int64 size, float* dst_out, float* arr_in, int64* map_in, int64 drest) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nrow = idx / drest;
        int64 ncol = idx % drest;

        int64 sidx = map_in[nrow] * drest + ncol;
        dst_out[idx] = arr_in[sidx];
    }
}

__global__ void ker_extract_selected_fill(int64 size, float* arr_inout, float* slice_in, int64* map_in, int64 drest) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nrow = idx / drest;
        int64 ncol = idx % drest;

        int64 sidx = map_in[nrow] * drest + ncol;
        arr_inout[sidx] = slice_in[idx];
    }
}

__global__ void ker_extract_selected_pickup_int(int64 size, int64* dst_out, int64* arr_in, int64* map_in, int64 drest) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nrow = idx / drest;
        int64 ncol = idx % drest;

        int64 sidx = map_in[nrow] * drest + ncol;
        dst_out[idx] = arr_in[sidx];
    }
}

__global__ void ker_expand(int64 size, float* y_out, float* x_in, int64 heights, int64 widths, int64 chns, int64 hratio, int64 wratio) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (heights * widths * chns);
        int64 nh = idx / (widths * chns) % heights;
        int64 nw = idx / chns % widths;
        int64 nc = idx % chns;

        int64 xheights = heights / hratio;
        int64 xwidths = widths / wratio;

        int64 xh = nh / hratio;
        int64 xw = nw / wratio;

        int64 xidx = ((nd * xheights + xh) * xwidths + xw) * chns + nc;

        y_out[idx] = x_in[xidx];
    }
}

__global__ void ker_expand_derv(int64 size, float* gx_out, float* gy_in, int64 heights, int64 widths, int64 chns, int64 hratio, int64 wratio) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (heights * widths * chns);
        int64 nh = idx / (widths * chns) % heights;
        int64 nw = idx / chns % widths;
        int64 nc = idx % chns;

        int64 yheights = heights * hratio;
        int64 ywidths = widths * wratio;

        int64 yh = nh * hratio;
        int64 yw = nw * wratio;

        gx_out[idx] = 0;

        int64 yidx_base = ((nd * yheights + yh) * ywidths + yw) * chns + nc;

        for (int64 h = 0; h < hratio; h++) {
            int64 yidx = yidx_base + h * ywidths * chns;
            for (int64 w = 0; w < wratio; w++) {
                gx_out[idx] += gy_in[yidx + w * chns];
            }
        }
    }
}

__global__ void ker_yolo_eval_true_box_score(int64 size, float* score_out, float* boxes_in, int64* anchors_in, int64 num_scales, int64 anchor_per_scale) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nb = idx / (num_scales * anchor_per_scale);
        int64 ns = idx / anchor_per_scale % num_scales;
        int64 na = idx % anchor_per_scale;

        int64 aidx = ((ns * anchor_per_scale) + na) * 2;

        float box_width = box_wd(boxes_in, nb);
        float box_height = box_ht(boxes_in, nb);

        float anchor_width = (float)anchors_in[aidx];
        float anchor_height = (float)anchors_in[aidx + 1];

        float inter_width = (box_width < anchor_width) ? box_width : anchor_width;
        float inter_height = (box_height < anchor_height) ? box_height : anchor_height;

        float box_area = box_width * box_height;
        float anchor_area = anchor_width * anchor_height;
        float inter_area = inter_width * inter_height;

        float union_area = __fsub_rn(__fadd_rn(box_area, anchor_area), inter_area);
        float iou = __fdiv_rn(inter_area, union_area);

        score_out[idx] = iou;
    }
}

__global__ void ker_yolo_eval_true_box_select(int64 size, float* seletced_out, float* score_in, int64 num_scales, int64 anchor_per_scale) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nom_cnt = num_scales * anchor_per_scale;
        int64 nom_idx = (int64) idx * nom_cnt;
        int64 best_idx = nom_idx;

        for (int64 n = 0; n < nom_cnt; n++) {
            //seletced_out[nom_idx + n] = score_in[nom_idx + n] > 0.4f ? 1.0f : 0.0f;
            seletced_out[nom_idx + n] = 0.0f; // 텐서 버전 대조 결과 가장 나은 경우 하나만 선택
            if (score_in[nom_idx + n] > score_in[best_idx]) best_idx = nom_idx + n;
        }

        seletced_out[best_idx] = 1.0f;
    }
}

__global__ void ker_yolo_eval_true_count_selected(int64 size, int64* selected_cnt_out, float* selected_in, int64 dat_size, int64 num_scales, int64 anchor_per_scale) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        for (int64 n = 0; n < dat_size; n++) {
            if (n / anchor_per_scale % num_scales != idx) continue;
            if (selected_in[n] <= 0.5f) continue;
            
            selected_cnt_out[idx]++;
        }
    }
}

__global__ void ker_yolo_eval_true_lookup_scale_box(int64 size, int64* box_scale_cood_out, float* selected_in, int64 nscale, int64 dat_size, int64 num_scales, int64 anchor_per_scale) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        for (int64 n = 0, sidx=0; n < dat_size; n++) {
            //int64 nb = n / (num_scales * anchor_per_scale);
            int64 ns = n / anchor_per_scale % num_scales;
            int64 na = n % anchor_per_scale;

            if (ns != nscale) continue;
            if (selected_in[n] <= 0.5f) continue;

            int64 nbox = n / (num_scales * anchor_per_scale);

            set_scale_nbox(box_scale_cood_out, sidx, nbox);
            //set_scale_nbox(box_scale_cood_out, n, nbox);
            set_scale_nanchor(box_scale_cood_out, sidx, (int64)na);

            sidx++;
        }
    }
}

__global__ void ker_yolo_eval_true_eval_box_cood(int64 size, int64* box_scale_cood_inout, float* box_info_in, int64 grid_size) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 bidx = scale_nbox(box_scale_cood_inout, idx);

        float center_x = box_cx(box_info_in, bidx);
        float center_y = box_cy(box_info_in, bidx);

        int64 nx = (int64)(center_x / (float) grid_size);
        int64 ny = (int64)(center_y / (float) grid_size);

        set_scale_nx(box_scale_cood_inout, idx, nx);
        set_scale_ny(box_scale_cood_inout, idx, ny);
    }
}

/*
__global__ void ker_yolo_set_scaled_true_box(int64 size, float* coods_out, float* sizes_out, float* confs_out, int64* class_out, float* selected_in, float* boxes_in, int64* catid_in,
                                             int64 nd, int64 nscale, int64 img_size, int64 grid_cnt, int64 num_scales, int64 anchor_per_scale) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (selected_in[idx] < 0.5f) return;

        int64 nb = idx / (num_scales * anchor_per_scale);
        int64 ns = idx / anchor_per_scale % num_scales;
        int64 na = idx % anchor_per_scale;

        if (ns != nscale) return;

        //selected_in[idx] = -999;

        //int64 bidx = nb * 4;

        float width = box_wd(boxes_in, nb);
        float height = box_ht(boxes_in, nb);

        float center_x = box_cx(boxes_in, nb) + width / 2;
        float center_y = box_cy(boxes_in, nb) + height / 2;

        float grid_size = (float)img_size / (float)grid_cnt;

        int64 nx = (int64)(center_x / grid_size);
        int64 ny = (int64)(center_y / grid_size);

        assert(nx >= 0 && nx < grid_cnt);
        assert(ny >= 0 && ny < grid_cnt);

        int64 didx = ((nd * anchor_per_scale + na) * grid_cnt + nx) * grid_cnt + ny;

        coods_out[2 * didx + 0] = center_x;
        coods_out[2 * didx + 1] = center_y;

        sizes_out[2 * didx + 0] = width;
        sizes_out[2 * didx + 1] = height;

        confs_out[didx] = 1.0f;

        class_out[didx] = catid_in[nb];
    }
}
*/

__global__ void ker_yolo_conv_fmap(int64 size, float* pred_out, float* fmap_in, int64* anchors_in, int64 img_size, int64 grid_cnt, int64 anchors_cnt, int64 class_num) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ny = idx / (grid_cnt * anchors_cnt) % grid_cnt;
        int64 nx = idx / anchors_cnt % grid_cnt;
        int64 na = idx % anchors_cnt;

        int64* anchors = anchors_in + (na * 2);

        float rescale_ratio = __fdiv_rn((float)img_size, (float)grid_cnt);

        float box_width = __fmul_rn(__expf(fmap_wd(fmap_in, idx)), (float)anchors[0]);
        float box_height = __fmul_rn(__expf(fmap_ht(fmap_in, idx)), (float)anchors[1]);

        float center_x = __fmul_rn(__fadd_rn(dev_sigmoid(fmap_cx(fmap_in, idx)), (float)nx), rescale_ratio);
        float center_y = __fmul_rn(__fadd_rn(dev_sigmoid(fmap_cy(fmap_in, idx)), (float)ny), rescale_ratio);

        set_pred_xmin(pred_out, idx, __fsub_rn(center_x, __fdiv_rn(box_width, 2.0f)));
        set_pred_ymin(pred_out, idx, __fsub_rn(center_y, __fdiv_rn(box_height, 2.0f)));
        set_pred_xmax(pred_out, idx, __fadd_rn(center_x, __fdiv_rn(box_width, 2.0f)));
        set_pred_ymax(pred_out, idx, __fadd_rn(center_y, __fdiv_rn(box_height, 2.0f)));

        set_pred_conf(pred_out, idx, dev_sigmoid(fmap_conf(fmap_in, idx)));

        for (int64 n = 0; n < class_num; n++) {
            set_pred_class(pred_out, idx, n, dev_sigmoid(fmap_class(fmap_in, idx, n)));
        }
    }
}

__global__ void ker_yolo_eval_iou(int64 size, float* iou_out, int64* boxid_out, float* fmap_in, int64* img_info_in, float* box_info_in, int64* scale_boxes_in, int64* anchors_in,
                                  int64 nanchor, int64 img_size, int64 grid_cnt, int64 box_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (grid_cnt * grid_cnt * nanchor * box_cnt);
        int64 ny = idx / (grid_cnt * nanchor * box_cnt) % grid_cnt;
        int64 nx = idx / (nanchor * box_cnt) % grid_cnt;
        int64 na = idx / box_cnt % nanchor;
        int64 nb = idx % box_cnt;

        iou_out[idx] = -1.0f;
        boxid_out[idx] = -1;

        int64 nbox = scale_nbox(scale_boxes_in, nb);
        int64 nimg = box_nimg(box_info_in, nbox);

        if (img_ndata(img_info_in, nimg) != nd) return;  // different data in minibatch

        bool matched = true;

        if (scale_nanchor(scale_boxes_in, nb) != na) matched = false; // different anchor
        else if (scale_nx(scale_boxes_in, nb) != nx) matched = false; // different center_x
        else if (scale_ny(scale_boxes_in, nb) != ny) matched = false; // different center_y

        int64 midx = idx / box_cnt;

        int64* anchors = anchors_in + (na * 2);
        float rescale_ratio = __fdiv_rn((float)img_size, (float)grid_cnt);

        float pcx = (dev_sigmoid(fmap_cx(fmap_in, midx)) + (float)nx) * rescale_ratio;
        float pcy = (dev_sigmoid(fmap_cy(fmap_in, midx)) + (float)ny) * rescale_ratio;

        float pwd = __expf(fmap_wd(fmap_in, midx)) * (float)anchors[0];
        float pht = __expf(fmap_ht(fmap_in, midx)) * (float)anchors[1];

        float pxmin = pcx - pwd / 2.0f;
        float pxmax = pcx + pwd / 2.0f;
        float pymin = pcy - pht / 2.0f;
        float pymax = pcy + pht / 2.0f;

        float tcx = box_cx(box_info_in, nbox);
        float tcy = box_cy(box_info_in, nbox);
        float twd = box_wd(box_info_in, nbox);
        float tht = box_ht(box_info_in, nbox);

        float txmin = tcx - twd / 2.0f;
        float txmax = tcx + twd / 2.0f;
        float tymin = tcy - tht / 2.0f;
        float tymax = tcy + tht / 2.0f;

        float ixmin = myfmax(pxmin, txmin);
        float ixmax = myfmin(pxmax, txmax);
        float iymin = myfmax(pymin, tymin);
        float iymax = myfmin(pymax, tymax);

        float iwd = ixmax - ixmin;
        float iht = iymax - iymin;

        float iou = 0.0f;

        if (iwd > 0 && iht > 0) {
            float pred_box_area = pwd * pht;
            float true_box_area = twd * tht;
            float intersect_area = iwd * iht;
            float union_area = pred_box_area + true_box_area - intersect_area;

            iou = intersect_area / union_area;
        }

        iou_out[idx] = iou;
        boxid_out[idx] = matched ? nbox : -1;
    }
}

__global__ void ker_yolo_select_best_iou(int64 size, int64* best_box_out, float* best_iou_out, float* iou_in, int64* boxid_in, int64 box_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 matched_idx = -1;
        float best_iou = -1.0f;

        int64 uidx = idx * box_cnt;

        for (int64 n = 0; n < box_cnt; n++) {
            if (iou_in[uidx + n] > best_iou) best_iou = iou_in[uidx + n];
            if (boxid_in[uidx + n] >= 0) matched_idx = boxid_in[uidx + n];
        }

        best_box_out[idx] = matched_idx;
        best_iou_out[idx] = best_iou;
    }
}

__global__ void ker_yolo_eval_losses(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, float* best_iou_in, int64* anchors_in,
    int64 mb_size, int64 img_size, int64 grid_cnt, int64 nanchor, int64 class_num, bool use_focal, bool smooth_onehot) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nc = idx % FMAP_SIZE;
        int64 fidx = idx / FMAP_SIZE;

        loss_out[idx] = 0;

        int64 nbox = best_box_in[fidx];

        float grid_size = (float)img_size / (float)grid_cnt;
        float felem = fmap_in[idx];
        //float felem = fmap_elem(fmap_in, fidx, nc);

        float mixed = (nbox >= 0) ? box_mixed(box_info_in, nbox) : 1.0f;
        float coef = mixed / (float)mb_size;

        if (nc < 4) {
            if (nbox < 0) return;

            int64 ny = fidx / (grid_cnt * nanchor) % grid_cnt;
            int64 nx = fidx / (nanchor) % grid_cnt;
            int64 na = fidx % nanchor;

            float twidth = box_wd(box_info_in, nbox);
            float theight = box_ht(box_info_in, nbox);

            float loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (float)img_size), __fdiv_rn(theight, (float)img_size)));

            if (nc < 2) {
                float pxy = dev_sigmoid(felem);
                float tcenter = box_rect(box_info_in, nbox, (int64) nc);
                float txy = tcenter / grid_size - (float)((nc == 0) ? nx : ny);

                float xy_diff = __fsub_rn(pxy, txy);
                float dist_sq = __fmul_rn(xy_diff, xy_diff);

                loss_out[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, coef));
            }
            else {
                float anchor_size = (float)anchors_in[2 * na + (nc -2)];

                float pwh = felem;
                float tsz = box_rect(box_info_in, nbox, (int64) nc);
                float twh = __logf(clip(__fdiv_rn(tsz, anchor_size), 1e-9f, 1e9f));

                float wh_diff = __fsub_rn(pwh, twh);
                float dist_sq = __fmul_rn(wh_diff, wh_diff);

                loss_out[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, coef));
            }
        }
        else if (nc == 4) {
            bool object_mask = nbox >= 0;
            bool ignore_mask = best_iou_in[fidx] < 0.5f;
            
            if (!object_mask && !ignore_mask) return;

            float loss_conf = dev_sigmoid_cross_entropy_with_logits(felem, object_mask);

            if (use_focal) {
                float focal_diff = dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask);
                float focal_mask = __fmul_rn(focal_diff, focal_diff);

                loss_conf = __fmul_rn(loss_conf, focal_mask);
            }

            loss_out[idx] = __fmul_rn(loss_conf, coef);
        }
        else {
            if (nbox < 0) return;

            int64 pclass = (int64) nc - 5;
            int64 tclass = box_catid(box_info_in, nbox);

            float z = (pclass == tclass) ? 1.0f : 0.0f;

            if (smooth_onehot) {
                float delta = 0.01f;
                z += delta / (float)class_num;
                if (pclass == tclass) z -= delta;
            }

            float entropy = dev_sigmoid_cross_entropy_with_logits(felem, z);

            loss_out[idx] = __fmul_rn(entropy, coef);
        }
    }
}

__global__ void ker_yolo_eval_grads(int64 size, float* grad_out, float* fmap_in, float* box_info_in, int64* best_box_in, float* best_iou_in, int64* anchors_in,
    int64 mb_size, int64 img_size, int64 grid_cnt, int64 nanchor, int64 class_num, bool use_focal, bool smooth_onehot) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nc = idx % FMAP_SIZE;
        int64 fidx = idx / FMAP_SIZE;

        grad_out[idx] = 0;

        /*
        if (idx == 0) {
            grad_out[0] = dev_sigmoid(-16.5f);
            grad_out[1] = dev_sigmoid_cross_entropy_with_logits(-16.5f, 0.0f);
            grad_out[2] = dev_sigmoid_cross_entropy_with_logits_derv(-16.5f, 0.0f);
            return;
        }
        else if (idx < 4) return;
        */

        int64 nbox = best_box_in[fidx];

        float grid_size = (float)img_size / (float)grid_cnt;
        float felem = fmap_in[idx];

        float mixed = (nbox >= 0) ? box_mixed(box_info_in, nbox) : 1.0f;
        float coef = mixed / (float)mb_size;

        if (nc < 4) {
            if (nbox < 0) return;

            int64 ny = fidx / (grid_cnt * nanchor) % grid_cnt;
            int64 nx = fidx / (nanchor) % grid_cnt;
            int64 na = fidx % nanchor;

            float twidth = box_wd(box_info_in, nbox);
            float theight = box_ht(box_info_in, nbox);

            float loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (float)img_size), __fdiv_rn(theight, (float)img_size)));

            if (nc < 2) {
                float pxy = dev_sigmoid(felem);
                float tcenter = box_rect(box_info_in, nbox, (int64) nc);
                float txy = tcenter / grid_size - (float)((nc == 0) ? nx : ny);

                float xy_diff = __fsub_rn(pxy, txy);
                float g_pxy = __fmul_rn(__fmul_rn(2.0f, xy_diff), __fmul_rn(loss_scale, coef));

                grad_out[idx] = __fmul_rn(g_pxy, dev_sigmoid_derv(felem, pxy));
            }
            else {
                float anchor_size = (float)anchors_in[2 * na + (nc -2)];

                float pwh = felem;
                float tsz = box_rect(box_info_in, nbox, (int64) nc);
                float twh = __logf(clip(__fdiv_rn(tsz, anchor_size), 1e-9f, 1e9f));

                float wh_diff = __fsub_rn(pwh, twh);

                grad_out[idx] = __fmul_rn(__fmul_rn(2.0f, wh_diff), __fmul_rn(loss_scale, coef));
            }
        }
        else if (nc == 4) {
            bool object_mask = nbox >= 0;
            bool ignore_mask = best_iou_in[fidx] < 0.5f;

            if (!object_mask && !ignore_mask) return;

            float g_loss_conf = coef; // backprop for 'loss_out = __fmul_rn(loss_out, coef)'
            float g_focal_x = 0;

            if (use_focal) {
                float focal_diff = dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask);
                float focal_mask = __fmul_rn(focal_diff, focal_diff);

                float loss_conf = dev_sigmoid_cross_entropy_with_logits(felem, object_mask);

                float g_focal_mask = __fmul_rn(g_loss_conf, loss_conf); // backprop for 'loss_conf = __fmul_rn(loss_conf, focal_mask)'
                float g_focal_diff = __fmul_rn(g_focal_mask, __fmul_rn(2.0f, focal_diff)); // backprop for 'focal_mask = __fmul_rn(focal_diff, focal_diff)'

                g_focal_x = __fmul_rn(g_focal_diff, dev_sigmoid_derv(felem)); // backprop for 'focal_diff = dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask)'
                g_loss_conf = __fmul_rn(g_loss_conf, focal_mask); // backprop for 'loss_conf = __fmul_rn(loss_conf, focal_mask)'
            }

            float g_loss_x = __fmul_rn(g_loss_conf, dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask)); // backprop for 'loss_conf = dev_sigmoid_cross_entropy_with_logits(felem, object_mask)'

            grad_out[idx] = __fadd_rn(g_loss_x, g_focal_x);
        }
        else {
            if (nbox < 0) return;

            int64 pclass = (int64) nc - 5;
            int64 tclass = box_catid(box_info_in, nbox);

            float z = (pclass == tclass) ? 1.0f : 0.0f;

            if (smooth_onehot) {
                float delta = 0.01f;
                z += delta / (float)class_num;
                if (pclass == tclass) z -= delta;
            }

            float sig_derv = dev_sigmoid_cross_entropy_with_logits_derv(felem, z);

            grad_out[idx] = __fmul_rn(sig_derv, coef);
        }
    }
}

/*
__global__ void ker_yolo_loss_cood(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, int64 img_size, int64 nanchor, int64 grid_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 ny = idx / (grid_cnt * nanchor) % grid_cnt;
        int64 nx = idx / nanchor % grid_cnt;

        loss_out[idx] = 0;

        int64 nbox = best_box_in[idx];

        if (nbox < 0) return;

        float pcx = dev_sigmoid(fmap_cx(fmap_in, idx));
        float pcy = dev_sigmoid(fmap_cy(fmap_in, idx));

        float grid_size = (float)img_size / (float)grid_cnt;

        float tcenter_x = box_cx(box_info_in, nbox);
        float tcenter_y = box_cy(box_info_in, nbox);

        float tcx = tcenter_x / grid_size - (float)nx;
        float tcy = tcenter_y / grid_size - (float)ny;

        float x_diff = __fsub_rn(pcx, tcx);
        float y_diff = __fsub_rn(pcy, tcy);

        float dist_sq = __fadd_rn(__fmul_rn(x_diff, x_diff), __fmul_rn(y_diff, y_diff));

        float twidth = box_wd(box_info_in, nbox);
        float theight = box_ht(box_info_in, nbox);

        float loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (float)img_size), __fdiv_rn(theight, (float)img_size)));
        float mixed = box_mixed(box_info_in, nbox);
         
        loss_out[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, mixed));
    }
}

__global__ void ker_yolo_loss_size(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, int64* anchors_in, int64 img_size, int64 nanchor, int64 grid_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 na = idx % nanchor;
        int64 nbox = best_box_in[idx];

        loss_out[idx] = 0;

        if (nbox < 0) return;

        float anchor_width = (float) anchors_in[2 * na];
        float anchor_height = (float)anchors_in[2 * na + 1];

        float pwd = fmap_wd(fmap_in, idx);
        float pht = fmap_ht(fmap_in, idx);

        float twidth = box_wd(box_info_in, nbox);
        float theight = box_ht(box_info_in, nbox);

        float twd = __logf(clip(__fdiv_rn(twidth, anchor_width), 1e-9f, 1e9f));
        float tht = __logf(clip(__fdiv_rn(theight, anchor_height), 1e-9f, 1e9f));

        float w_diff = __fsub_rn(pwd, twd);
        float h_diff = __fsub_rn(pht, tht);

        float dist_sq = __fadd_rn(__fmul_rn(w_diff, w_diff), __fmul_rn(h_diff, h_diff));
        float loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (float)img_size), __fdiv_rn(theight, (float)img_size)));
        float mixed = box_mixed(box_info_in, nbox); 

        loss_out[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, mixed));
    }
}

__global__ void ker_yolo_loss_conf(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, float* best_iou_in, bool use_focal) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        loss_out[idx] = 0;

        int64 nbox = best_box_in[idx];

        bool object_mask = nbox >= 0;
        bool ignore_mask = best_iou_in[idx] < 0.5f;

        bool conf_pos = object_mask;
        bool conf_neg = !object_mask && ignore_mask;

        float x = fmap_conf(fmap_in, idx);

        float loss_pos = conf_pos ? dev_sigmoid_cross_entropy_with_logits(x, object_mask) : 0;
        float loss_neg = conf_neg ? dev_sigmoid_cross_entropy_with_logits(x, object_mask) : 0;
        
        float conf_loss = loss_pos +loss_neg;

        if (use_focal) {
            float sigmoid = dev_sigmoid(x);

            float focal_diff = __fsub_rn(object_mask, sigmoid);
            float focal_mask = __fmul_rn(focal_diff, focal_diff);

            conf_loss = __fmul_rn(conf_loss, focal_mask);
        }

        float mixed = (nbox >= 0) ? box_mixed(box_info_in, nbox) : 1.0f;

        loss_out[idx] = __fmul_rn(conf_loss, mixed);
    }
}

__global__ void ker_yolo_loss_class(int64 size, float* loss_out, float* fmap_in, float* box_info_in, int64* best_box_in, int64 class_num, bool smooth_onehot) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        loss_out[idx] = 0;

        int64 nbox = best_box_in[idx];

        if (nbox < 0) return;

        int64 target_id = box_catid(box_info_in, nbox);

        float loss_class = 0;

        for (int64 n = 0; n < class_num; n++) {
            float x = fmap_class(fmap_in, idx, n);
            float z = (n == target_id) ? 1 : 0;

            if (smooth_onehot) {
                float delta = 0.01f;
                z += delta / (float) class_num;
                if (n == target_id) z -= delta;
            }

            float entropy = dev_sigmoid_cross_entropy_with_logits(x, z);

            loss_class = __fadd_rn(loss_class, entropy);
        }

        float mixed = box_mixed(box_info_in, nbox);

        loss_out[idx] = __fmul_rn(loss_class, mixed);
    }
}
*/

/*
__global__ void ker_yolo_coods_derv(int64 size, float* gcoods_out, float* gmixed_inout, float* pcood_in, float* pmixed_in, float* box_rect_in, int64* best_box_in, int64 img_size, int64 nanchor, int64 grid_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 box_id = best_box_in[idx];
        if (box_id < 0) return;

        int64 pidx = idx * 2;
        int64 tidx = box_id * 4;

        float pcenter_x = pcood_in[pidx];
        float pcenter_y = pcood_in[pidx + 1];

        float tcenter_x = box_rect_in[tidx];
        float tcenter_y = box_rect_in[tidx + 1];
        float twidth = box_rect_in[tidx + 2];
        float theight = box_rect_in[tidx + 3];

        float grid_size = __fdiv_rn((float)img_size, (float)grid_cnt);

        float x_diff = __fdiv_rn(__fsub_rn(pcenter_x, tcenter_x), grid_size);
        float y_diff = __fdiv_rn(__fsub_rn(pcenter_y, tcenter_y), grid_size);

        float dist_sq = __fadd_rn(__fmul_rn(x_diff, x_diff), __fmul_rn(y_diff, y_diff));

        float loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (float)img_size), __fdiv_rn(theight, (float)img_size)));
        float mixed = pmixed_in[idx];

        int64 xidx = pidx;
        int64 yidx = pidx + 1;

        gcoods_out[xidx] = 2 * x_diff * loss_scale * mixed / grid_size;
        gcoods_out[yidx] = 2 * y_diff * loss_scale * mixed / grid_size;

        gmixed_inout[idx] += dist_sq * loss_scale;
    }
}

__global__ void ker_yolo_sizes_derv(int64 size, float* gsizes_out, float* gmixed_inout, float* psize_in, float* pmixed_in, float* box_rect_in, int64* best_box_in, int64* anchors_in, int64 img_size, int64 nanchor, int64 grid_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 na = idx / (grid_cnt * grid_cnt) % nanchor;
        int64 box_id = best_box_in[idx];

        if (box_id < 0) return;

        int64 pidx = idx * 2;
        int64 tidx = box_id * 4;

        float anchor_width = (float)anchors_in[2 * na];
        float anchor_height = (float)anchors_in[2 * na + 1];

        float pwidth = psize_in[pidx];
        float pheight = psize_in[pidx + 1];

        float twidth = box_rect_in[tidx + 2];
        float theight = box_rect_in[tidx + 3];

        float log_pw = __logf(clip(__fdiv_rn(pwidth, anchor_width), 1e-9f, 1e9f));
        float log_ph = __logf(clip(__fdiv_rn(pheight, anchor_height), 1e-9f, 1e9f));

        float log_tw = __logf(clip(__fdiv_rn(twidth, anchor_width), 1e-9f, 1e9f));
        float log_th = __logf(clip(__fdiv_rn(theight, anchor_height), 1e-9f, 1e9f));

        float w_diff = __fsub_rn(log_pw, log_tw);
        float h_diff = __fsub_rn(log_ph, log_th);

        float dist_sq = __fadd_rn(__fmul_rn(w_diff, w_diff), __fmul_rn(h_diff, h_diff));
        float loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (float)img_size), __fdiv_rn(theight, (float)img_size)));
        float mixed = pmixed_in[idx];

        //loss_out[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, mixed));

        int64 widx = pidx;
        int64 hidx = pidx + 1;

        gsizes_out[widx] = 2 * w_diff * loss_scale * mixed / pwidth;    // anchor  크기는 미분 과정에서 상쇄되어 서라짐, 상수 나눗셈은 로그에서 상수 뺄셈으로 변환됨에 유이
        gsizes_out[hidx] = 2 * h_diff * loss_scale * mixed / pheight;   // anchor  크기는 미분 과정에서 상쇄되어 서라짐, 상수 나눗셈은 로그에서 상수 뺄셈으로 변환됨에 유이

        gmixed_inout[idx] += dist_sq * loss_scale;
    }
}

__global__ void ker_yolo_confs_derv(int64 size, float* gconfs_out, float* gmixed_inout, float* pconfs_in, float* pmixed_in, int64* best_box_in, float* best_iou_in, bool use_focal) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 box_id = best_box_in[idx];

        bool object_mask = (box_id >= 0);
        bool ignore_mask = (best_iou_in[idx] < 0.5f);

        bool conf_pos = object_mask;
        bool conf_neg = ! object_mask && ignore_mask;

        float x = pconfs_in[idx];
        float z = object_mask;

        float entropy = dev_sigmoid_cross_entropy_with_logits(pconfs_in[idx], object_mask);

        float loss_pos = conf_pos ? entropy : 0;
        float loss_neg = conf_neg ? entropy : 0;

        float conf_loss = loss_pos + loss_neg;

        float mixed = pmixed_in[idx];

        float sigmoid = dev_sigmoid(x);

        float g_conf_loss = mixed;
        float g_x = 0;

        if (use_focal) {
            float focal_diff = __fsub_rn(object_mask, sigmoid);
            float focal_mask = __fmul_rn(focal_diff, focal_diff);

            float alpha = 1.0f;

            float y = sigmoid;
            float sig_derv = dev_sigmoid_derv(x, y); // __fmul_rn(y, __fsub_rn(1.0f, y));

            float g_focal_mask = conf_loss;
            float g_focal_diff = 2 * alpha * focal_diff * g_focal_mask;
            float g_sigmoid = - g_focal_diff;
            float g_x_focal = sig_derv * g_sigmoid;

            g_conf_loss *= focal_mask;

            g_x += g_x_focal;
            //g_x += 2 * alpha * focal_diff * conf_loss * sig_derv * mixed;

            conf_loss *= focal_mask;
        }

        float g_entropy = __fsub_rn(sigmoid, z) * g_conf_loss;

        g_x += conf_pos ? g_entropy : 0;
        g_x += conf_neg ? g_entropy : 0;

        gconfs_out[idx] = g_x;
        
        gmixed_inout[idx] += conf_loss;
    }
}

__global__ void ker_yolo_probs_derv(int64 size, float* gprobs_out, float* gmbufs_out, float* pprobs_in, float* pmixed_in, int64* best_box_in, int64* box_info_in, int64 class_num, bool smooth_onehot) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 midx = idx / class_num;
        int64 nclass = idx % class_num;

        int64 box_id = best_box_in[midx];
        if (box_id < 0) return;

        int64 target_id = box_info_in[box_id * 5 + 4];

        float x = pprobs_in[idx];
        float z = (nclass == target_id) ? 1 : 0;
        
        if (smooth_onehot) {
            float delta = 0.01f;
            z += delta / (float) class_num;
            if (nclass == target_id) z -= delta;
        }

        float sigmoid = dev_sigmoid(x);
        float sig_derv = dev_sigmoid_derv(x, sigmoid);
        float entropy = dev_sigmoid_cross_entropy_with_logits(x, z);

        float loss_class = entropy;
        float g_loss_class = sig_derv;

        float mixed = pmixed_in[midx];

        gprobs_out[idx] = g_loss_class * mixed;
        
        if (gmbufs_out) gmbufs_out[idx] = loss_class;
    }
}


__global__ void ker_yolo_acc_gmbufs(int64 size, float* gmixed_inout, float* gmbufs_in, int64 class_num) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 pidx = idx * class_num;

        for (int64 n = 0; n < class_num; n++) {
            gmixed_inout[idx] += gmbufs_in[pidx + n];
        }
    }
}

__global__ void ker_yolo_fmap_derv(int64 size, float* grad_fmap_out, float* gcoods_in, float* gsizes_in, float* gconfs_in, float* gprobs_in, float* gmixed_in, float* coods_in, float* sizes_in, float* confs_in, float* probs_in, float* mixed_in,
    int64* anchors_in, int64 img_size, int64 grid_cnt, int64 anchors_cnt, int64 class_num, bool use_mixed) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nvec = class_num + (use_mixed ? 6 : 5);

        int64 nd = idx / (grid_cnt * grid_cnt * anchors_cnt * nvec);
        int64 nx = idx / (grid_cnt * anchors_cnt * nvec) % grid_cnt;
        int64 ny = idx / (anchors_cnt * nvec) % grid_cnt;
        int64 na = idx / nvec % anchors_cnt;
        int64 nc = idx % nvec;

        //int64 fidx = (((nd * grid_cnt + nh) * grid_cnt + nw) * anchors_cnt + na) * nvec + nc;
        //float x = fmap_in[fidx];

        // box centers
        if (nc < 2) {
            int64 cidx = ((((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny) * 2) + nc;

            float y = coods_in[cidx];;
            float g_y = gcoods_in[cidx];

            // rescale backprop
            float rescale_ratio = __fdiv_rn((float)img_size, (float)grid_cnt);
            float g_offset_term = __fmul_rn(g_y, rescale_ratio);

            // add offset backprop: do nothing

            // sigmoid backprop
            float g_sigmoid = dev_sigmoid_derv(0, y); // x is don't care in dev_sigmoid_derv //__fmul_rn(y, __fsub_rn(1.0f, y));
            float g_x = __fmul_rn(g_offset_term, g_sigmoid);

            grad_fmap_out[idx] = g_x;
        }
        // box sizes
        else if (nc < 4) {
            int64 cidx = ((((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny) * 2) + (nc - 2);

            float y = sizes_in[cidx];
            float g_y = gsizes_in[cidx];

            float g_x = __fmul_rn(g_y, y);
            grad_fmap_out[idx] = g_x;
        }
        // conf
        else if (nc < 5) {
            int64 cidx = ((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny;
            grad_fmap_out[idx] = gconfs_in[cidx];
        }
        // conf
        else if (nc < class_num + 5) {
            int64 pidx = (((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny) * class_num + (nc - 5);
            grad_fmap_out[idx] = gprobs_in[pidx];
        }
        // mixed: 대안 1: 사용 않음(1), 대안 2: sigmoid(x), 대안3: x => 일단 대안2로 테스트, 대안3은 비용 함수에 음수 난무 초래
        else {
            if (use_mixed) {
                int64 cidx = ((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny;

                float y = mixed_in[cidx];
                float g_y = gmixed_in[cidx];

                float g_sigmoid = __fmul_rn(y, __fsub_rn(1.0f, y));
                float g_x = __fmul_rn(g_y, g_sigmoid);

                grad_fmap_out[idx] = g_x;
            }
            else {
                grad_fmap_out[idx] = 0;
            }
        }
    }
}
*/

__global__ void ker_yolo_select_pred_boxes(int64 size, unsigned char* flag_out, int64* ndata_out, float* conf_rects_out, float* fmap_in, int64* anchors_in, float pred_conf_thr, int64 out_base, int64 img_size, int64 grid_cnt, int64 anchor_per_scale, int64 class_num) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (anchor_per_scale * grid_cnt * grid_cnt);
        int64 na = idx / (grid_cnt * grid_cnt) % anchor_per_scale;
        int64 nx = idx / (grid_cnt) % grid_cnt;
        int64 ny = idx % grid_cnt;

        int64 flag_bytes = (class_num + 7) / 8;

        int64 nidx = idx + out_base;
        int64 fidx = nidx * flag_bytes;
        int64 ridx = nidx * 5;

        int64 fmap_idx = (((nd * grid_cnt + nx) * grid_cnt + ny) * anchor_per_scale + na);

        float sigmoid_conf = dev_sigmoid(pred_conf(fmap_in, fmap_idx));
        if (sigmoid_conf < pred_conf_thr) return;

        int64 cat_id = 0;
        float max_term = fmap_in[fmap_idx + 5];

        for (int64 n = 1; n < class_num; n++) {
            float x = fmap_in[fmap_idx + n + 5]; // category logits
            if (x > max_term) {
                cat_id = n;
                max_term = x;
            }
        }

        conf_rects_out[ridx] = sigmoid_conf;
        flag_out[fidx+(cat_id)/8] = (unsigned char) (1 << (cat_id % 8));
        ndata_out[nidx] = (int64) nd;

        // box centers
        for (int64 n = 0; n < 2; n++) {
            float x = fmap_in[fmap_idx + n]; // center_x or center_y
                
            // sigmoid
            float term1 = (x > 0) ? 1 : __expf(x);
            float term2 = __fadd_rn(1.0f, __expf((x > 0) ? -x : x));
            float sigmoid_term = __fdiv_rn(term1, term2);

            // add offset
            float offset = (float)((n == 0) ? nx : ny);
            float offset_term = __fadd_rn(sigmoid_term, offset);

            // rescale
            float rescale_ratio = __fdiv_rn((float)img_size, (float)grid_cnt);
            float rescaled_term = __fmul_rn(offset_term, rescale_ratio);

            conf_rects_out[ridx + n + 1] = rescaled_term;
        }

        // box sizes
        for (int64 n = 0; n < 2; n++) {
            float x = fmap_in[fmap_idx + n + 2]; // width or height
            float exp_term = __expf(x);
            float anchor = (float) anchors_in[na * 2 + n];
            float rescaled_term = __fmul_rn(exp_term, anchor);

            conf_rects_out[ridx + n + 3] = rescaled_term;
        }
    }
}

/*
__global__ void ker_yolo_eval_box_pair_ious(int64 size, int64* pinfo_out, float* ious_out, int64* idxs_in, int64* cats_in, float* rects_in, int64* box_info_in, float* box_rect_in, int64 tbox_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 npred = idx / tbox_cnt;
        int64 ntrue = idx % tbox_cnt;

        int64 pfidx = idxs_in[npred];
        int64 pnidx = pfidx * 2;
        int64 pridx = pfidx * 4;

        int64 tnidx = ntrue * 5;
        int64 tridx = ntrue * 4;

        int64 pred_ndata = cats_in[pnidx + 0];
        int64 pred_catid = cats_in[pnidx + 1];

        int64 true_ndata = box_info_in[tnidx + 0];
        int64 true_catid = box_info_in[tnidx + 4];

        int64 pidx = idx * 2;

        pinfo_out[pidx] = 0;  // means invalid pair
        pinfo_out[pidx + 1] = pred_catid;

        if (pred_ndata != true_ndata) return;
        if (pred_catid != true_catid) return;

        pinfo_out[pidx] = 1;  // means valid pair

        float pcenter_x = rects_in[pridx];
        float pcenter_y = rects_in[pridx + 1];
        float pwidth = rects_in[pridx + 2];
        float pheight = rects_in[pridx + 3];

        float pleft = pcenter_x - pwidth / 2.0f;
        float pright = pcenter_x + pwidth / 2.0f;
        float ptop = pcenter_y - pheight / 2.0f;
        float pbottom = pcenter_y + pheight / 2.0f;

        float tcenter_x = box_rect_in[tridx];
        float tcenter_y = box_rect_in[tridx + 1];
        float twidth = box_rect_in[tridx + 2];
        float theight = box_rect_in[tridx + 3];

        float tleft = tcenter_x - twidth / 2.0f;
        float tright = tcenter_x + twidth / 2.0f;
        float ttop = tcenter_y - theight / 2.0f;
        float tbottom = tcenter_y + theight / 2.0f;

        if (pleft > tright) return;
        if (tleft > pright) return;
        if (ptop > tbottom) return;
        if (ttop > pbottom) return;

        float ileft = (pleft > tleft) ? pleft : tleft;
        float iright = (pright < tright) ? pright : tright;
        float itop = (ptop > ttop) ? ptop : ttop;
        float ibottom = (pbottom < tbottom) ? pbottom : tbottom;

        float iwidth = iright - ileft;
        float iheight = ibottom - itop;

        float pred_box_area = pwidth * pheight;
        float true_box_area = twidth * theight;
        float intersect_area = iwidth * iheight;
        float union_area = pred_box_area + true_box_area - intersect_area;

        float iou = intersect_area / union_area;
        
        ious_out[idx] = iou;
    }
}

__global__ void ker_yolo_count_true_boxes(int64 size, int64* tbox_cnt_out, int64* box_info_in, int64 tbox_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nclass = idx;

        int64 found = 0;

        for (int64 ntrue = 0; ntrue < tbox_cnt; ntrue++) {
            int64 tnidx = ntrue * 5;
            int64 true_catid = box_info_in[tnidx + 4];
            if (true_catid == nclass) found++;
        }

        tbox_cnt_out[idx] = found;
    }
}

__global__ void ker_yolo_count_pred_boxes(int64 size, int64* pbox_cnt_out, int64* pair_info_in, int64 pbox_cnt, int64 tbox_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nclass = idx;

        int64 found = 0;

        for (int64 npred = 0; npred < pbox_cnt; npred++) {
            int64 pnidx = npred * tbox_cnt;
            int64 pred_catid = pair_info_in[pnidx * 2 + 1];
            if (pred_catid == nclass) found++;
        }

        pbox_cnt_out[idx] = found;
    }
}

__global__ void ker_yolo_count_matched_box_pairs(int64 size, int64* match_cnt_out, int64* pair_info_in, float* ious_in, int64 iou_thr_cnt, float iou_thr_from, float iou_thr_step, int64 pbox_cnt, int64 tbox_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nclass = idx / iou_thr_cnt;
        int64 niouthr = idx % iou_thr_cnt;

        float iou_thr = iou_thr_from + iou_thr_step * (float) niouthr;

        int64 found = 0;

        for (int64 npred = 0; npred < pbox_cnt; npred++) {
            int64 pnidx = npred * tbox_cnt;
            
            int64 pred_catid = pair_info_in[pnidx * 2+ 1];
            if (pred_catid != nclass) continue;

            for (int64 ntrue = 0; ntrue < tbox_cnt; ntrue++, pnidx++) {
                int64 is_valid_pair = pair_info_in[pnidx * 2];
                if (!is_valid_pair) continue;
                float iou = ious_in[pnidx];
                if (iou >= iou_thr) {
                    found++;
                    break;
                }
            }
        }

        match_cnt_out[idx] = found;
    }
}

__global__ void ker_yolo_eval_prec_recall(int64 size, float* precision_out, float* recall_out, int64* tbox_cnt_in, int64* pbox_cnt_in, int64* match_cnt_in, int64 iou_thr_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nclass = idx / iou_thr_cnt;
        int64 niouthr = idx % iou_thr_cnt;

        int64 tbox_cnt = tbox_cnt_in[idx];
        int64 pbox_cnt = pbox_cnt_in[idx];

        int64 match_cnt = match_cnt_in[idx];

        float precision = 0;
        float recall = 0;

        if (match_cnt > 0) {
            precision = (float) match_cnt / (float)pbox_cnt;
            recall = (float)match_cnt / (float)tbox_cnt;
        }

        precision_out[idx] = precision;
        recall_out[idx] = recall;
    }
}
*/

__global__ void ker_yolo_eval_predict_score(int64 size, float* score_out, float* pred_in, int64 class_num, float score_thresh) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / class_num;
        int64 nc = idx % class_num;

        float conf_prob = pred_conf(pred_in, nr);
        float class_prof = pred_class(pred_in, nr, nc);

        float score = __fmul_rn(conf_prob, class_prof);
        score_out[idx] = (score >= score_thresh) ? score : 0;
    }
}

__global__ void ker_yolo_get_boxes(int64 size, float* boxes_out, float* old_boxes_in, int64* idxs_in, float* score_in, float* pred_in, int64 old_count, int64 grid_cnt, int64 anchors_cnt, int64 class_num) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        //int64 bidx = idx * 8;

        if (idx < old_count) {
            memcpy(boxes_out + idx * RES_BOX_SIZE, old_boxes_in + idx * RES_BOX_SIZE, RES_BOX_SIZE * sizeof(float));
        }
        else {
            int64 sidx = idxs_in[idx - old_count];

            int64 nd = sidx / (grid_cnt * grid_cnt * anchors_cnt * class_num);
            int64 pidx = sidx / class_num;

            set_res_xmin(boxes_out, idx, pred_xmin(pred_in, pidx));
            set_res_ymin(boxes_out, idx, pred_ymin(pred_in, pidx));
            set_res_xmax(boxes_out, idx, pred_xmax(pred_in, pidx));
            set_res_ymax(boxes_out, idx, pred_ymax(pred_in, pidx));

            set_res_score(boxes_out, idx, score_in[sidx]);
            set_res_class(boxes_out, idx, (int64) sidx % class_num);
            set_res_ndata(boxes_out, idx, (int64)nd);
            set_res_flag(boxes_out, idx, 0);
        }
    }
}

__global__ void ker_yolo_non_max_suppression(int64 size, float* boxes_inout, int64 box_cnt, int64 max_boxes, float iou_thr) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {   // idx: class index
        //int64 step = 0;

        while (true) {
            //step++;

            int64 max_id = -1;
            float max_score = 0;
            for (int64 n = 0; n < box_cnt; n++) {
                if (res_flag(boxes_inout, n) != 0) continue;
                if (res_class(boxes_inout, n) != (float)idx) continue;
                if (res_score(boxes_inout, n) > max_score) {
                    max_id = n;
                    max_score = res_score(boxes_inout, n);
                }
            }
            if (max_id < 0) break;
            
            set_res_flag(boxes_inout, max_id, 1);

            float width = res_xmax(boxes_inout, max_id) - res_xmin(boxes_inout, max_id);
            float height = res_ymax(boxes_inout, max_id) - res_ymin(boxes_inout, max_id);

            float box1_area = width * height;

            for (int64 n = 0; n < box_cnt; n++) {
                if (res_flag(boxes_inout, n) != 0) continue;
                if (res_class(boxes_inout, n) != (float)idx) continue;
                
                //float box2_area = boxes_inout[n * 7 + 2] * boxes_inout[n * 7 + 3];

                float left   = myfmax(res_xmin(boxes_inout, max_id), res_xmin(boxes_inout, n));
                float right  = myfmin(res_xmax(boxes_inout, max_id), res_xmax(boxes_inout, n)); 
                float top    = myfmax(res_ymin(boxes_inout, max_id), res_ymin(boxes_inout, n));
                float bottom = myfmin(res_ymax(boxes_inout, max_id), res_ymax(boxes_inout, n));

                float width = (left < right) ? (right - left) : 0;
                float height = (top < bottom) ? (bottom - top) : 0;
                
                float inter_area = width * height;
                //float union_area = box1_area + box2_area - inter_area;

                //float iou = inter_area / union_area;
                float iou = inter_area / box1_area;

                if (iou >= iou_thr) {
                    set_res_flag(boxes_inout, n, 2);
                }
            }
        }
    }
}

__global__ void ker_real_to_complex(int64 size, float* c_out, float* f_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        c_out[idx] = (idx % 2 == 0) ? f_in[idx / 2] : 0;
    }
}

__global__ void ker_short_to_complex(int64 size, float* c_out, short* f_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        c_out[idx] = (idx % 2 == 0) ? (float) f_in[idx / 2] : 0;
    }
}

__global__ void ker_wave_slices_to_complex(int64 size, float* c_out, float* w_in, int64 step_width, int64 step_cnt, int64 fft_width, int64 fetch_width) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (step_cnt * fft_width * 2);
        int64 ns = idx / (fft_width * 2) % step_cnt;
        int64 nx = (idx / 2) % fft_width;
        bool is_real = (idx % 2) == 0;

        if (is_real) {
            int64 wpos = nd * fetch_width + ns * step_width + nx;
            c_out[idx] = w_in[wpos];
        }
        else {
            c_out[idx] = 0;
        }
    }
}

__global__ void ker_complex_to_abs(int64 size, float* a_out, float* c_in) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float real = c_in[idx * 2];
        float image = c_in[idx * 2 + 1];
        a_out[idx] = __fsqrt_rn(real * real + image * image);
    }
}

__global__ void ker_complex_to_abs_mean(int64 size, float* a_out, float* c_in, int64 data_num, int64 freq_cnt) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / freq_cnt;
        int64 nx = idx % freq_cnt;

        int64 win_size = data_num / (2 * freq_cnt);
        int64 win_start = data_num / 2 - win_size * (freq_cnt - nx);

        float sum = 0;

        for (int64 n = 0; n < win_size; n++) {
            int64 cidx = nd * data_num + win_start + n;

            float real = c_in[cidx * 2];
            float image = c_in[cidx * 2 + 1];
            
            sum += __fsqrt_rn(real * real + image * image);
        }

        a_out[idx] = sum / win_size;
    }
}

__global__ void ker_fft_step(int64 size, float* dst_out, float* src_in, int64 data_num, int64 step) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (data_num * 2);
        int64 nx = (idx / 2) % data_num;
        bool is_real = (idx % 2) == 0;

        int64 stride = data_num / step;

        int64 pos1 = (nx / stride * (2 * stride)) % data_num + nx % stride;
        int64 pos2 = pos1 + stride;

        float x1_real = src_in[(nd * data_num + pos1) * 2];
        float x1_image = src_in[(nd * data_num + pos1) * 2 + 1];

        float x2_real = src_in[(nd * data_num + pos2) * 2];
        float x2_image = src_in[(nd * data_num + pos2) * 2 + 1];

        float theta = -2 * CUDART_PI_F * (nx / stride * stride) / data_num;

        float t_real = __cosf(theta);
        float t_image = __sinf(theta);

        if (is_real)
            dst_out[idx] = x1_real + x2_real * t_real - x2_image * t_image;
        else
            dst_out[idx] = x1_image + x2_real * t_image + x2_image * t_real;
    }
}

__global__ void ker_fft_step_split(int64 size, float* dst_out, float* src_in, int64 data_num, int64 step, int64 nd_base) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nd = idx / (data_num * 2) + nd_base;
        int64 nx = (idx / 2) % data_num;
        bool is_real = (idx % 2) == 0;

        int64 stride = data_num / step;

        int64 pos1 = (nx / stride * (2 * stride)) % data_num + nx % stride;
        int64 pos2 = pos1 + stride;

        float x1_real = src_in[(nd * data_num + pos1) * 2];
        float x1_image = src_in[(nd * data_num + pos1) * 2 + 1];

        float x2_real = src_in[(nd * data_num + pos2) * 2];
        float x2_image = src_in[(nd * data_num + pos2) * 2 + 1];

        float theta = -2 * CUDART_PI_F * (nx / stride * stride) / data_num;

        float t_real = __cosf(theta);
        float t_image = __sinf(theta);

        int64 didx = (nd * data_num + nx) * 2;

        if (is_real)
            dst_out[didx] = x1_real + x2_real * t_real - x2_image * t_image;
        else
            dst_out[didx+1] = x1_image + x2_real * t_image + x2_image * t_real;
    }
}

__global__ void ker_eveal_hash_match_point(int64 size, float* p_out, float* c1_in, float* c2_in, int64 nrow, int64 ncol, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / ncol;
        int64 nc = idx % ncol;

        float* c1 = c1_in + nr * nvec;
        float* c2 = c2_in + nc * nvec;

        float point = 0;

        for (int64 n = 0; n < nvec; n++) {
            if ((c1[n] > 0.5f) == (c2[n] > 0.5f)) point = point + 1;
        }
        
        p_out[idx] = point;
    }
}

__global__ void ker_eveal_vector_dist(int64 size, float* d_out, float* c1_in, float* c2_in, int64 nrow, int64 ncol, int64 nvec) {
    int64 idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64 nr = idx / ncol;
        int64 nc = idx % ncol;

        float* c1 = c1_in + nr * nvec;
        float* c2 = c2_in + nc * nvec;

        float sq_sum = 0;

        for (int64 n = 0; n < nvec; n++) {
            sq_sum += (c1[n] - c2[n]) * (c1[n] - c2[n]);
        }
        
        d_out[idx] = __fsqrt_rn(sq_sum / nvec);
    }
}
