/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kai_kernels.cuh"
#include "math_constants.h"

__global__ void kai_ker_set(KInt size, KFloat* y, KFloat term) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = term;
    }
}

__global__ void kai_ker_matmul(KInt size, KFloat* a, KFloat* h, KFloat* w, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(nrow * ncol == size);

        KInt nr = idx / ncol;
        KInt nc = idx % ncol;

        KInt xpos = nr * nvec;
        KInt wpos = nc;

        KFloat sum = 0;

        for (KInt nv = 0; nv < nvec; nv++) {
            sum += h[xpos] * w[wpos];
            xpos++, wpos += ncol;
        }

        a[idx] = sum;
    }
}

__global__ void kai_ker_add_bias(KInt size, KFloat* y, KFloat* a, KFloat* b, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nc = idx % ncol;
        y[idx] = a[idx] + b[nc];
    }
}

__global__ void kai_ker_transpose(KInt size, KFloat* t, KFloat* a, KInt rows, KInt cols) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / cols;
        KInt nc = idx % cols;

        t[idx] = a[nc * rows + nr];
    }
}

__global__ void kai_ker_sum_on_column(KInt size, KFloat* s, KFloat* a, KInt rows, KInt cols) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat sum = 0;

        for (KInt n = 0; n < rows; n++) {
            sum += a[n * cols + idx];
        }
        s[idx] = sum;
    }
}

__global__ void kai_ker_activate(KInt size, KFloat* y, KFloat* x, KInt nFunc, KFloat leaky_alpha) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    KFloat t;

    if (idx < size) {
        switch ((Ken_actfunc)nFunc) {
        case Ken_actfunc::none:
            y[idx] = x[idx];
            break;
        case Ken_actfunc::relu:
            y[idx] = (x[idx] > 0) ? x[idx] : 0;
            break;
        case Ken_actfunc::sigmoid:
            t = x[idx];
            y[idx] = (KFloat)((t > 0) ? (1.0f / (1.0f + ::expf(-t))) : (::expf(t) / (::expf(t) + 1.0)));
            break;
        case Ken_actfunc::tanh:
            t = x[idx] * 2;
            y[idx] = (KFloat)((t > 0) ? ((1.0 - ::expf(-t)) / (1.0 + ::expf(-t))) : ((::expf(t) - 1.0) / (::expf(t) + 1.0)));
            break;
        case Ken_actfunc::leaky_relu:
            y[idx] = (x[idx] > 0) ? x[idx] : x[idx] * leaky_alpha;
            break;
        case Ken_actfunc::gelu:
            t = x[idx];
            t = t * 0.797885f + t * t * t * 0.035677f;
            t = t * 2;
            t = (KFloat)((t > 0) ? ((1.0 - ::expf(-t)) / (1.0 + ::expf(-t))) : ((::expf(t) - 1.0) / (::expf(t) + 1.0)));

            y[idx] = x[idx] * 0.5f * t;
            break;
        }
    }
}

__global__ void kai_ker_activate_backprop(KInt size, KFloat* g_out, KFloat* g_in, KFloat* x, KFloat* y, KInt nFunc, KFloat leaky_alpha) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    KFloat s, t, u, v, w;

    if (idx < size) {
        switch ((Ken_actfunc)nFunc) {
        case Ken_actfunc::none:
            g_out[idx] = g_in[idx];
            break;
        case Ken_actfunc::relu:
            g_out[idx] = (x[idx] > 0) ? g_in[idx] : 0;
            break;
        case Ken_actfunc::sigmoid:
            g_out[idx] = g_in[idx] * y[idx] * (1 - y[idx]);
            break;
        case Ken_actfunc::tanh:
            g_out[idx] = g_in[idx] * (1 - y[idx] * y[idx]);
            break;
        case Ken_actfunc::leaky_relu:
            g_out[idx] = (x[idx] > 0) ? g_in[idx] : g_in[idx] * leaky_alpha;
            break;
        case Ken_actfunc::gelu:
            s = x[idx];
            t = s * 0.797885f + s * s * s * 0.035677f;
            u = (KFloat)((t > 0) ? ((1.0 - ::expf(-t)) / (1.0 + ::expf(-t))) : ((::expf(t) - 1.0) / (::expf(t) + 1.0)));
            v = s * s * 2 * 0.035677f + 0.797885f;
            w = -0.5f * (u + 1) * ((u - 1) * s * v + 1.0f);
            g_out[idx] = g_in[idx] * w;
            break;
        }
    }
}

__global__ void kai_ker_minus(KInt size, KFloat* y, KFloat* a) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = -a[idx];
    }
}

__global__ void kai_ker_binary_op(KInt size, KFloat* y, KFloat* a, KFloat* b, exp_op op_code, KInt vec_size1, KInt vec_size2) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt apos = idx / vec_size2;
        KInt bpos = idx / vec_size1;

        switch (op_code) {
        case exp_op::add:
            y[idx] = a[apos] + b[bpos];
            break;
        case exp_op::sub:
            y[idx] = a[apos] - b[bpos];
            break;
        case exp_op::mult:
            y[idx] = a[apos] * b[bpos];
            break;
        case exp_op::div:
            y[idx] = a[apos] / b[bpos];
            break;
        case exp_op::_and:
            y[idx] = (a[apos] != 0 && b[bpos] != 0) ? 1 : 0;
            break;
        case exp_op::_or:
            y[idx] = (a[apos] != 0 || b[bpos] != 0) ? 1 : 0;
            break;
        case exp_op::gt:
            y[idx] = (a[apos] > b[bpos]) ? 1 : 0;
            break;
        case exp_op::lt:
            y[idx] = (a[apos] < b[bpos]) ? 1 : 0;
            break;
        case exp_op::ge:
            y[idx] = (a[apos] >= b[bpos]) ? 1 : 0;
            break;
        case exp_op::le:
            y[idx] = (a[apos] <= b[bpos]) ? 1 : 0;
            break;
        case exp_op::equal:
            y[idx] = (a[apos] == b[bpos]) ? 1 : 0;
            break;
        case exp_op::sigmoid_cross_entropy_with_logits:
            y[idx] = MAX(a[apos], 0) - a[apos] * b[bpos] + ::logf(1.0f + ::expf(-::fabs(a[apos])));
            break;
        default:
            if (idx == 0) assert(0);
            y[idx] = 0;
            break;
        }
    }
}

__global__ void kai_ker_binary_op(KInt size, KFloat* y, KFloat* a, KFloat term, exp_op op_code) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        switch (op_code) {
        case exp_op::add:
            y[idx] = a[idx] + term;
            break;
        case exp_op::sub:
            y[idx] = a[idx] - term;
            break;
        case exp_op::mult:
            y[idx] = a[idx] * term;
            break;
        case exp_op::div:
            y[idx] = (term != 0.0f) ? a[idx] / term : term;
            break;
        case exp_op::sigmoid_cross_entropy_with_logits:
            y[idx] = MAX(a[idx], 0) - a[idx] * term + ::logf(1.0f + ::expf(-::fabs(a[idx])));
            break;
        case exp_op::_and:
            y[idx] = (a[idx] != 0 && term != 0) ? 1 : 0;
            break;
        case exp_op::_or:
            y[idx] = (a[idx] != 0 || term != 0) ? 1 : 0;
            break;
        case exp_op::gt:
            y[idx] = (a[idx] > term) ? 1 : 0;
            break;
        case exp_op::lt:
            y[idx] = (a[idx] < term) ? 1 : 0;
            break;
        case exp_op::ge:
            y[idx] = (a[idx] >= term) ? 1 : 0;
            break;
        case exp_op::le:
            y[idx] = (a[idx] <= term) ? 1 : 0;
            break;
        case exp_op::equal:
            y[idx] = (a[idx] == term) ? 1 : 0;
            break;
        default:
            y[idx] = 0;
            break;
        }
    }
}

__global__ void kai_ker_add(KInt size, KFloat* y, KFloat* a, KFloat* b) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] + b[idx];
    }
}

__global__ void kai_ker_sub(KInt size, KFloat* y, KFloat* a, KFloat* b) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] - b[idx];
    }
}

__global__ void kai_ker_mul(KInt size, KFloat* y, KFloat* a, KFloat* b) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] * b[idx];
    }
}

__global__ void kai_ker_div(KInt size, KFloat* y, KFloat* a, KFloat* b) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] / b[idx];
    }
}

__global__ void kai_ker_add(KInt size, KFloat* y, KFloat* a, KFloat term) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] + term;
    }
}

__global__ void kai_ker_sub(KInt size, KFloat* y, KFloat* a, KFloat term) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] - term;
    }
}

__global__ void kai_ker_mul(KInt size, KFloat* y, KFloat* a, KFloat term) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] * term;
    }
}

__global__ void kai_ker_div(KInt size, KFloat* y, KFloat* a, KFloat term) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] / term;
    }
}

__global__ void kai_ker_sign(KInt size, KFloat* y, KFloat* a) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = (a[idx] > 0) ? 1.0f : ((a[idx] < 0) ? -1.0f : 0);
    }
}

__global__ void kai_ker_square(KInt size, KFloat* y, KFloat* a) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = a[idx] * a[idx];
    }
}

__global__ void kai_ker_sqrt(KInt size, KFloat* y, KFloat* a) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = ::sqrtf(a[idx]);
    }
}

__global__ void kai_ker_log(KInt size, KFloat* y, KFloat* a) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = ::logf(a[idx]);
    }
}

__global__ void kai_ker_exp(KInt size, KFloat* y, KFloat* a) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = ::expf(a[idx]);
    }
}

__global__ void kai_ker_sigmoid(KInt size, KFloat* y, KFloat* a) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (a[idx] >= 0) {
            y[idx] = 1.0f / (1.0f + ::expf(-a[idx]));
        }
        else {
            y[idx] = ::expf(a[idx]) / (1.0f + ::expf(a[idx]));
        }
    }
}

__global__ void kai_ker_sigmoid_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* y_in) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx_out[idx] = gy_in[idx] * y_in[idx] * (1.0f - y_in[idx]);
    }
}


__global__ void kai_ker_sum(KInt size, KFloat* x, KInt xsize, KInt range) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt dest = idx * range;
        KInt start = dest;
        KInt end = MIN(dest + range, xsize);
        KInt step = range / KAI_ADD_RANGE;

        KFloat sum = 0;
        for (KInt n = start; n < end; n += step) {
            sum += x[n];
        }
        x[dest] = sum;
    }
}

__global__ void kai_ker_mul_scalar_on(KInt size, KFloat* y, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] *= coef;
    }
}

__global__ void kai_ker_eval_adam_delta(KInt size, KFloat* d, KFloat* g, KFloat* s, KFloat* t, KFloat n, KFloat ro1, KFloat ro2, KFloat epsilon) {
    //__global__ void kai_ker_update_param_adam(KInt size, KFloat* pm, KFloat* s, KFloat* t, KFloat* g, KFloat n, KFloat ro1, KFloat ro2, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KFloat grad = g[idx];

        s[idx] = s[idx] * ro1 + grad * (1.0f - ro1);
        t[idx] = t[idx] * ro2 + grad * grad * (1.0f - ro2);

        KFloat sm = s[idx] / (1.0f - ::powf(ro1, n));
        KFloat tm = t[idx] / (1.0f - ::powf(ro2, n));

        KFloat delta = sm / (::sqrtf(tm) + epsilon);

        //d[idx] = grad + delta;
        // 교재 내용에 따르면 아래와 같아야 한다. 확실히 확인하자.
        d[idx] = delta;
    }
}

__global__ void kai_ker_apply_decay(KInt size, KFloat* d, KFloat* p, KFloat* g, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KFloat pm = p[idx];
        KFloat grad = g[idx];

        if (l2_decay > 0) grad += pm * l2_decay;
        if (l1_decay > 0) {
            if (pm > 0) grad += l1_decay;
            else if (pm < 0) grad -= l1_decay;
        }

        d[idx] = grad;
    }
}

__device__ void kai_dev_get_max_sum_for_softmax(KFloat* logits, KInt nvec, KFloat* pmax, KFloat* psum) {
    KFloat max_term = logits[0];
    KFloat sum_exp = 0;

    for (KInt n = 1; n < nvec; n++) {
        if (logits[n] > max_term) max_term = logits[n];
    }

    for (KInt n = 0; n < nvec; n++) {
        sum_exp = sum_exp + ::expf(logits[n] - max_term);
    }

    *pmax = max_term;
    *psum = sum_exp;
}

__global__ void kai_ker_softmax(KInt size, KFloat* y, KFloat* est, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nth = idx % nvec;
        KInt pos = idx - nth;

        KFloat* logits = est + pos;
        KFloat max_term, sum_exp;

        kai_dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KFloat idx_term = ::expf(logits[nth] - max_term);
        y[idx] = idx_term / sum_exp;
    }
}

__global__ void kai_ker_softmax_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* y_in, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt pos = idx - idx % nvec;

        KFloat gsum = 0;

        for (KInt n = 0; n < nvec; n++, pos++) {
            KFloat yac = y_in[pos] * -y_in[idx];
            if (pos == idx) yac += y_in[idx];

            gsum += yac * gy_in[idx];   // 2020 버전의 이 위치 내용이 gy_in[pos]를 곱하도록 되어 있었음, gy_in[idx]가 맞는 듯, 확인 필요
        }

        gx_out[idx] = gsum;
    }
}

__global__ void kai_ker_softmax_cross_entropy_with_logits(KInt size, KFloat* y, KFloat* est, KFloat* ans, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* logits = est + idx * nvec;
        KFloat* answer = ans + idx * nvec;

        KFloat max_term, sum_exp;
        KFloat entropy = 0;

        kai_dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        for (KInt n = 0; n < nvec; n++) {
            KFloat prob_term = ::expf(logits[n] - max_term) / sum_exp;
            KFloat log_term = ::logf(prob_term + 1.0e-10f);
            entropy -= answer[n] * log_term;
        }

        y[idx] = entropy;
    }
}

__global__ void kai_ker_softmax_cross_entropy_with_logits_idx(KInt size, KFloat* y_out, KFloat* est_in, KInt* ans_in, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* logits = est_in + idx * nvec;

        KFloat max_term, sum_exp;

        kai_dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KInt nth = ans_in[idx];

        KFloat prob_term = ::expf(logits[nth] - max_term) / sum_exp;
        KFloat log_term = ::logf(prob_term + 1.0e-10f);
        KFloat entropy = -log_term;

        y_out[idx] = entropy;
    }
}

__global__ void kai_ker_softmax_cross_entropy_with_logits_idx_derv(KInt size, KFloat* y_out, KFloat* est_in, KInt* ans_in, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / nvec;
        KInt nv = idx % nvec;
        KInt begin = idx - nv;

        KFloat* logits = est_in + begin;
        KFloat max_term, sum_exp;

        kai_dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KFloat prob_term = ::expf(logits[nv] - max_term) / sum_exp;
        KFloat ans_prob = (ans_in[nr] == nv) ? 1.0f : 0;

        y_out[idx] = prob_term - ans_prob;
    }
}

__global__ void kai_ker_equal_row(KInt size, KFloat* y, KFloat* est, KFloat* ans, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* p_est = est + idx * nvec;
        KFloat* p_ans = ans + idx * nvec;

        KFloat equal = 1.0f;

        for (KInt n = 0; n < nvec; n++) {
            if (p_est[n] != p_ans[n]) {
                equal = 0;
                break;
            }
        }

        y[idx] = equal;
    }
}

__global__ void kai_ker_max_row(KInt size, KFloat* y, KFloat* x, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* px = x + idx * nvec;

        KFloat max = px[0];

        for (KInt n = 1; n < nvec; n++) {
            if (px[n] > max) max = px[n];
        }

        y[idx] = max;
    }
}

__global__ void kai_ker_max_row_derv(KInt size, KFloat* gx, KFloat* x, KFloat* gy, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx[idx] = 0;

        KInt nr = idx / nvec;
        KInt nv = idx % nvec;

        KFloat* px = x + nr * nvec;

        KFloat max = px[0];
        KInt maxpos = 0;

        for (KInt n = 1; n < nvec; n++) {
            if (px[n] > max) {
                max = px[n];
                maxpos = n;
            }
        }

        if (nv == maxpos) gx[idx] = gy[nr];
    }
}

__global__ void kai_ker_vstack(KInt size, KFloat* dst, KFloat* src, KInt ncol, KInt nfrom, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / nvec;
        KInt nv = idx % nvec;

        KInt dpos = nr * ncol + nfrom + nv;

        dst[dpos] = src[idx];
    }
}
 
__global__ void kai_ker_vstack_derv(KInt size, KFloat* dst, KFloat* src, KInt ncol, KInt nfrom, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / nvec;
        KInt nv = idx % nvec;

        KInt dpos = nr * ncol + nfrom + nv;

        dst[idx] = src[dpos];
    }
}

__global__ void kai_ker_iou(KInt size, KFloat* piou, KFloat* pa1, KFloat* pa2, KInt col1, KInt col2) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (col1 * col2);
        KInt nr = idx / col2 % col1;
        KInt nc = idx % col2;

        KFloat* p1 = pa1 + (nd * col1 + nr) * 4;
        KFloat* p2 = pa2 + (nd * col2 + nc) * 4;

        KFloat lt1 = p1[0], rt1 = p1[1], tp1 = p1[2], bt1 = p1[3];
        KFloat lt2 = p2[0], rt2 = p2[1], tp2 = p2[2], bt2 = p2[3];

        KFloat lt = MAX(lt1, lt2), rt = MIN(rt1, rt2), tp = MAX(tp1, tp2), bt = MIN(bt1, bt2);

        KFloat iou = 0;

        if (lt < rt && tp < bt) {
            KFloat area1 = (rt1 - lt1) * (bt1 - tp1);
            KFloat area2 = (rt2 - lt2) * (bt2 - tp2);
            KFloat inter_area = (rt - lt) * (bt - tp);
            KFloat union_area = area1 + area2 - inter_area;

            iou = inter_area / union_area;
        }

        piou[idx] = iou;
    }
}

/*
__global__ void kai_ker_iou_grad(KInt size, KFloat* pgx, KFloat* pgiou, KFloat* pa2, KInt col1, KInt col2) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (col1 * 4);
        KInt nr = idx / 4 % col1;
        KInt nc = idx % 4;

        KFloat* p1 = pa1 + (nd * col1 + nr) * 4;
        KFloat* p2 = pa2 + (nd * col2 + nc) * 4;

        KFloat lt1 = p1[0], rt1 = p1[1], tp1 = p1[2], bt1 = p1[3];
        KFloat lt2 = p2[0], rt2 = p2[1], tp2 = p2[2], bt2 = p2[3];

        KFloat lt = MAX(lt1, lt2), rt = MIN(rt1, rt2), tp = MAX(tp1, tp2), bt = MIN(bt1, bt2);

        KFloat iou = 0;

        if (lt < rt && tp < bt) {
            KFloat area1 = (rt1 - lt1) * (bt1 - tp1);
            KFloat area2 = (rt2 - lt2) * (bt2 - tp2);
            KFloat inter_area = (rt - lt) * (bt - tp);
            KFloat union_area = area1 + area2 - inter_area;

            iou = inter_area / union_area;
        }

        piou[idx] = iou;
    }
}
*/

__global__ void kai_ker_argmax(KInt size, KFloat* y, KFloat* a, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* row = a + idx * nvec;
        KFloat max_term = row[0];
        KInt argmax = 0;

        for (KInt n = 0; n < nvec; n++) {
            if (row[n] > max_term) {
                max_term = row[n];
                argmax = n;
            }
        }

        y[idx] = (KFloat)argmax;
    }
}

__global__ void kai_ker_maxcol(KInt size, KFloat* y, KFloat* a, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* row = a + idx * nvec;
        KFloat max_term = row[0];

        for (KInt n = 0; n < nvec; n++) {
            if (row[n] > max_term) {
                max_term = row[n];
            }
        }

        y[idx] = max_term;
    }
}

__global__ void kai_ker_conv_kernel(KInt size, KFloat* b, KFloat* x, KFloat* pm, KInt xh, KInt xw, KInt kh, KInt kw, KInt xchn, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * ychn * xchn);
        KInt xrow = (idx / (xw * ychn * xchn)) % xh;
        KInt xcol = (idx / (ychn * xchn)) % xw;
        KInt yn = (idx / xchn) % ychn;
        KInt xn = idx % xchn;

        KFloat sum = 0;

        KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        KInt xpos1 = ndat * xh * xw * xchn + xn;
        KInt kpos1 = xn * ychn + yn;

        for (KInt kr = 0; kr < kh; kr++) {
            KInt row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;

            KInt xpos2 = xpos1 + row * xw * xchn;
            KInt kpos2 = kpos1 + kr * kw * xchn * ychn;

            for (KInt kc = 0; kc < kw; kc++) {
                KInt col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;

                KInt xpos3 = xpos2 + col * xchn;
                KInt kpos3 = kpos2 + kc * xchn * ychn;

                sum += x[xpos3] * pm[kpos3];
            }
        }

        b[idx] = sum;
    }
}

__global__ void kai_ker_conv_sum(KInt size, KFloat* y, KFloat* b, KInt xchn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat sum = 0;
        for (KInt n = 0; n < xchn; n++) sum += b[idx * xchn + n];

        y[idx] = sum;
    }
}

__global__ void kai_ker_conv_add_bias(KInt size, KFloat* y, KFloat* pm, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] += pm[idx % ychn];
    }
}

__global__ void kai_ker_conv_derv_x_kernel(KInt size, KFloat* c, KFloat* gy, KFloat* k, KInt xh, KInt xw, KInt kh, KInt kw, KInt xchn, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * xchn * ychn);
        KInt xrow = (idx / (xw * xchn * ychn)) % xh;
        KInt xcol = (idx / (xchn * ychn)) % xw;
        KInt xn = (idx / ychn) % xchn;
        KInt yn = idx % ychn;

        KFloat sum = 0;

        KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        KInt ypos1 = ndat * xh * xw * ychn + yn;
        KInt kpos1 = xn * ychn + yn;

        for (KInt kr = 0; kr < kh; kr++) {
            KInt yrow = xrow + bh - kr;
            if (yrow < 0 || yrow >= xh) continue;

            KInt ypos2 = ypos1 + yrow * xw * ychn;
            KInt kpos2 = kpos1 + kr * kw * xchn * ychn;

            for (KInt kc = 0; kc < kw; kc++) {
                KInt ycol = xcol + bw - kc;
                if (ycol < 0 || ycol >= xw) continue;

                KInt ypos3 = ypos2 + ycol * ychn;
                KInt kpos3 = kpos2 + kc * xchn * ychn;

                sum += gy[ypos3] * k[kpos3];
            }
        }

        c[idx] = sum;
    }
}

__global__ void kai_ker_conv_derv_x_sum(KInt size, KFloat* gx, KFloat* c, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat sum = 0;
        for (KInt n = 0; n < ychn; n++) sum += c[idx * ychn + n];

        gx[idx] = sum;
    }
}

__global__ void kai_ker_conv_derv_kw_x(KInt size, KFloat* d, KFloat* gy, KFloat* x, KInt mb_size, KInt xh, KInt xw, KInt kh, KInt kw, KInt xchn, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt kr = (idx / (kw * xchn * ychn * mb_size * xh)) % kh;
        KInt kc = (idx / (xchn * ychn * mb_size * xh)) % kw;
        KInt xn = (idx / (ychn * mb_size * xh)) % xchn;
        KInt yn = (idx / (mb_size * xh)) % ychn;
        KInt ndat = (idx / xh) % mb_size;
        KInt xrow = idx % xh;

        KFloat sum = 0;

        KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        for (KInt xcol = 0; xcol < xw; xcol++) {
            KInt yrow = xrow - kr + bh;
            KInt ycol = xcol - kc + bw;

            if (yrow < 0 || yrow >= xh) continue;
            if (ycol < 0 || ycol >= xw) continue;

            KInt xpos = ((ndat * xh + xrow) * xw + xcol) * xchn + xn;
            KInt ypos = ((ndat * xh + yrow) * xw + ycol) * ychn + yn;

            sum += gy[ypos] * x[xpos];
        }

        d[idx] = sum;
    }
}

__global__ void kai_ker_conv_derv_kw_sum1(KInt size, KFloat* dout, KInt xh) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt spos = idx * xh;

        KFloat sum = 0;

        for (KInt n = 0; n < xh; n++) {
            sum += dout[spos + n];
        }

        dout[spos] = sum;
    }
}

__global__ void kai_ker_conv_derv_kw_sum2(KInt size, KFloat* gw, KFloat* d, KInt mb_size, KInt xh) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt spos = idx * mb_size * xh;

        KFloat sum = 0;

        for (KInt n = 0; n < mb_size; n++) {
            sum += d[spos + n * xh];
        }

        gw[idx] = sum;
    }
}

__global__ void kai_ker_max(KInt size, KFloat* y, KInt* n, KFloat* x, KInt xh, KInt xw, KInt chn, KInt kh, KInt kw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * chn);
        KInt xrow = (idx / (xw * chn)) % xh;
        KInt xcol = (idx / chn) % xw;
        KInt xn = idx % chn;

        KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        KInt xpos_base = ndat * xh * xw * chn + xn;
        //KInt xpos_base = ((ndat * xh + xrow - bh) * xw + xcol - bw) * xchn + xn;

        KFloat maxval = 0;
        KInt argmax = -1;

        for (KInt kr = 0; kr < kh; kr++) {
            KInt row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;
            for (KInt kc = 0; kc < kw; kc++) {
                KInt col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;
                KInt xpos = xpos_base + row * xw * chn + col * chn;
                KFloat term = x[xpos];
                if (argmax < 0 || maxval < term) {
                    maxval = term;
                    argmax = (KInt)(kr * kw + kc);
                }
            }
        }

        y[idx] = maxval;
        n[idx] = argmax;
    }
}

__global__ void kai_ker_max_derv(KInt size, KFloat* gx, KInt* n, KFloat* gy, KInt xh, KInt xw, KInt chn, KInt kh, KInt kw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * chn);
        KInt xrow = (idx / (xw * chn)) % xh;
        KInt xcol = (idx / chn) % xw;
        KInt xn = idx % chn;

        KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        KInt ypos_base = ndat * xh * xw * chn + xn;

        KFloat sum = 0;

        for (KInt kr = 0; kr < kh; kr++) {
            KInt row = xrow - kr + bh;
            if (row < 0 || row >= xh) continue;
            for (KInt kc = 0; kc < kw; kc++) {
                KInt col = xcol - kc + bw;
                if (col < 0 || col >= xw) continue;
                KInt ypos = ypos_base + row * xw * chn + col * chn;
                KInt argmax = (KInt)(kr * kw + kc);
                if (n[ypos] != argmax) continue;
                sum += gy[ypos];
            }
        }

        gx[idx] = sum;
    }
}

__global__ void kai_ker_avg(KInt size, KFloat* y, KInt* n, KFloat* x, KInt xh, KInt xw, KInt xchn, KInt kh, KInt kw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * xchn);
        KInt xrow = (idx / (xw * xchn)) % xh;
        KInt xcol = (idx / xchn) % xw;
        KInt xn = idx % xchn;

        KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        KInt xpos = ((ndat * xh + xrow - bh) * xw + xcol - bw) * xchn + xn;

        KFloat sum = 0;
        KInt cnt = 0;

        for (KInt kr = 0; kr < kh; kr++) {
            KInt row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;
            for (KInt kc = 0; kc < kw; kc++) {
                KInt col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;
                sum += x[xpos + (kr * xw + kc) * xchn];
                cnt++;
            }
        }

        y[idx] = (KFloat)sum / (KFloat)cnt;
        n[idx] = cnt;
    }
}

__global__ void kai_ker_avg_derv(KInt size, KFloat* gx, KInt* n, KFloat* gy, KInt xh, KInt xw, KInt chn, KInt kh, KInt kw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * chn);
        KInt xrow = (idx / (xw * chn)) % xh;
        KInt xcol = (idx / chn) % xw;
        KInt xn = idx % chn;

        KInt bh = kh / 2, bw = kw / 2;
        KInt ypos_base = ((ndat * xh + xrow - bh) * xw + xcol - bw) * chn + xn;

        KFloat sum = 0;

        for (KInt kr = 0; kr < kh; kr++) {
            KInt row = xrow + kr - bh;
            if (row < 0 || row >= xh) continue;
            for (KInt kc = 0; kc < kw; kc++) {
                KInt col = xcol + kc - bw;
                if (col < 0 || col >= xw) continue;
                KInt ypos = ypos_base + (kr * xw + kc) * chn;
                sum += gy[ypos] / (KFloat)n[ypos];
            }
        }

        gx[idx] = sum;
    }
}

__global__ void kai_ker_globalavg(KInt size, KFloat* y, KFloat* x, KInt xh, KInt xw, KInt chn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / chn;
        KInt xn = idx % chn;

        KFloat* xp = x + ndat * xh * xw * chn + xn;

        KFloat sum = 0;

        for (KInt nh = 0; nh < xh; nh++) {
            for (KInt nw = 0; nw < xw; nw++) {
                sum += xp[(nh * xw + nw) * chn];
            }
        }

        y[idx] = sum / (xh * xw);
    }
}

__global__ void kai_ker_globalavg_derv(KInt size, KFloat* gx, KFloat* gy, KInt xh, KInt xw, KInt chn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * chn);
        KInt xn = idx % chn;

        gx[idx] = gy[ndat * chn + xn] / (xh * xw);
    }
}

// 2020 버전에 비해 커널 크기, 패딩 방식으 고려 는 방식으로 단순화함
__global__ void kai_ker_stride(KInt size, KFloat* y, KFloat* x, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (yh * yw * chn);
        KInt row = (idx / (yw * chn)) % yh;
        KInt col = (idx / chn) % yw;
        KInt xn = idx % chn;

        KInt bh = (sh - 1) / 2, bw = (sw - 1) / 2;

        KInt rpos = row * sh + bh;
        KInt cpos = col * sw + bw;

        KInt xpos = ((ndat * xh + rpos) * xw + cpos) * chn + xn;

        y[idx] = x[xpos];
    }
}

__global__ void kai_ker_stride_derv(KInt size, KFloat* gx, KFloat* gy, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * chn);
        KInt nh = (idx / (xw * chn)) % xh;
        KInt nw = (idx / chn) % xw;
        KInt nx = idx % chn;

        KInt bh = (sh - 1) / 2, bw = (sw - 1) / 2;

        if ((nh - bh) % sh != 0) return;
        if ((nw - bw) % sw != 0) return;

        KInt mh = (nh - bh) / sh;
        KInt mw = (nw - bw) / sw;

        KInt ypos = ((ndat * yh + mh) * yw + mw) * chn + nx;

        gx[idx] = gy[ypos];
    }
}

__global__ void kai_ker_bn_collect(KInt size, KFloat* avgout, KFloat* varout, KFloat* mavgout, KFloat* mvarout, KFloat* x, KInt hsize, KFloat momentum) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nrest = hsize / size;

        KFloat sum = 0, sqsum = 0;

        for (KInt n = idx; n < hsize; n += size) {
            KFloat term = x[n];
            sum += term;
            sqsum += term * term;
        }

        KFloat avg = sum / (KFloat) nrest;
        KFloat var = sqsum / (KFloat)nrest - avg * avg;

        avgout[idx] = avg;
        varout[idx] = var;

        mavgout[idx] = mavgout[idx] * momentum + avg * (1 - momentum);
        mvarout[idx] = mvarout[idx] * momentum + var * (1 - momentum);
    }
}

__global__ void kai_ker_bn_normalize(KInt size, KFloat* y, KFloat* x, KFloat* avg, KFloat* var, KInt bsize, KFloat epsilon) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt bidx = idx % bsize;
        
        KFloat std = ::sqrtf(var[bidx] + epsilon);
        y[idx] = (x[idx] - avg[bidx]) / std;
    }
}

__global__ void kai_ker_bn_rescale(KInt size, KFloat* y, KFloat* x, KFloat* scale, KFloat* shift, KInt bsize) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt bidx = idx % bsize;

        y[idx] = x[idx] * scale[bidx] + shift[bidx];
    }
}

__global__ void kai_ker_bn_rescale_derv_pm(KInt size, KFloat* gscale, KFloat* gshift, KFloat* gx, KFloat* x, KInt hsize) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat scale_sum = 0, shift_sum = 0;

        for (KInt n = idx; n < hsize; n += size) {
            scale_sum += gx[n] * x[n];
            shift_sum += gx[n];
        }

        gscale[idx] = scale_sum;
        gshift[idx] = shift_sum;
    }
}

__global__ void kai_ker_bn_rescale_derv_x(KInt size, KFloat* gx_out, KFloat* gy_in, KFloat* scale, KInt bsize) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt bidx = idx % bsize;

        gx_out[idx] = gy_in[idx] * scale[bidx];
    }
}

__global__ void kai_ker_bn_norm_derv(KInt size, KFloat* ghout, KFloat* var, KInt bsize, KFloat epsilon) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt bidx = idx % bsize;
        KFloat std = ::sqrtf(var[bidx] + epsilon);

        ghout[idx] = ghout[idx] / std;
    }
}

__global__ void kai_ker_put_branch(KInt size, KFloat* y, KFloat* b, KInt ychn, KInt bchn, KInt chn_from) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / bchn;
        KInt bcol = idx % bchn;
        
        KInt ypos = row * ychn + chn_from + bcol;

        y[ypos] = b[idx];
    }
}
__global__ void kai_ker_get_branch(KInt size, KFloat* gb, KFloat* gy, KInt ychn, KInt bchn, KInt chn_from) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / bchn;
        KInt bcol = idx % bchn;

        KInt ypos = row * ychn + chn_from + bcol;

        gb[idx] = gy[ypos];
    }
}

__global__ void kai_ker_dropout(KInt size, KFloat* y, KFloat* x, KFloat* m, KFloat keep_ratio) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = x[idx] * m[idx] / keep_ratio;
    }
}

__global__ void kai_ker_dropout_derv(KInt size, KFloat* gx, KFloat* gy, KFloat* m, KFloat keep_ratio) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx[idx] = gy[idx] * m[idx] / keep_ratio;
    }
}

__global__ void kai_ker_tile_chn(KInt size, KFloat* y, KFloat* x, KInt ychn, KInt xchn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / ychn;
        KInt tn = idx % ychn;

        KInt ratio = ychn / xchn;
        KInt xn = tn / ratio;

        KInt xidx = row * xchn + xn;

        y[idx] += x[xidx];
    }
}

__global__ void kai_ker_untile_chn(KInt size, KFloat* gt, KFloat* gy, KInt ychn, KInt xchn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / xchn;
        KInt tn = idx % xchn;

        KInt ratio = ychn / xchn;

        KFloat sum = 0;

        for (KInt n = 0; n < ratio; n++) {
            KInt yidx = row * ychn + tn * ratio + n;
            sum += gy[yidx];
        }

        gt[idx] = sum;
    }
}

__global__ void kai_ker_wave_slices_to_complex(KInt size, KFloat* c, KFloat* w, KInt step_width, KInt step_cnt, KInt fft_width, KInt fetch_width) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (step_cnt * fft_width * 2);
        KInt ns = idx / (fft_width * 2) % step_cnt;
        KInt nx = (idx / 2) % fft_width;
        bool is_real = (idx % 2) == 0;

        if (is_real) {
            KInt wpos = nd * fetch_width + ns * step_width + nx;
            c[idx] = w[wpos];
        }
        else {
            c[idx] = 0;
        }
    }
}

__global__ void kai_ker_fft_step_split(KInt size, KFloat* dst, KFloat* src, KInt data_num, KInt step, KInt nd_base) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (data_num * 2) + nd_base;
        KInt nx = (idx / 2) % data_num;
        bool is_real = (idx % 2) == 0;

        KInt stride = data_num / step;

        KInt pos1 = (nx / stride * (2 * stride)) % data_num + nx % stride;
        KInt pos2 = pos1 + stride;

        KFloat x1_real = src[(nd * data_num + pos1) * 2];
        KFloat x1_image = src[(nd * data_num + pos1) * 2 + 1];

        KFloat x2_real = src[(nd * data_num + pos2) * 2];
        KFloat x2_image = src[(nd * data_num + pos2) * 2 + 1];

        KFloat theta = -2 * CUDART_PI_F * (nx / stride * stride) / data_num;

        KFloat t_real = __cosf(theta);
        KFloat t_image = __sinf(theta);

        KInt didx = (nd * data_num + nx) * 2;

        if (is_real)
            dst[didx] = x1_real + x2_real * t_real - x2_image * t_image;
        else
            dst[didx + 1] = x1_image + x2_real * t_image + x2_image * t_real;
    }
}

__global__ void kai_ker_complex_to_abs_mean(KInt size, KFloat* a, KFloat* c, KInt data_num, KInt freq_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / freq_cnt;
        KInt nx = idx % freq_cnt;

        KInt win_size = data_num / (2 * freq_cnt);
        KInt win_start = data_num / 2 - win_size * (freq_cnt - nx);

        KFloat sum = 0;

        for (KInt n = 0; n < win_size; n++) {
            KInt cidx = nd * data_num + win_start + n;

            KFloat real = c[cidx * 2];
            KFloat image = c[cidx * 2 + 1];

            sum += __fsqrt_rn(real * real + image * image);
        }

        a[idx] = sum / win_size;
    }
}

__global__ void kai_ker_rnn_combine_ex_inp(KInt size, KFloat* ex_inp_out, KFloat* x_in, KFloat* rec_in, KInt timesteps, KInt timefeats, KInt recur_size, bool inseq, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / (timefeats + recur_size);
        KInt col = idx % (timefeats + recur_size);

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

__global__ void kai_ker_rnn_split_ex_inp(KInt size, KFloat* gx_out, KFloat* grec_out, KFloat* gex_inp_in, KInt timesteps, KInt timefeats, KInt recur_size, bool inseq, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / (timefeats + recur_size);
        KInt col = idx % (timefeats + recur_size);

        if (col < timefeats) {
            if (inseq) {
                KInt xidx = ((row * timesteps) + tn) * timefeats + col;
                gx_out[xidx] = gex_inp_in[idx];
            }
            else {
                KInt xidx = row * timefeats + col;
                //gx_out[xidx] = gex_inp_in[idx]; // 입력을 첫 시간대에만 공급하는 경우를 선택했을 때 이 처리를 지원
                gx_out[xidx] = gx_out[xidx] + gex_inp_in[idx]; // 입력을 매 시간대에 공급하는 경우를 선택했을 때 이 처리를 지원
            }
        }
        else {
            KInt ridx = row * recur_size + (col - timefeats);
            grec_out[ridx] = gex_inp_in[idx];
        }
    }
}

__global__ void kai_ker_rnn_fill_output_slice(KInt size, KFloat* y_out, KFloat* x_in, KInt timesteps, KInt recur_size, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / recur_size;
        KInt col = idx % recur_size;

        y_out[((row * timesteps) + tn) * recur_size + col] = x_in[idx];
    }
}

__global__ void kai_ker_rnn_add_time_slice(KInt size, KFloat* gy_out, KFloat* gx_in, KInt timesteps, KInt recur_size, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / recur_size;
        KInt col = idx % recur_size;

        gy_out[idx] += gx_in[((row * timesteps) + tn) * recur_size + col];
    }
}

__global__ void kai_ker_lstm_gate(KInt size, KFloat* gates, KFloat* affine) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt gate_type = idx % 4;

        KFloat x = affine[idx];

        if (gate_type < 3) {    // forget gate, input_gate, output_gate: sigmoid
            gates[idx] = (KFloat)((x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (::expf(x) + 1.0f)));   // sigmoid
        }
        else {  // block_input: tanh
            gates[idx] = (KFloat)((x > 0) ? ((1.0f - ::expf(-x)) / (1.0f + ::expf(-x))) : ((::expf(x) - 1.0f) / (::expf(x) + 1.0f)));   // tanh
        }
    }
}

__global__ void kai_ker_lstm_proc(KInt size, KFloat* recur_out, KFloat* state_out, KFloat* state_in, KFloat* gates_in) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat forget_gate = gates_in[4 * idx + 0];
        KFloat input_gate = gates_in[4 * idx + 1];
        KFloat output_gate = gates_in[4 * idx + 2];
        KFloat input_block = gates_in[4 * idx + 3];

        KFloat memory = state_in[idx] * forget_gate;
        KFloat input = input_gate * input_block;
        KFloat new_state = memory + input;

        KFloat n = new_state;

        KFloat output = (KFloat)((n > 0) ? ((1.0f - ::expf(-n)) / (1.0f + ::expf(-n))) : ((::expf(n) - 1.0f) / (::expf(n) + 1.0f)));   // tanh

        state_out[idx] = new_state;
        recur_out[idx] = output * output_gate;
    }
}

__global__ void kai_ker_lstm_gate_derv(KInt size, KFloat* g_affine, KFloat* g_gates, KFloat* gates) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt gate_type = idx % 4;

        KFloat g = g_gates[idx];
        KFloat y = gates[idx];

        if (gate_type < 3) {    // forget gate, input_gate, output_gate: sigmoid
            g_affine[idx] = g * y * (1.0f - y);
        }
        else {  // block_input: tanh
            g_affine[idx] = g * (1.0f - y * y);
        }
    }
}

__global__ void kai_ker_lstm_proc_derv(KInt size, KFloat* g_gates_out, KFloat* g_state_out, KFloat* g_state_in, KFloat* g_recur_in, KFloat* gates_in, KFloat* state_in, KFloat* rec_in) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat forget_gate = gates_in[4 * idx + 0];
        KFloat input_gate = gates_in[4 * idx + 1];
        KFloat output_gate = gates_in[4 * idx + 2];
        KFloat input_block = gates_in[4 * idx + 3];

        KFloat output = rec_in[idx] / output_gate;

        KFloat g_output = g_recur_in[idx] * output_gate;
        KFloat g_output_gate = g_recur_in[idx] * output;

        KFloat g_new_state = g_state_in[idx] + g_output * (1 - output * output); // g_memory = g_input = g_new_state
        //KFloat g_memory = g_new_state;
        //KFloat g_input = g_new_state;

        KFloat g_input_gate = input_block * g_new_state; // g_input = g_new_state
        KFloat g_input_block = input_gate * g_new_state; // g_input = g_new_state

        KFloat old_state = state_in[idx];

        KFloat g_forget_gate = old_state * g_new_state; // g_memory = g_new_state
        KFloat g_old_state = forget_gate * g_new_state; // g_memory = g_new_state

        g_gates_out[4 * idx + 0] = g_forget_gate;
        g_gates_out[4 * idx + 1] = g_input_gate;
        g_gates_out[4 * idx + 2] = g_output_gate;
        g_gates_out[4 * idx + 3] = g_input_block;

        g_state_out[idx] = g_old_state;
    }
}

__global__ void kai_ker_gru_combine_extra(KInt size, KFloat* cuda_x2, KFloat* ext_in, KFloat* gates, KInt ext_size, KInt inp_size, KInt rec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / ext_size;
        KInt nc = idx % ext_size;

        if (nc < inp_size) cuda_x2[idx] = ext_in[idx];
        else {
            KInt rpos = nr * rec_size + (nc - inp_size);
            KFloat r_gate = gates[rpos * 2];
            cuda_x2[idx] = r_gate * ext_in[idx];
        }
    }
}

__global__ void kai_ker_gru_combine_extra_derv(KInt size, KFloat* g_gates, KFloat* g_ext_inout, KFloat* ext_in, KFloat* gates, KInt ext_size, KInt inp_size, KInt rec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / ext_size;
        KInt nc = idx % ext_size;

        if (nc < inp_size) {
            // g_ext_inout[idx] = g_ext_inout[idx]; // need nothing
        }
        else {
            KInt rpos = nr * rec_size + (nc - inp_size);
            KFloat r_gate = gates[rpos * 2];
            g_gates[rpos] = g_ext_inout[idx] * ext_in[idx];
            g_ext_inout[idx] = g_ext_inout[idx] * r_gate;
        }
    }
}

__global__ void kai_ker_gru_proc(KInt size, KFloat* cuda_r2, KFloat* cuda_r1, KFloat* cuda_gt, KFloat* cuda_in) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat x = cuda_in[idx];
        KFloat r = cuda_r1[idx];

        KFloat z = cuda_gt[idx * 2 + 1];
        KFloat input = (KFloat)((x > 0) ? ((1.0f - ::expf(-x)) / (1.0f + ::expf(-x))) : ((::expf(x) - 1.0f) / (::expf(x) + 1.0f)));   // tanh

        cuda_r2[idx] = (1 - z) * r + z * input;
    }
}

__global__ void kai_ker_gru_proc_derv(KInt size, KFloat* g_inp, KFloat* g_gate, KFloat* g_rec_out, KFloat* g_rec_in, KFloat* pre_rec, KFloat* gate, KFloat* inp) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat x = inp[idx];
        KFloat r = pre_rec[idx];
        KFloat z = gate[idx * 2 + 1];

        KFloat input = (KFloat)((x > 0) ? ((1.0f - ::expf(-x)) / (1.0f + ::expf(-x))) : ((::expf(x) - 1.0f) / (::expf(x) + 1.0f)));   // tanh

        //rec[idx] = (1 - z) * r + z * input;

        KFloat g_rec = g_rec_in[idx];

        KFloat g_input = g_rec * z;

        g_inp[idx] = g_input * (1 - input * input);
        g_gate[idx * 2 + 1] = g_rec * (input - r);
        g_rec_out[idx] = g_rec * (-z);
    }
}

__global__ void kai_ker_add_embed_dict(KInt size, float* y_out, float* dic_in, KInt* token_in, KInt vec_size, KInt dic_kind, KInt axis) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / vec_size;
        KInt nv = idx % vec_size;
        
        KInt wid = token_in[nr * dic_kind + axis];
        y_out[idx] += dic_in[wid * vec_size + nv];
    }
}

__global__ void kai_kernel_sgd_update_embed(KInt size, KFloat* w, KFloat* grads, KInt* tokens, KInt dic_cnt, KInt nth, KInt dic_size, KInt vec_size, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nd = idx / vec_size;
        KInt nv = idx % vec_size;

        KInt word_cnt = size / vec_size;
        KInt wid = tokens[nd * dic_cnt + nth];

        assert(wid >= 0 && wid < dic_size);

        float grad = 0;

        for (KInt n = 0; n < word_cnt; n++) {
            if (tokens[n * dic_cnt + nth] != wid) continue;
            if (n < nd) return;
            grad += grads[n * vec_size + nv];
        }

        KInt dpos = wid * vec_size + nv;

        KFloat pm = w[dpos];
        KFloat delta = 0;

        if (l2_decay > 0) delta += pm * l2_decay;
        if (l1_decay > 0) {
            if (pm > 0) delta += l1_decay;
            else if (pm < 0) delta -= l1_decay;
        }

        grad += delta;

        w[dpos] -= grad * learning_rate;
    }
}

__global__ void kai_kernel_adam_update_embed(KInt size, KFloat* w, KFloat* s, KFloat* t, KFloat* n, KFloat* grads, KInt* tokens, KInt dic_cnt, KInt nth, KInt dic_size, KInt vec_size, 
    KFloat learning_rate, KFloat l2_decay, KFloat l1_decay, KFloat ro1, KFloat ro2, KFloat epsilon) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nd = idx / vec_size;
        KInt nv = idx % vec_size;

        KInt word_cnt = size / vec_size;
        KInt wid = tokens[nd * dic_cnt + nth];

        assert(wid >= 0 && wid < dic_size);

        float grad = 0;

        for (KInt k = 0; k < word_cnt; k++) {
            if (tokens[k * dic_cnt + nth] != wid) continue;
            if (k < nd) return;
            grad += grads[k * vec_size + nv];
        }

        KInt dpos = wid * vec_size + nv;

        KFloat pm = w[dpos];
        KFloat delta = 0;

        if (l2_decay > 0) delta += pm * l2_decay;
        if (l1_decay > 0) {
            if (pm > 0) delta += l1_decay; 
            else if (pm < 0) delta -= l1_decay;
        }

        grad += delta;

        n[wid] += 1.0f;

        s[idx] = s[idx] * ro1 + grad * (1.0f - ro1);
        t[idx] = t[idx] * ro2 + grad * grad * (1.0f - ro2);

        KFloat sm = s[idx] / (1.0f - ::powf(ro1, n[wid]));
        KFloat tm = t[idx] / (1.0f - ::powf(ro2, n[wid]));

        KFloat adam = sm / (::sqrtf(tm) + epsilon);

        // 아래 두 줄 중 어느 쪽이 맞는지 확인 필요, 교재 내용은 단순 대치가 맞고 생각해봐도 이 쪽이 맞는 듯
        grad = adam;
        //grad += adam;

        w[dpos] -= grad * learning_rate;
    }
}

__global__ void kai_ker_split_array(KInt size, KFloat* piece_0, KFloat* piece_1, KFloat* piece_2, KFloat* piece_3, KFloat* x_in, KInt piece_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / piece_cnt;
        KInt nv = idx % piece_cnt;

        if (nv == 0) piece_0[nr] = x_in[idx];
        else if (nv == 1) piece_1[nr] = x_in[idx];
        else if (nv == 2) piece_2[nr] = x_in[idx];
        else if (nv == 3) piece_3[nr] = x_in[idx];
    }
}

__global__ void kai_ker_merge_array(KInt size, KFloat* y_out, KFloat* piece_0, KFloat* piece_1, KFloat* piece_2, KFloat* piece_3, KInt piece_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / piece_cnt;
        KInt nv = idx % piece_cnt;

        if (nv == 0) y_out[idx] = piece_0[nr];
        else if (nv == 1) y_out[idx] = piece_1[nr];
        else if (nv == 2) y_out[idx] = piece_2[nr];
        else if (nv == 3) y_out[idx] = piece_3[nr];
    }
}

__global__ void kai_ker_multi_head_matmul_qk(KInt size, KFloat* r_out, KFloat* q_in, KFloat* k_in, KInt timesteps, KInt head_cnt, KInt vec_per_head) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / (head_cnt * timesteps * timesteps);
        KInt nh = idx / (timesteps * timesteps) % head_cnt;
        KInt nt1 = idx / timesteps % timesteps;
        KInt nt2 = idx % timesteps;

        KFloat sum = 0;

        for (KInt nv = 0; nv < vec_per_head; nv++) {
            KFloat q_elem = q_in[((nr * timesteps + nt1) * head_cnt + nh) * vec_per_head + nv];
            KFloat k_elem = k_in[((nr * timesteps + nt2) * head_cnt + nh) * vec_per_head + nv];

            sum += q_elem * k_elem;
        }

        r_out[idx] = sum;
    }
}

__global__ void kai_ker_multi_head_matmul_qk_derv_q(KInt size, KFloat* gq_out, KFloat* gy_in, KFloat* k_in, KInt timesteps, KInt head_cnt, KInt vec_per_head) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / (timesteps * head_cnt * vec_per_head);
        KInt nt1 = idx / (head_cnt * vec_per_head) % timesteps;
        KInt nh = idx / vec_per_head % head_cnt;
        KInt nv = idx % vec_per_head;

        KFloat sum = 0;

        for (KInt nt2 = 0; nt2 < timesteps; nt2++) {
            KFloat gy_elem = gy_in[((nr * head_cnt + nh) * timesteps + nt1) * timesteps + nt2];
            KFloat k_elem = k_in[((nr * timesteps + nt2) * head_cnt + nh) * vec_per_head + nv];

            sum += gy_elem * k_elem;
        }

        gq_out[idx] = sum;
    }
}

__global__ void kai_ker_multi_head_matmul_qk_dev_k(KInt size, KFloat* gk_out, KFloat* gy_in, KFloat* q_in, KInt timesteps, KInt head_cnt, KInt vec_per_head) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / (timesteps * head_cnt * vec_per_head);
        KInt nt2 = idx / (head_cnt * vec_per_head) % timesteps;
        KInt nh = idx / vec_per_head % head_cnt;
        KInt nv = idx % vec_per_head;

        KFloat sum = 0;

        for (KInt nt1 = 0; nt1 < timesteps; nt1++) {
            KFloat q_elem = q_in[((nr * timesteps + nt1) * head_cnt + nh) * vec_per_head + nv];
            KFloat gy_elem = gy_in[((nr * head_cnt + nh) * timesteps + nt1) * timesteps + nt2];

            sum += q_elem * gy_elem;
        }

        gk_out[idx] = sum;
    }
}

__global__ void kai_ker_multi_head_matmul_pv(KInt size, KFloat* r_out, KFloat* p_in, KFloat* v_in, KInt timesteps, KInt head_cnt, KInt vec_per_head) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / (timesteps * head_cnt * vec_per_head);
        KInt nt1 = idx / (head_cnt * vec_per_head) % timesteps;
        KInt nh = idx / vec_per_head % head_cnt;
        KInt nv = idx % vec_per_head;

        KFloat sum = 0;

        for (KInt nt2 = 0; nt2 < timesteps; nt2++) {
            KFloat p_elem = p_in[((nr * head_cnt + nh) * timesteps + nt1) * timesteps + nt2];
            KFloat v_elem = v_in[((nr * timesteps + nt2) * head_cnt + nh) * vec_per_head + nv];

            sum += p_elem * v_elem;
        }

        r_out[idx] = sum;
    }
}

__global__ void kai_ker_multi_head_matmul_pv_derv_p(KInt size, KFloat* gp_out, KFloat* gy_in, KFloat* v_in, KInt timesteps, KInt head_cnt, KInt vec_per_head) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / (head_cnt * timesteps * timesteps);
        KInt nh = idx / (timesteps * timesteps) % head_cnt;
        KInt nt1 = idx / timesteps % timesteps;
        KInt nt2 = idx % timesteps;

        KFloat sum = 0;

        for (KInt nv = 0; nv < vec_per_head; nv++) {
            KFloat gy_elem = gy_in[((nr * timesteps + nt1) * head_cnt + nh) * vec_per_head + nv];
            KFloat v_elem = v_in[((nr * timesteps + nt2) * head_cnt + nh) * vec_per_head + nv];

            sum += gy_elem * v_elem;
        }

        gp_out[idx] = sum;
    }
}

__global__ void kai_ker_multi_head_matmul_pv_derv_v(KInt size, KFloat* gv_out, KFloat* gy_in, KFloat* p_in, KInt timesteps, KInt head_cnt, KInt vec_per_head) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nr = idx / (timesteps * head_cnt * vec_per_head);
        KInt nt2 = idx / (head_cnt * vec_per_head) % timesteps;
        KInt nh = idx / vec_per_head % head_cnt;
        KInt nv = idx % vec_per_head;

        KFloat sum = 0;

        for (KInt nt1 = 0; nt1 < timesteps; nt1++) {
            KFloat p_elem = p_in[((nr * head_cnt + nh) * timesteps + nt1) * timesteps + nt2];
            KFloat gy_elem = gy_in[((nr * timesteps + nt1) * head_cnt + nh) * vec_per_head + nv];

            sum += p_elem * gy_elem;
        }

        gv_out[idx] = sum;
    }
}

__global__ void kai_ker_extract(KInt size, KFloat* e_out, KFloat* x_in, KInt ax_size, KInt index, KInt nCount, KInt nProd) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / (nCount * nProd);
        KInt nx = idx / nProd % nCount;
        KInt nc = idx % nProd;

        KInt xpos = (nr * ax_size + index + nx) * nProd + nc;

        e_out[idx] = x_in[xpos];
    }
}

__global__ void kai_ker_extract_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt ax_size, KInt index, KInt nCount, KInt nProd) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / (nCount * nProd);
        KInt nx = idx / nProd % nCount;
        KInt nc = idx % nProd;

        KInt xpos = (nr * ax_size + index + nx) * nProd + nc;

        gx_out[xpos] = gy_in[idx];
    }
}

__global__ void kai_ker_mask_to_idx(KInt size, KInt* p_map, KInt* p_mask, KInt msize) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt count = 0;

        for (KInt n = 0; n < msize; n++) {
            if (p_mask[n]) {
                p_map[count++] = n;
            }
        }

        p_map[msize] = count;
    }
}

__global__ void kai_ker_filter(KInt size, KFloat* filtered_out, KFloat* x_in, KInt* map_in, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / vec_size;
        KInt nv = idx % vec_size;

        KInt xpos = map_in[nr] * vec_size + nv;

        filtered_out[idx] = x_in[xpos];
    }
}

__global__ void kai_ker_filter_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt* map_in, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / vec_size;
        KInt nv = idx % vec_size;

        KInt xpos = map_in[nr] * vec_size + nv;

        gx_out[xpos] = gy_in[idx];
    }
}

__global__ void kai_ker_expand(KInt size, KFloat* y_out, KFloat* x_in, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (heights * widths * chns);
        KInt nh = idx / (widths * chns) % heights;
        KInt nw = idx / chns % widths;
        KInt nc = idx % chns;

        KInt xheights = heights / hratio;
        KInt xwidths = widths / wratio;

        KInt xh = nh / hratio;
        KInt xw = nw / wratio;

        KInt xidx = ((nd * xheights + xh) * xwidths + xw) * chns + nc;

        y_out[idx] = x_in[xidx];
    }
}

__global__ void kai_ker_expand_derv(KInt size, KFloat* gx_out, KFloat* gy_in, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (heights * widths * chns);
        KInt nh = idx / (widths * chns) % heights;
        KInt nw = idx / chns % widths;
        KInt nc = idx % chns;

        KInt yheights = heights * hratio;
        KInt ywidths = widths * wratio;

        KInt yh = nh * hratio;
        KInt yw = nw * wratio;

        gx_out[idx] = 0;

        KInt yidx_base = ((nd * yheights + yh) * ywidths + yw) * chns + nc;

        for (KInt h = 0; h < hratio; h++) {
            KInt yidx = yidx_base + h * ywidths * chns;
            for (KInt w = 0; w < wratio; w++) {
                gx_out[idx] += gy_in[yidx + w * chns];
            }
        }
    }
}

__global__ void kai_ker_stack_on(KInt size, KFloat* cuda_d, KFloat* cuda_s, KInt block_size, KInt region_size, KInt tail_size, KInt nFrom) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / block_size;
        KInt nb = idx / block_size;

        KInt dpos = nd * region_size + nFrom * tail_size + nb;

        cuda_d[dpos] = cuda_s[idx];
    }
}

__global__ void kai_ker_stack_on_grad(KInt size, KFloat* cuda_gx, KFloat* cuda_gy, KInt block_size, KInt region_size, KInt tail_size, KInt nFrom) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / block_size;
        KInt nb = idx / block_size;

        KInt dpos = nd * region_size + nFrom * tail_size + nb;

        cuda_gx[idx] = cuda_gy[dpos];
    }
}

__global__ void kai_ker_get_subvector(KInt size, KFloat* cuda_s, KFloat* cuda_a, KInt vec_size, KInt nStart, KInt nCount) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / nCount;
        KInt nb = idx % nCount;

        KInt apos = nd * vec_size + nStart + nb;

        cuda_s[idx] = cuda_a[apos];
    }
}

__global__ void kai_ker_acc_grad_subvector(KInt size, KFloat* cuda_gy, KFloat* cuda_gs, KInt vec_size, KInt nStart, KInt nCount) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / nCount;
        KInt nb = idx % nCount;

        KInt apos = nd * vec_size + nStart + nb;

        cuda_gy[apos] += cuda_gs[idx];
    }
}

#ifdef XXX
/*****************************************************************************
       device functions
*****************************************************************************/

__device__ KFloat dev_sigmoid(KFloat x) {
    if (x > 0) return __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-x)));
    else       return __fdiv_rn(__expf(x), __fadd_rn(1.0f, __expf(x)));
}

__device__ KFloat dev_sigmoid_derv(KFloat x) {
    KFloat y = dev_sigmoid(x);
    return __fmul_rn(y, __fsub_rn(1.0f, y));
}

__device__ KFloat dev_sigmoid_derv(KFloat x, KFloat y) {
    return __fmul_rn(y, __fsub_rn(1.0f, y));
}

__device__ KFloat dev_tanh(KFloat x) {
    return 2 * dev_sigmoid(2 * x) - 1;
}

__device__ KFloat dev_tanh_derv(KFloat x, KFloat y) {
    return __fsub_rn(1, __fmul_rn(y, y));
}

__device__ KFloat dev_sigmoid_cross_entropy_with_logits(KFloat x, KFloat z) {
    KFloat pv = __fadd_rn(1.0f, __expf(-x));
    KFloat term1 = __fmul_rn(__logf(pv), z);
    KFloat term2 = __fmul_rn(__fadd_rn(x, __logf(pv)), __fsub_rn(1.0f, z));
    KFloat ent = __fadd_rn(term1, term2);
    return ent;

    //return __fsub_rn(__logf(__fadd_rn(1.0f, __expf(x))), __fmul_rn(x, z));

    /*
    if (x > 0) return __fsub_rn(__fadd_rn(x, __logf(__fadd_rn(1.0f, __expf(-x)))), __fmul_rn(x, z));
    else       return __fsub_rn(__logf(__fadd_rn(1.0f, __expf(x))), __fmul_rn(x, z));
    */

    /*
    KFloat p = 1.0f / (1.0f + __expf(-x));
    KFloat ent = -__logf(p) * z - __logf(1.0f - p) * (1.0f - z);
    return ent;
    */

    /*
    KFloat p = __fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-x)));
    KFloat term1 = __fmul_rn(__logf(p), z);
    KFloat term2 = __fmul_rn(__logf(__fsub_rn(1.0f, p)), __fsub_rn(1.0f, z));
    KFloat ent = -__fadd_rn(term1, term2);
    return ent;
    */
}

__device__ KFloat dev_sigmoid_cross_entropy_with_logits_derv(KFloat x, KFloat z) {
    return __fsub_rn(dev_sigmoid(x), z);
}

/*****************************************************************************
       basic operation
*****************************************************************************/
__global__ void kai_kerit(KInt size, KFloat* y, KFloat value) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = value;
    }
}

__global__ void kai_keritt(KInt size, int* y, int value) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = value;
    }
}

__global__ void kai_keritt64(KInt size, KInt* y, KInt value) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = value;
    }
}

__global__ void kai_ker_add_on(KInt size, KFloat* xout, KFloat* p) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        xout[idx] = __fadd_rn(xout[idx], p[idx]);
    }
}

__global__ void kai_ker_copy_to(KInt size, KFloat* y, KFloat* x) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = x[idx];
    }
}

__global__ void kai_ker_mult_to(KInt size, KFloat* y, KFloat* x1, KFloat* x2) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = __fmul_rn(x1[idx], x2[idx]);
    }
}

__global__ void kai_ker_mult_on(KInt size, KFloat* yout, KFloat* x) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        yout[idx] = __fmul_rn(yout[idx], x[idx]);
    }
}

__global__ void kai_ker_mult_scalar_to(KInt size, KFloat* yout, KFloat* x1, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        yout[idx] = x1[idx] * coef;
    }
}

__global__ void kai_ker_sigmoid_on(KInt size, KFloat* xout) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        xout[idx] = dev_sigmoid(xout[idx]);
    }
}

__global__ void kai_ker_tanh_on(KInt size, KFloat* xout) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        xout[idx] = dev_tanh(xout[idx]);
    }
}

__global__ void kai_ker_tanh_to(KInt size, KFloat* y, KFloat* x) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = dev_tanh(x[idx]);
    }
}

__global__ void kai_ker_sigmoid_derv_on(KInt size, KFloat* gxout, KFloat* y) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gxout[idx] = __fmul_rn(dev_sigmoid_derv(0, y[idx]), gxout[idx]);
    }
}

__global__ void kai_ker_tanh_derv_on(KInt size, KFloat* gxout, KFloat* y) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gxout[idx] = __fmul_rn(dev_tanh_derv(0, y[idx]), gxout[idx]);
    }
}

__global__ void kai_ker_abs(KInt size, KFloat* y, KFloat* x) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat x = x[idx];;
        y[idx] = (x > 0) ? x : -x;
    }
}

__global__ void kai_ker_sqr(KInt size, KFloat* y, KFloat* x) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat x = x[idx];;
        y[idx] = __fmul_rn(x, x);
    }
}

__global__ void kai_ker_binomial(KInt size, KFloat* xout, KFloat prob_threshod) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        xout[idx] = (xout[idx] < prob_threshod) ? 1.0f : 0.0f;
    }
}

__global__ void kai_ker_argmax(KInt size, KInt* n, KFloat* arr, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt from = idx * nvec;
        KInt argmax = 0;
        for (KInt n = 1; n < nvec; n++) {
            if (arr[from+n] > arr[from + argmax]) argmax = n;
        }
        n[idx] = argmax;
    }
}

__global__ void kai_ker_get_hash_idx(KInt size, KInt* n, KFloat* arr, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt from = idx * nvec;
        KInt hash_idx = 0;
        for (KInt n = 0; n < nvec; n++) {
            hash_idx *= 2;
            if (arr[from + n] > 0) hash_idx++; // tanh 활성화 함수를 이용해 0을 기준으로 대칭 분포가 이루어지도록 하고 있음
        }
        n[idx] = hash_idx;
    }
}

__global__ void kai_ker_get_hash_diff(KInt size, KInt* n, KFloat* h1, KFloat* h2, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt hash1 = h1[idx];
        KInt hash2 = h2[idx];

        KInt bit_mask = 1;
        KInt match_cnt = 0;

        for (KInt n = 0; n < nvec; n++) {
            if ((hash1 & bit_mask) == (hash2 & bit_mask)) match_cnt++;
            bit_mask = bit_mask << 1;
        }
        n[idx] = match_cnt;
    }
}

__global__ void kai_ker_near_zero(KInt size, KFloat* y, KFloat* x, KFloat threshold) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat x = x[idx];;
        y[idx] = (-threshold < x && x < threshold) ? 1.0f : 0.0f;
    }
}

__global__ void kai_ker_rescale(KInt size, KFloat* xout, KFloat mean, KFloat std) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        xout[idx] = __fmaf_rn(xout[idx], std, mean);
    }
}

__global__ void kai_ker_sum(KInt size, KFloat* xout, KInt xsize, KInt range) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt dest = idx * range;
        KInt start = dest;
        KInt end = MIN(dest + range, xsize);
        KInt step = range / ADD_RANGE;

        KFloat sum = 0;
        for (KInt n = start; n < end; n += step) {
            sum = __fadd_rn(sum, xout[n]);
        }
        xout[dest] = sum;
    }
}

__global__ void kai_ker_sum_rows(KInt size, KFloat* xout, KInt xsize, KInt range, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / ncol;
        KInt nc = idx % ncol;

        KInt dest = nr * range * ncol + nc;
        KInt start = dest;
        KInt end = MIN(dest + range * ncol, xsize * ncol);
        KInt step = (range / ADD_RANGE) * ncol;

        KFloat sum = 0;

        for (KInt n = start; n < end; n += step) {
            sum = __fadd_rn(sum, xout[n]);
        }
        
        xout[dest] = sum;
    }
}

/*****************************************************************************
       dataset builtin postproc kernels
*****************************************************************************/
__global__ void kai_ker_mse(KInt size, KFloat* y, KFloat* est, KFloat* ans) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat diff = __fsub_rn(est[idx], ans[idx]);
        y[idx] = __fmul_rn(diff, diff);
    }
}

__global__ void kai_ker_sigmoid_cross_entropy_with_logits(KInt size, KFloat* y, KFloat* est, KFloat* ans) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = dev_sigmoid_cross_entropy_with_logits(est[idx], ans[idx]);
    }
}

__global__ void kai_ker_softmax_cross_entropy_with_logits_1st(KInt size, KFloat* y, KFloat* est, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* logits = est + idx * nvec;

        KFloat max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KFloat prob_term = __fdiv_rn(__expf(__fsub_rn(logits[0], max_term)), sum_exp);
        KFloat log_term = __logf(__fadd_rn(prob_term, 1.0e-10f));
        KFloat entropy = log_term;

        y[idx] = -entropy;
    }
}

__global__ void kai_ker_mult_diff_coef(KInt size, KFloat* y, KFloat* est, KFloat* ans, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = __fmul_rn(__fsub_rn(est[idx], ans[idx]), coef);
    }
}

__global__ void kai_ker_bin_acc(KInt size, KFloat* y, KFloat* est_logit, KFloat* ans_prob) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        bool estimate = est_logit[idx] > 0;
        bool answer = ans_prob[idx] > 0.5;
        y[idx] = (estimate == answer) ? 1.0f : 0;
    }
}

__global__ void kai_ker_class_acc(KInt size, KFloat* y, KFloat* est_logit, KFloat* ans_probs, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt pos = idx * nvec;
        
        KFloat est_max = est_logit[pos];
        KFloat ans_max = ans_probs[pos];

        KInt est_arg = pos;
        KInt ans_arg = pos++;

        for (KInt n = 1; n < nvec; n++, pos++) {
            if (est_max < est_logit[pos]) est_max = est_logit[pos], est_arg = pos;
            if (ans_max < ans_probs[pos]) ans_max = ans_probs[pos], ans_arg = pos;
        }

        y[idx] = (est_arg == ans_arg) ? 1.0f : 0;
    }
}

__global__ void kai_ker_class_idx_acc(KInt size, KFloat* y, KFloat* est_logit, KInt* ans_probs, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt pos = idx * nvec;

        KInt ans_pos = pos + (KInt)ans_probs[idx];

        KFloat est_max = est_logit[pos];
        KInt est_pos = pos++;

        for (KInt n = 1; n < nvec; n++, pos++) {
            if (est_max < est_logit[pos]) est_max = est_logit[pos], est_pos = pos;
        }

        y[idx] = (est_pos == ans_pos) ? 1.0f : 0;
    }
}

__global__ void kai_ker_class_1st_acc(KInt size, KFloat* y, KFloat* est_logit, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt pos = idx * nvec;

        KFloat est_max = est_logit[pos];
        KInt est_arg = pos++;

        for (KInt n = 1; n < nvec; n++, pos++) {
            if (est_max < est_logit[pos]) est_max = est_logit[pos], est_arg = pos;
        }

        y[idx] = (est_arg == 0) ? 1.0f : 0;
    }
}

__global__ void kai_ker_mse_diff_sq(KInt size, KFloat* y, KFloat* x1, KFloat* x2) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat diff = __fsub_rn(x1[idx] , x2[idx]);
        y[idx] =__fmul_rn( diff, diff);
    }
}

/*****************************************************************************
       affine kernels
*****************************************************************************/
__global__ void kai_ker_matmul(KInt size, KFloat* a, KFloat* h, KFloat* w, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(nrow * ncol == size);

        KInt nr = idx / ncol;
        KInt nc = idx % ncol;

        KInt xpos = nr * nvec;
        KInt wpos = nc;

        KFloat sum = 0;

        for (KInt nv = 0; nv < nvec; nv++) {
            sum = __fadd_rn(sum, __fmul_rn(h[xpos], w[wpos]));
            xpos++, wpos += ncol;
        }

        a[idx] = sum;
    }
}

__global__ void kai_ker_multi_matmul(KInt size, KFloat* a, KFloat* h, KFloat* w, KInt mb_size, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(mb_size * nrow * ncol == size);

        KInt nd = idx / (nrow * ncol);
        KInt nr = (idx / ncol) % nrow;
        KInt nc = idx % ncol;

        KInt xpos = nd * (nrow * nvec) + nr * nvec;
        KInt wpos = nd * (nvec * ncol) + nc;

        KFloat sum = 0;

        for (KInt nv = 0; nv < nvec; nv++) {
            sum = __fadd_rn(sum, __fmul_rn(h[xpos], w[wpos]));
            xpos++, wpos += ncol;
        }

        a[idx] = sum;
    }
}

__global__ void kai_ker_add_bias(KInt size, KFloat* aout, KFloat* b, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nc = idx % ncol;
        aout[idx] = __fadd_rn(aout[idx], b[nc]);
    }
}

__global__ void kai_ker_matmul_derv_x(KInt size, KFloat* gx, KFloat* gy, KFloat* w, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(nrow * nvec == size);

        KInt nr = idx / nvec;
        KInt nv = idx % nvec;

        KInt wpos = nv * ncol;
        KInt ypos = nr * ncol;

        KFloat sum = 0;

        for (KInt n = 0; n < ncol; n++) {
            sum = __fadd_rn(sum, __fmul_rn(w[wpos++], gy[ypos++]));
        }

        gx[idx] = sum;
    }
}

__global__ void kai_ker_multi_matmul_derv_x(KInt size, KFloat* gx, KFloat* gy, KFloat* w, KInt mb_size, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(mb_size * nrow * nvec == size);

        KInt nd = idx / (nrow * nvec);
        KInt nr = (idx / nvec) % nrow;
        KInt nv = idx % nvec;

        KInt wpos = nd * (nvec * ncol) + nv * ncol;
        KInt ypos = nd * (nrow * ncol) + nr * ncol;

        KFloat sum = 0;

        for (KInt n = 0; n < ncol; n++) {
            sum = __fadd_rn(sum, __fmul_rn(w[wpos++], gy[ypos++]));
        }

        gx[idx] = sum;
    }
}

__global__ void kai_ker_matmul_derv_w(KInt size, KFloat* gw, KFloat* gy, KFloat* x, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(nvec * ncol == size);

        KInt nv = idx / ncol;  // 여기에 nd 관련 내용 추가
        KInt nc = idx % ncol;

        KInt xpos = nv;
        KInt ypos = nc;

        KFloat wsum = 0;

        for (KInt n = 0; n < nrow; n++) {
            wsum = __fadd_rn(wsum, __fmul_rn(x[xpos], gy[ypos]));
            xpos += nvec, ypos += ncol;
        }

        gw[idx] = wsum;
    }
}

__global__ void kai_ker_multi_matmul_derv_w(KInt size, KFloat* gw, KFloat* gy, KFloat* x, KInt mb_size, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(mb_size * nvec * ncol == size);

        KInt nd = idx / (nvec * ncol);
        KInt nv = (idx / ncol) % nvec;
        KInt nc = idx % ncol;

        KInt xpos = nd * nrow * nvec + nv;
        KInt ypos = nd * nrow * ncol + nc;

        KFloat wsum = 0;

        for (KInt n = 0; n < nrow; n++) {
            wsum = __fadd_rn(wsum, __fmul_rn(x[xpos], gy[ypos]));
            xpos += nvec, ypos += ncol;
        }

        gw[idx] = wsum;
    }
}

__global__ void kai_ker_add_bias_derv(KInt size, KFloat* gb, KFloat* gy, KInt nrow, KInt nvec, KInt ncol) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ypos = idx;
        KFloat bsum = 0;

        for (KInt n = 0; n < nrow; n++) {
            bsum = __fadd_rn(bsum, gy[ypos]);
            ypos += ncol;
        }

        gb[idx] = bsum;
    }
}

__global__ void kai_ker_update_affine_param(KInt size, KFloat* gpm, KFloat* wpout, KFloat* bpout, KInt nrows, KInt ncols) {
    assert(0);
}
;
/*****************************************************************************
       parameter update kernels
*****************************************************************************/
__global__ void kai_ker_update_param_sgd(KInt size, KFloat* pmout, KFloat* gpm, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KFloat delta = gpm[idx];

        if (l2_decay > 0) delta = __fmaf_rn(pmout[idx], l2_decay, delta);
        if (l1_decay > 0) {
            if (pmout[idx] >= -l1_decay && pmout[idx] <= l1_decay) {
                pmout[idx] = 0;
                return;
            }
            else if (pmout[idx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pmout[idx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pmout[idx] = __fsub_rn(pmout[idx], __fmul_rn(delta, learning_rate));
    }
}

__global__ void kai_ker_update_param_adam(KInt size, KFloat* pmout, KFloat* sout, KFloat* tout, KFloat* gpm, KFloat ro1, KFloat ro2, KInt nstep, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KFloat delta = gpm[idx];

        sout[idx] = __fadd_rn(__fmul_rn(sout[idx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
        tout[idx] = __fadd_rn(__fmul_rn(tout[idx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

        KFloat sm = __fdiv_rn(sout[idx], __fsub_rn(1.0f, __powf(ro1, (KFloat)nstep)));
        KFloat tm = __fdiv_rn(tout[idx], __fsub_rn(1.0f, __powf(ro2, (KFloat)nstep)));

        delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

        if (l2_decay) delta += pmout[idx] * l2_decay;
        if (l1_decay) {
            if (pmout[idx] >= -l1_decay && pmout[idx] <= l1_decay) {
                pmout[idx] = 0;
                return;
            }
            else if (pmout[idx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pmout[idx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pmout[idx] = __fsub_rn(pmout[idx], __fmul_rn(delta, learning_rate));
    }
}

__global__ void kai_ker_update_param_sgd_select(KInt size, KFloat* pmout, KInt* wid, KFloat* gpm, KInt word_cnt, KInt vec_size, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) {
    KInt nw = idx / vec_size;
    KInt nv = idx % vec_size;

    KInt pmidx = (KInt)wid[nw] * vec_size + nv;

    KFloat delta = gpm[idx];

    if (l2_decay > 0) delta = __fmaf_rn(pmout[pmidx], l2_decay, delta);
    if (l1_decay > 0) {
        if (pmout[pmidx] >= -l1_decay && pmout[pmidx] <= l1_decay) {
            pmout[pmidx] = 0;
            return;
        }
        else if (pmout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
        else if (pmout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
    }

    pmout[pmidx] = __fsub_rn(pmout[pmidx], __fmul_rn(delta, learning_rate));
}
}

__global__ void kai_ker_update_param_adam_select(KInt size, KFloat* pmout, KFloat* sout, KFloat* tout, KFloat* nout, KInt* wid, KFloat* gpm, KInt word_cnt, KInt vec_size, KFloat ro1, KFloat ro2, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nw = idx / vec_size;
        KInt nv = idx % vec_size;

        for (KInt n = 0; n < word_cnt; n++) {
            if (wid[n] != nw) continue;

            KInt pmidx = (KInt)wid[n] * vec_size + nv;

            KFloat delta = gpm[idx];

            sout[idx] = __fadd_rn(__fmul_rn(sout[idx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
            tout[idx] = __fadd_rn(__fmul_rn(tout[idx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

            KFloat step = nout[nw] = __fadd_rn(nout[nw], 1.0f);

            KFloat sm = __fdiv_rn(sout[idx], __fsub_rn(1.0f, __powf(ro1, step)));
            KFloat tm = __fdiv_rn(tout[idx], __fsub_rn(1.0f, __powf(ro2, step)));

            delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

            if (l2_decay) delta += pmout[pmidx] * l2_decay;
            if (l1_decay) {
                if (pmout[idx] >= -l1_decay && pmout[idx] <= l1_decay) {
                    pmout[idx] = 0;
                    return;
                }
                else if (pmout[idx] > 0) delta = __fadd_rn(delta, l1_decay);
                else if (pmout[idx] < 0) delta = __fsub_rn(delta, l1_decay);
            }

            pmout[idx] = __fsub_rn(pmout[idx], __fmul_rn(delta, learning_rate));
        }
    }
}

__global__ void kai_ker_update_param_sgd_select_multi_dic(KInt size, KFloat* pmout, KInt* wid, KFloat* gpm, KInt dic_count, KInt* voc_counts, KInt vec_size, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nw = idx / (dic_count * vec_size);
        KInt nd = (idx / vec_size) % dic_count;
        KInt nv = idx % vec_size;

        KInt gpmidx = nw * vec_size + nv;
        KFloat delta = gpm[gpmidx];

        KInt ndic = (KInt)wid[nw * dic_count + nd];

        for (KInt n = 0; n < nd; n++) ndic += voc_counts[n];

        KInt pmidx = ndic * vec_size + nv;

        if (l2_decay > 0) delta = __fmaf_rn(pmout[pmidx], l2_decay, delta);
        if (l1_decay > 0) {
            if (pmout[pmidx] >= -l1_decay && pmout[pmidx] <= l1_decay) {
                pmout[pmidx] = 0;
                return;
            }
            else if (pmout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pmout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pmout[pmidx] = __fsub_rn(pmout[pmidx], __fmul_rn(delta, learning_rate));
    }
}

__global__ void kai_ker_update_param_dup_count(KInt size, KFloat* delta, KFloat* count, KInt* wid, KFloat* gpm, KInt dic_count, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nw = idx / vec_size;
        KInt nv = idx % vec_size;

        //KInt nt = nw / dic_count;
        KInt nd = nw % dic_count;

        delta[idx] = 0;
        if (nv == 0) count[nw] = -1;

        KInt wid = (KInt)wid[nw];

        //if (wid == 0 && nd_batch == 0) return;

        KInt count = 0;
        KInt terms_batch = size / vec_size;

        for (KInt n = nd; n < terms_batch; n += dic_count) {
            KInt wid_nom = (KInt)wid[n];
            if (wid_nom != wid) continue;
            if (n < nw) return;

            KInt pmidx = (n / dic_count) * vec_size + nv;

            delta[idx] += gpm[pmidx];
            count++;
        }
        if (nv == 0) count[nw] = (KFloat)count;
    }
}

__global__ void kai_ker_update_param_adam_select_multi_dic(KInt size, KFloat* pmout, KFloat* sout, KFloat* tout, KFloat* nout, KInt* wid, KFloat* delta, KFloat* count, KInt dic_count, KInt* voc_counts, KInt vec_size, KFloat ro1, KFloat ro2, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        KInt nw = idx / vec_size;
        KInt nv = idx % vec_size;

        //KInt nt = nw / dic_count;
        KInt nd = nw % dic_count;

        if (count[nw] <= 0) return;

        KInt wid = (KInt)wid[nw];

        //if (wid == 0 && nd == 0) return;

        KInt wid_dics = wid;
        for (KInt n = 0; n < nd; n++) wid_dics += voc_counts[n];

        KInt pmidx = wid_dics * vec_size + nv;

        KFloat delta = delta[idx];

        sout[pmidx] = __fadd_rn(__fmul_rn(sout[pmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
        tout[pmidx] = __fadd_rn(__fmul_rn(tout[pmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

        KFloat step = __fadd_rn(nout[wid_dics], 1.0f);
        if (nv == 0) nout[wid_dics] = step;

        KFloat sm = __fdiv_rn(sout[pmidx], __fsub_rn(1.0f, __powf(ro1, step)));
        KFloat tm = __fdiv_rn(tout[pmidx], __fsub_rn(1.0f, __powf(ro2, step)));

        delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

        if (l2_decay) delta += pmout[pmidx] * l2_decay;
        if (l1_decay) {
            if (pmout[pmidx] >= -l1_decay && pmout[pmidx] <= l1_decay) {
                pmout[pmidx] = 0;
                return;
            }
            else if (pmout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pmout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pmout[pmidx] = __fsub_rn(pmout[pmidx], __fmul_rn(delta, learning_rate));
        /*
        KInt terms_batch = size / vec_size;

        for (KInt n = nd_batch; n < terms_batch; n += dic_count) {
            KInt wid_nom = (KInt) wid[n];
            if (wid_nom != wid) continue;
            if (n < wid_pos) return;

            KInt ipmidx = n * vec_size + nv_batch;

            KFloat delta = gpm[ipmidx];

            sout[dpmidx] = __fadd_rn(__fmul_rn(sout[dpmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
            tout[dpmidx] = __fadd_rn(__fmul_rn(tout[dpmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

            KFloat step = __fadd_rn(nout[wid_dics], 1.0f);
            if (nv_batch == 0) nout[wid_dics] = step;

            //sout[dpmidx] = delta; // __fadd_rn(__fmul_rn(sout[dpmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
            tout[dpmidx] = __fadd_rn(__fmul_rn(tout[dpmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

            KFloat step = __fadd_rn(nout[wid_dics], 1.0f);
            if (nv_batch == 0) nout[wid_dics] = step;

            KFloat sm = __fdiv_rn(sout[dpmidx], __fsub_rn(1.0f, __powf(ro1, step)));
            KFloat tm = __fdiv_rn(tout[dpmidx], __fsub_rn(1.0f, __powf(ro2, step)));

            delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

            if (l2_decay) delta += pmout[dpmidx] * l2_decay;
            if (l1_decay) {
                if (pmout[dpmidx] >= -l1_decay && pmout[dpmidx] <= l1_decay) {
                    pmout[dpmidx] = 0;
                    return;
                }
                else if (pmout[dpmidx] > 0) delta = __fadd_rn(delta, l1_decay);
                else if (pmout[dpmidx] < 0) delta = __fsub_rn(delta, l1_decay);
            }

            pmout[dpmidx] = __fsub_rn(pmout[dpmidx], __fmul_rn(delta, learning_rate));
        }
        */

        /*
        sout[pmidx] = __fadd_rn(__fmul_rn(sout[pmidx], ro1), __fmul_rn(__fsub_rn(1.0f, ro1), delta));
        tout[pmidx] = __fadd_rn(__fmul_rn(tout[pmidx], ro2), __fmul_rn(__fsub_rn(1.0f, ro2), __fmul_rn(delta, delta)));

        KFloat step = nout[nw] = __fadd_rn(nout[nw], 1.0f);

        KFloat sm = __fdiv_rn(sout[pmidx], __fsub_rn(1.0f, __powf(ro1, step)));
        KFloat tm = __fdiv_rn(tout[pmidx], __fsub_rn(1.0f, __powf(ro2, step)));

        delta = __fdiv_rn(sm, __fadd_rn(__fsqrt_rn(tm), epsilon));

        if (l2_decay) delta += pmout[pmidx] * l2_decay;
        if (l1_decay) {
            if (pmout[pmidx] >= -l1_decay && pmout[pmidx] <= l1_decay) {
                pmout[pmidx] = 0;
                return;
            }
            else if (pmout[pmidx] > 0) delta = __fadd_rn(delta, l1_decay);
            else if (pmout[pmidx] < 0) delta = __fsub_rn(delta, l1_decay);
        }

        pmout[pmidx] = __fsub_rn(pmout[pmidx], __fmul_rn(delta, learning_rate));
        */
    }
}

// depreciated
__global__
void kai_ker_param_update(KInt size, KFloat* p, KFloat* g, KFloat* s, KFloat* t,
    KFloat ro1, KFloat ro2, KFloat ro1_pow, KFloat ro2_pow, KFloat epsilon, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay)
{
    KInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        KFloat delta = g[i];

        if (s != NULL) {
            s[i] = s[i] * ro1 + (1.0f - ro1) * delta;
            t[i] = t[i] * ro2 + (1.0f - ro2) * delta * delta;

            KFloat sm = s[i] / (1.0f - ro1_pow);
            KFloat tm = t[i] / (1.0f - ro2_pow);

            delta = sm / ((KFloat)::__fsqrt_rn(tm) + epsilon);
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
__global__ void kai_ker_sigmoid(KInt size, KFloat* y, KFloat* x) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = dev_sigmoid(x[idx]);
    }
}

__global__ void kai_ker_sigmoid_cross_entropy_with_logits_derv(KInt size, KFloat* y, KFloat* est_logit, KFloat* ans_prob, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat derv = dev_sigmoid_cross_entropy_with_logits_derv(est_logit[idx], ans_prob[idx]);
        y[idx] = __fmul_rn(derv, coef);
        /*
        KFloat x = est_logit[idx];
        KFloat term1 = (x > 0) ? 1 : __expf(x);
        KFloat term2 = __fadd_rn(1.0f, __expf((x > 0) ? -x : x));
        y[idx] = __fmul_rn(__fsub_rn(__fdiv_rn(term1, term2), ans_prob[idx]), coef);
        */
    }
}

/*****************************************************************************
       softmax kernels
*****************************************************************************/
__global__ void kai_ker_softmax(KInt size, KFloat* y, KFloat* x, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nth = idx % nvec;
        KInt pos = idx - nth;

        KFloat* logits = x + pos;
        KFloat max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KFloat idx_term = __expf(__fsub_rn(logits[nth], max_term));
        y[idx] = __fdiv_rn(idx_term, sum_exp);

        /*
        KInt pos = idx - idx % nvec;

        KFloat* xp = x + pos;

        KFloat max_term = xp[0];
        KFloat sum_exp = 0;
        KFloat idx_term = 0;

        for (KInt n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (KInt n = 0; n < nvec; n++) {
            KFloat exp_term = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, exp_term);
            if (pos + n == idx) idx_term = exp_term;
        }

        y[idx] = __fdiv_rn(idx_term, sum_exp);
        */
    }
}

/*
__global__ void kai_ker_softmax(KInt size, KFloat* y, KFloat* x, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat* xp = x + idx * nvec;
        KFloat* yp = y + idx * nvec;

        KFloat max_term = xp[0];
        KFloat sum_exp = 0;

        for (KInt n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (KInt n = 0; n < nvec; n++) {
            yp[n] = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, yp[n]);
        }

        for (KInt n = 0; n < nvec; n++) {
            yp[n] = __fdiv_rn(yp[n], sum_exp);
        }
    }
}
*/

__global__ void kai_ker_softmax_cross_entropy_with_logits_derv(KInt size, KFloat* y, KFloat* est_logit, KFloat* ans_probs, KInt nvec, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nth = idx % nvec;
        KInt pos = idx - nth;

        KFloat* logits = est_logit + pos;
        KFloat max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KFloat idx_term = __expf(__fsub_rn(logits[nth], max_term));
        KFloat prob_term = __fdiv_rn(idx_term, sum_exp);

        y[idx] = __fmul_rn(__fsub_rn(prob_term, ans_probs[idx]), coef);
    }
}

__global__ void kai_ker_softmax_cross_entropy_with_logits_idx_derv(KInt size, KFloat* y, KFloat* est_logit, KInt* ans_probs, KInt nvec, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt wid = idx / nvec;
        KInt nth = idx % nvec;
        KInt pos = idx - nth;

        KFloat* logits = est_logit + pos;
        KFloat max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KFloat idx_term = __expf(__fsub_rn(logits[nth], max_term));
        KFloat prob_term = __fdiv_rn(idx_term, sum_exp);

        KFloat ans = (ans_probs[wid] == nth) ? 1.0f : 0;

        y[idx] = __fmul_rn(__fsub_rn(prob_term, ans), coef);

        /*
        KInt nw = idx / nvec;
        KInt nv = idx % nvec;

        KFloat* xp = est_logit + idx - nv;

        KFloat max_term = xp[0];
        KFloat sum_exp = 0;

        for (KInt n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (KInt n = 0; n < nvec; n++) {
            KFloat exp_term = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, exp_term);
            if (n == nv) y[idx] = exp_term;
        }

        KFloat ans = (ans_probs[nw] == nv) ? 1.0f : 0;

        y[idx] = __fmul_rn(__fsub_rn(__fdiv_rn(y[idx], sum_exp), ans), coef);
        */
    }
}

__global__ void kai_ker_softmax_cross_entropy_with_logits_1st_derv(KInt size, KFloat* y, KFloat* est_logit, KInt nvec, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nth = idx % nvec;
        KInt pos = idx - nth;

        KFloat* logits = est_logit + pos;
        KFloat max_term, sum_exp;

        dev_get_max_sum_for_softmax(logits, nvec, &max_term, &sum_exp);

        KFloat idx_term = __expf(__fsub_rn(logits[nth], max_term));
        KFloat prob_term = __fdiv_rn(idx_term, sum_exp);

        KFloat ans = (nth == 0) ? 1.0f : 0;

        y[idx] = __fmul_rn(__fsub_rn(prob_term, ans), coef);

        /*
        KInt nv = idx % nvec;

        KFloat* xp = est_logit + idx - nv;

        KFloat max_term = xp[0];
        KFloat sum_exp = 0;

        for (KInt n = 1; n < nvec; n++) {
            if (xp[n] > max_term) max_term = xp[n];
        }

        for (KInt n = 0; n < nvec; n++) {
            KFloat exp_term = __expf(__fsub_rn(xp[n], max_term));
            sum_exp = __fadd_rn(sum_exp, exp_term);
            if (n == nv) y[idx] = exp_term;
        }

        KFloat ans = (nv == 0) ? 1.0f : 0;

        y[idx] = __fmul_rn(__fsub_rn(__fdiv_rn(y[idx], sum_exp), ans), coef);
        */
    }
}

/*****************************************************************************
       dropout kernels
*****************************************************************************/
/*****************************************************************************
       activate function kernels
*****************************************************************************/
__global__ void kai_ker_activate(KInt size, KFloat* y, KFloat* x, KInt nFunc, KFloat alpha) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat x = x[idx];
        KFloat y = x;

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
                KFloat x3 = __fmul_rn(x, __fmul_rn(x, x));
                KFloat xt = __fmul_rn(2.0f, __fadd_rn(__fmul_rn(x, 0.797885f), __fmul_rn(0.035677f, x3)));
                KFloat term1 = (xt > 0) ? 1 : __expf(xt);
                KFloat term2 = __fadd_rn(1.0f, __expf((xt > 0) ? -xt : xt));
                KFloat tanh = __fsub_rn(__fmul_rn(__fdiv_rn(term1, term2), 2.0f), 1.0f);
                KFloat coef = __fadd_rn(tanh, 1.0f);
                y = __fmul_rn(__fmul_rn(x, 0.5f), coef);
            }
            break;
        default:
            break;
        }

        y[idx] = y;
    }
}

__global__ void kai_ker_activate_derv(KInt size, KFloat* gx, KFloat* gy, KFloat* x, KFloat* y, KInt nFunc, KFloat alpha) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat x = x[idx];
        KFloat y = y[idx];
        KFloat gy = gy[idx];
        KFloat gx = 1;

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
                KFloat x3 = __fmul_rn(x, __fmul_rn(x, x));
                KFloat tan_x = __fadd_rn(__fmul_rn(x, 0.797885f), __fmul_rn(x3, 0.035677f));
                KFloat sig_x = __fmul_rn(2.0f, tan_x);
                KFloat term1 = (sig_x > 0) ? 1 : __expf(sig_x);
                KFloat term2 = __fadd_rn(1.0f, __expf((sig_x > 0) ? -sig_x : sig_x));

                KFloat gelu_a = __fsub_rn(__fmul_rn(__fdiv_rn(term1, term2), 2.0f), 1.0f);
                KFloat gelu_b = __fadd_rn(__fmul_rn(__fmul_rn(x, x), 0.071356f), 0.797885f);

                gx = __fmul_rn(__fmul_rn(__fadd_rn(gelu_a, 1.0f), -0.5f), __fadd_rn(__fmul_rn(__fmul_rn(x, __fsub_rn(gelu_a, 1.0f)), gelu_b), 1.0f));
            }
            break;
        default:
            break;
        }

        gx[idx] = __fmul_rn(gx, gy);
    }
}

/*****************************************************************************
       convolution kernels
*****************************************************************************/

__global__ void kai_ker_conv_derv_kb_sum1(KInt size, KFloat* d, KFloat* gy, KInt mb_size, KInt xh, KInt xw, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt yn = idx / (mb_size * xh);
        KInt nd = (idx / xh) % mb_size;
        KInt xr = idx % xh;

        KFloat sum = 0;

        KInt ypos = (nd * xh + xr) * xw * ychn + yn;

        for (KInt xc = 0; xc < xw; xc++) {
            sum = __fadd_rn(sum, gy[ypos + xc * ychn]);
        }

        d[idx] = sum;
    }
}

__global__ void kai_ker_conv_derv_kb_sum2(KInt size, KFloat* dout, KInt mb_size, KInt xh, KInt xw, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt yn = idx / mb_size;
        KInt nd = idx % mb_size;

        KFloat sum = 0;

        KInt bpos = (yn * mb_size + nd) * xh;

        for (KInt xr = 0; xr < xh; xr++) {
            sum = __fadd_rn(sum, dout[bpos + xr]);
        }

        dout[bpos] = sum;
    }
}

__global__ void kai_ker_conv_derv_kb_sum3(KInt size, KFloat* gb, KFloat* d, KInt mb_size, KInt xh, KInt xw, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt yn = idx;

        KFloat sum = 0;

        KInt bpos = yn * mb_size * xh;

        for (KInt nd = 0; nd < mb_size; nd++) {
            sum = __fadd_rn(sum, d[bpos + nd * xh]);
        }

        gb[idx] = sum;
    }
}

/*
__global__ void kai_ker_conv_derv_x(KInt size, KFloat* gx, KFloat* c_buf, KFloat* gy, KFloat* k, KInt mb_size, KInt xh, KInt xw, KInt kh, KInt kw, KInt xchn, KInt ychn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (xh * xw * xchn);
        KInt xrow = (idx / (xw * xchn)) % xh;
        KInt xcol = (idx / xchn) % xw;
        KInt xn = idx % xchn;

        KFloat sum = 0;

        KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;
        KInt aidx0 = ndat * xh;

        for (KInt kr = 0; kr < kh; kr++) {
            KInt row = xrow - kr + bh;
            if (row < 0 || row >= xh) continue;
            KInt aidx1 = (aidx0 + row) * xw, kidx1 = kr * kw;
            for (KInt kc = 0; kc < kw; kc++) {
                KInt col = xcol - kc + bw;
                if (col < 0 || col >= xw) continue;
                KInt aidx2 = (aidx1 + col) * ychn, kidx2 = ((kidx1 + kc) * xchn + xn) * ychn;
                for (KInt n = 0; n < ychn; n++) {
                    KInt aidx = aidx2 + n, kidx = kidx2 + n; // (kidx2 + n)* xchn + xn;
                    sum += gyp[aidx] * kp[kidx];
                }
            }
        }

        gxp[xidx] = sum;

        if (idx == 0) assert(0);
        __global__
            void kai_ker_backprop_conv(KInt xsize, KInt ksize, KInt bsize,
                KFloat* gyp, KFloat* xp, KFloat* kp, KFloat* gxp, KFloat* gkp, KFloat* gbp,
                KInt mb_size, KInt xh, KInt xw, KInt xchn, KInt kh, KInt kw, KInt ychn) {
            KInt cidx = blockIdx.x * blockDim.x + threadIdx.x;
            if (cidx < xsize) {
            }
            else if (cidx < xsize + ksize) {
                KInt kidx = cidx - xsize;

                KInt krow = kidx / (kw * xchn * ychn);
                KInt kcol = (kidx / (xchn * ychn)) % kw;
                KInt xn = (kidx / ychn) % xchn;
                KInt yn = kidx % ychn;

                KFloat sum = 0;

                KInt bh = (kh - 1) / 2, bw = (kw - 1) / 2;
                KInt xidx0 = xn, yidx0 = yn;

                for (KInt yrow = 0; yrow < xh; yrow++) {
                    KInt xrow = yrow + krow - bh;
                    if (xrow < 0 || xrow >= xh) continue;
                    KInt xidx1 = xidx0 + xrow * xw * xchn;
                    KInt yidx1 = yidx0 + yrow * xw * ychn;
                    for (KInt ycol = 0; ycol < xw; ycol++) {
                        KInt xcol = ycol + kcol - bw;
                        if (xcol < 0 || xcol >= xw) continue;
                        KInt xidx2 = xidx1 + xcol * xchn;
                        KInt yidx2 = yidx1 + ycol * ychn;
                        for (KInt n = 0; n < mb_size; n++) {
                            KInt xidx = xidx2 + n * xh * xw * xchn;
                            KInt yidx = yidx2 + n * xh * xw * ychn;
                            sum += xp[xidx] * gyp[yidx];
                        }
                    }
                }

                gkp[kidx] = sum;
            }
            else if (cidx < xsize + ksize + bsize) {
                KInt bidx = cidx - xsize - ksize;
                KInt yn = bidx % ychn;
                KInt ysize = mb_size * xh * xw * ychn;

                KFloat sum = 0;

                for (KInt n = yn; n < ysize; n += ychn) {
                    sum += gyp[n];
                }

                gbp[bidx] = sum;
            }
        }
    }
}

__global__ void kai_ker_conv_derv_k(KInt size, KFloat* gk, KFloat* c_buf, KFloat* gy, KFloat* x, KInt mb_size, KInt xh, KInt xw, KInt kh, KInt kw, KInt ychn, bool use_bias) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(0);
    }
}
*/

/*****************************************************************************
       max pool kernels
*****************************************************************************/
/*****************************************************************************
       avg pool kernels
*****************************************************************************/
/*****************************************************************************
       add kernels
*****************************************************************************/
__global__ void kai_ker_avg_exact(KInt size, KFloat* t, KFloat* x, KInt xh, KInt xw, KInt xchn, KInt sh, KInt sw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt th = xh / sh;
        KInt tw = xw / sw;

        KInt ndat = idx / (th * tw * xchn);
        KInt trow = (idx / (tw * xchn)) % th;
        KInt tcol = (idx / xchn) % tw;
        KInt xn = idx % xchn;

        KFloat sum = 0;

        for (KInt r = 0; r < sh; r++) {
            KInt xrow = trow * sh + r;
            for (KInt c = 0; c < sw; c++) {
                KInt xcol = tcol * sw + c;
                KInt xidx = ((ndat * xh + xrow) * xw + xcol) * xchn + xn;
                sum = __fadd_rn(sum, x[xidx]);
            }
        }

        t[idx] = __fdiv_rn(sum, (KFloat)(sh*sw));
    }
}

__global__ void kai_ker_avg_exact_derv(KInt size, KFloat* gt, KFloat* gy, KInt xh, KInt xw, KInt xchn, KInt sh, KInt sw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt yh = xh / sh;
        KInt yw = xw / sw;

        KInt ndat = idx / (xh * xw * xchn);
        KInt xrow = (idx / (xw * xchn)) % xh;
        KInt xcol = (idx / xchn) % xw;
        KInt xn = idx % xchn;

        KInt yrow = xrow / sh;
        KInt ycol = xcol / sh;

        KInt yidx = ((ndat * yh + yrow) * yw + ycol) * xchn + xn;

        gt[idx] = __fdiv_rn(gy[yidx], (KFloat)(sh * sw));
    }
}

/*****************************************************************************
       stride kernels
*****************************************************************************/
__global__ void kai_ker_stride(KInt size, KFloat* y, KFloat* x, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt kh, KInt kw, KInt sh, KInt sw, bool valid_padding) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (yh * yw * chn);
        KInt row = (idx / (yw * chn)) % yh;
        KInt col = (idx / chn) % yw;
        KInt xn = idx % chn;

        KInt bh = (sh - 1) / 2, bw = (sw - 1) / 2;
        if (valid_padding) bh += (kh - 1) / 2, bw += (kw - 1) / 2;

        KInt rpos = row * sh + bh;
        KInt cpos = col * sw + bw;

        KInt xpos = ((ndat * xh + rpos) * xw + cpos) * chn + xn;

        y[idx] = x[xpos];
    }
}

__global__ void kai_ker_stride_derv(KInt size, KFloat* gx, KFloat* gy, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt kh, KInt kw, KInt sh, KInt sw, bool valid_padding) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx[idx] = 0;

        KInt ndat = idx / (xh * xw * chn);
        KInt row = (idx / (xw * chn)) % xh;
        KInt col = (idx / chn) % xw;
        KInt xn = idx % chn;

        KInt bh = (sh - 1) / 2, bw = (sw - 1) / 2;

        if (valid_padding) {
            bh += (kh - 1) / 2, bw += (kw - 1) / 2;
        }

        if ((row - bh) % sh != 0) return;
        if ((col - bw) % sw != 0) return;

        KInt rpos = (row - bh) / sh;
        KInt cpos = (col - bw) / sw;

        if (rpos < 0 | rpos >= yh) return;
        if (cpos < 0 | cpos >= yw) return;

        KInt spos = ((ndat * yh + rpos) * yw + cpos) * chn + xn;

        gx[idx] = gy[spos];
    }
}

__global__ void kai_ker_stride_expand(KInt size, KFloat* y, KFloat* x, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ndat = idx / (yh * yw * chn);
        KInt row = (idx / (yw * chn)) % yh;
        KInt col = (idx / chn) % yw;
        KInt xn = idx % chn;

        KInt rpos = row / sh;
        KInt cpos = col / sw;

        KInt xpos = ((ndat * xh + rpos) * xw + cpos) * chn + xn;

        y[idx] = x[xpos];
    }
}

__global__ void kai_ker_stride_expand_derv(KInt size, KFloat* gx, KFloat* gy, KInt xh, KInt xw, KInt yh, KInt yw, KInt chn, KInt sh, KInt sw) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        gx[idx] = 0;

        KInt ndat = idx / (xh * xw * chn);
        KInt row = (idx / (xw * chn)) % xh;
        KInt col = (idx / chn) % xw;
        KInt xn = idx % chn;

        KInt rpos = row * sh;
        KInt cpos = col * sw;

        KInt spos = ((ndat * yh + rpos) * yw + cpos) * chn + xn;

        gx[idx] = 0;

        for (KInt n = 0; n < sh; n++) {
            for (KInt m = 0; m < sw; m++) {
                gx[idx] += gy[spos + (n * yw + m) * chn];
            }
        }
    }
}

/*****************************************************************************
       batch normal kernels
*****************************************************************************/
__global__ void kai_ker_batch_normal(KInt size, KFloat* y, KFloat* x, KInt ngroup) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(0);
    }
}

__global__ void kai_ker_batch_normal_derv(KInt size, KFloat* gx, KFloat* gy, KFloat* y, KInt ngroup) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx == 0) assert(0);
    }
}

__global__ void kai_ker_bn_collect(KInt size, KFloat* avgout, KFloat* varout, KFloat* mavgout, KFloat* mvarout, KFloat* x, KInt hsize, KFloat momentum) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nrest = hsize / size;

        KFloat sum = 0, sqsum = 0;

        for (KInt n = idx; n < hsize; n+=size) {
            KFloat x = x[n];
            sum += x;
            sqsum += x * x;
        }

        KFloat avg = sum / (KFloat) nrest;
        KFloat var = sqsum / (KFloat)nrest - avg * avg;

        avgout[idx] = avg;
        varout[idx] = var;

        mavgout[idx] = mavgout[idx] * momentum + avg * (1 - momentum);
        mvarout[idx] = mvarout[idx] * momentum + var * (1 - momentum);
    }
}

__global__ void kai_ker_bn_normalize(KInt size, KFloat* hout, KFloat* avg, KFloat* var, KInt bsize, KFloat epsilon) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt bidx = idx % bsize;
        KFloat std = __fsqrt_rn(__fadd_rn(var[bidx], epsilon));

        hout[idx] = __fdiv_rn(__fsub_rn(hout[idx], avg[bidx]), std);
    }
}

__global__ void kai_ker_bn_rescale(KInt size, KFloat* hout, KFloat* scale, KFloat* shift, KInt bsize) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt bidx = idx % bsize;

        hout[idx] = __fmaf_rn(hout[idx], scale[bidx], shift[bidx]);
    }
}

/*****************************************************************************
       dropout kernels
*****************************************************************************/
/*****************************************************************************
       parallel layer
*****************************************************************************/
/*****************************************************************************
       rnn/lstm layer
*****************************************************************************/
__global__ void kai_ker_rnn_combine_exp(KInt size, KFloat* exp, KFloat* x, KFloat* rec, KInt timesteps, KInt timefeats, KInt recur_size, bool inseq, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / (timefeats + recur_size);
        KInt col = idx % (timefeats + recur_size);

        if (col < timefeats) {
            if (inseq) 
                exp[idx] = x[((row * timesteps) + tn) * timefeats + col];
            else
                exp[idx] = x[row * timefeats + col];
        }
        else {
            exp[idx] = rec[row * recur_size + (col - timefeats)];
        }
    }
}

__global__ void kai_ker_rnn_split_exp(KInt size, KFloat* gx, KFloat* grec, KFloat* gexp, KInt timesteps, KInt timefeats, KInt recur_size, bool inseq, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / (timefeats + recur_size);
        KInt col = idx % (timefeats + recur_size);

        if (col < timefeats) {
            if (inseq) {
                KInt xidx = ((row * timesteps) + tn) * timefeats + col;
                gx[xidx] = gexp[idx];
            }
            else {
                KInt xidx = row * timefeats + col;
                gx[xidx] = __fadd_rn(gx[xidx], gexp[idx]);
            }
        }
        else {
            KInt ridx = row * recur_size + (col - timefeats);
            grec[ridx] = gexp[idx];
        }
    }
}

__global__ void kai_ker_rnn_fillput_slice(KInt size, KFloat* y, KFloat* x, KInt timesteps, KInt recur_size, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / recur_size;
        KInt col = idx % recur_size;

        y[((row * timesteps) + tn) * recur_size + col] = x[idx];
    }
}

__global__ void kai_ker_rnn_add_time_slice(KInt size, KFloat* gy, KFloat* gx, KInt timesteps, KInt recur_size, KInt tn) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / recur_size;
        KInt col = idx % recur_size;

        gy[idx] = __fadd_rn(gy[idx], gx[((row * timesteps) + tn) * recur_size + col]);
    }
}

__global__ void kai_ker_rnn_copy_last_grad(KInt size, KFloat* grecout, KFloat* gx, KInt timesteps, KInt recur_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        //KInt row = idx / recur_size;

        grecout[idx] = gx[idx];
    }
}

__global__ void kai_ker_lstm_split_affine(KInt size, KFloat* fgate, KFloat* igate, KFloat* ogate, KFloat* block, KFloat* affine, KInt recur_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / (4 * recur_size);
        KInt col = idx % (4 * recur_size);
        KInt slice_idx = row * recur_size + col % recur_size;

        if (col < recur_size)
            fgate[slice_idx] = affine[idx];
        else if (col < 2 * recur_size)
            igate[slice_idx] = affine[idx];
        else if (col < 3 * recur_size)
            ogate[slice_idx] = affine[idx];
        else if (col < 4 * recur_size)
            block[slice_idx] = affine[idx];
    }
}

__global__ void kai_ker_lstm_combine_affine(KInt size, KFloat* gaffine, KFloat* gfgate, KFloat* gigate, KFloat* gogate, KFloat* gblock, KInt recur_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / (4 * recur_size);
        KInt col = idx % (4 * recur_size);
        KInt slice_idx = row * recur_size + col % recur_size;

        if (col < recur_size)
            gaffine[idx] = gfgate[slice_idx];
        else if (col < 2 * recur_size)
            gaffine[idx] = gigate[slice_idx];
        else if (col < 3 * recur_size)
            gaffine[idx] = gogate[slice_idx];
        else if (col < 4 * recur_size)
            gaffine[idx] = gblock[slice_idx];

        /*
        if (col < recur_size)
            gaffine[idx] = gfgate[slice_idx];
        else if (col < 2 * recur_size)
            gaffine[idx] = gigate[slice_idx];
        else if (col < 3 * recur_size)
            gaffine[idx] = gogate[slice_idx];
        else if (col < 4 * recur_size)
            gaffine[idx] = gblock[slice_idx];
        */
    }
}

__global__ void kai_ker_rnn_select_last_vecs(KInt size, KFloat* selected, KFloat* time_pool, KInt timesteps, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt row = idx / nvec;
        KInt col = idx % nvec;

        KInt tn = timesteps - 1;
        KInt pidx = (row * timesteps + tn) * nvec + col;

        selected[idx] = time_pool[pidx];
    }
}

__global__ void kai_ker_lstm_new_state(KInt size, KFloat* state, KFloat* state, KFloat* fgate, KFloat* block, KFloat* igate) { // state = state * forget_gate + blockput * input_gate;
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        state[idx] = __fadd_rn(__fmul_rn(state[idx], fgate[idx]), __fmul_rn(block[idx], igate[idx]));
    }
}

__global__ void kai_ker_lstm_state_derv(KInt size, KFloat* gstateout, KFloat* grec, KFloat* ogate, KFloat* rec) { // G_recur_tmp = G_recurrent * output_gate; G_state += kmath->tanh_derv(recur_tmp) * G_recur_tmp;
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat grec1 = grec[idx] * ogate[idx];
        KFloat gtanh = __fmul_rn(__fadd_rn(1.0f, rec[idx]), __fsub_rn(1.0f, rec[idx]));

        gstateout[idx] = __fadd_rn(gstateout[idx], __fmul_rn(grec1, gtanh));
    }
}

/*****************************************************************************
       attention layer: forward
*****************************************************************************/
__global__ void kai_ker_attention_split(KInt size, KFloat* q, KFloat* k, KFloat* v, KFloat* qkv, KInt L, KInt H, KInt R) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (L * 3 * H * R);
        KInt nt = (idx / (3 * H * R)) % L;
        KInt n3 = (idx / (H * R)) % 3;
        KInt nh = (idx / R) % H;
        KInt nr = idx % R;

        if (n3 == 0) {
            KInt qpos = ((nd * H + nh)* L + nt)* R + nr;
            q[qpos] = qkv[idx];
        }
        else if (n3 == 1) {
            KInt kpos = ((nd * H + nh) * R + nr) * L + nt;
            k[kpos] = qkv[idx];
        }
        else {
            KInt vpos = ((nd * H + nh) * L + nt) * R + nr;
            v[vpos] = qkv[idx];
        }
    }
}

__global__ void kai_ker_attention_combine(KInt size, KFloat* gqkv, KFloat* gq, KFloat* gk, KFloat* gv, KInt L, KInt H, KInt R) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (L * 3 * H * R);
        KInt nt = (idx / (3 * H * R)) % L;
        KInt n3 = (idx / (H * R)) % 3;
        KInt nh = (idx / R) % H;
        KInt nr = idx % R;

        if (n3 == 0) {
            KInt qpos = ((nd * H + nh) * L + nt) * R + nr;
            gqkv[idx] = gq[qpos];
        }
        else if (n3 == 1) {
            KInt kpos = ((nd * H + nh) * R + nr) * L + nt;
            gqkv[idx] = gk[kpos];
        }
        else {
            KInt vpos = ((nd * H + nh) * L + nt) * R + nr;
            gqkv[idx] = gv[vpos];
        }
    }
}

__global__ void kai_ker_attention_mask_future(KInt size, KFloat* scoreout, KInt L) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nt1 = (idx / L) % L;
        KInt nt2 = idx % L;
        
        if (nt2 > nt1) scoreout[idx] -= 10000.0f;
    }
}

__global__ void kai_ker_attention_reshape(KInt size, KFloat* a, KFloat* m, KInt L, KInt H, KInt R) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (L * H * R);
        KInt nt = (idx / (H * R)) % L;
        KInt nh = (idx / R) % H;
        KInt nr = idx % R;
        
        KInt mpos = ((nd * H + nh) * L + nt) * R + nr;

        a[idx] = m[mpos];
    }
}

__global__ void kai_ker_attention_reshape_mul(KInt size, KFloat* gm, KFloat* ga, KInt L, KInt H, KInt R) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (H * L * R);
        KInt nh = (idx / (L * R)) % H;
        KInt nt = (idx / R) % L;
        KInt nr = idx % R;

        KInt apos = ((nd * L + nt) * H + nh) * R + nr;

        gm[idx] = ga[apos];
    }
}

/*
__global__ void kai_ker_attention_forward_mult_val(KInt size, KFloat* att, KFloat* qkv, KFloat* buf, KInt B, KInt L, KInt V, KInt H, KInt R) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (L * V);
        KInt nt = idx / V % L;
        KInt nv = idx % V;

        KInt nh = nv / R;

        KInt apos = nd * (H * L * L) + nh * (L * L) + nt * L;
        KInt vpos = nd * (L * 3 * V) + nv + 2 * V;

        KFloat sum = 0;

        for (KInt nt1 = 0; nt1 < L; nt1++) {
            sum = __fadd_rn(sum, __fmul_rn(att[apos], qkv[vpos]));
            apos++, vpos += 3 * V;
        }

        buf[idx] = sum;
    }
}

__global__ void kai_ker_attention_backprop_matmul_probs(KInt size, KFloat* gy, KFloat* qkv, KFloat* gprobs, KInt B, KInt L, KInt V, KInt H, KInt R) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (H * L * L);
        KInt nh = idx / (L * L) % H;
        KInt nt1 = idx / L % L;
        KInt nt2 = idx % L;

        KInt vpos = nd * (3 * L * V) + nt2 * (3 * V) + nh * R + V + V;
        KInt ypos = nd * (L * V) + nh * R + nt1 * V;

        KFloat gsum = 0;

        for (KInt nr = 0; nr < R; nr++) {
            gsum = __fadd_rn(gsum, __fmul_rn(qkv[vpos], gy[ypos]));
            vpos += 1, ypos += 1;
        }

        gprobs[idx] = gsum;
    }
}

__global__ void kai_ker_attention_backprop_matmul_value(KInt size, KFloat* gy, KFloat* probs, KFloat* gqkv, KInt B, KInt L, KInt V, KInt H, KInt R) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (L * V);
        KInt nh = idx / R % H;
        KInt nt = idx / V % L;
        KInt nr = idx % R;

        KInt ppos = nd * (H * L * L) + nh * (L * L) + nt;
        KInt ypos = nd * (H * L * R) + nh * R + nr;
        KInt gpos = nd * (3 * L * V) + nt * (3 * V) + nh * R + nr + V + V;

        KFloat gsum = 0;

        for (KInt nt2 = 0; nt2 < L; nt2++) {
            gsum = __fadd_rn(gsum, __fmul_rn(probs[ppos], gy[ypos]));
            ppos += L, ypos += V;
        }

        gqkv[gpos] = gsum;
    }
}
*/

/*****************************************************************************
       attention layer: [Q(in qkv)] matmul [K(in qkv)]
*****************************************************************************/
/*
__global__ void kai_ker_attention_forward_qk_mult(KInt size, KFloat* qkv, KFloat* att, KInt B, KInt L, KInt V, KInt H, KInt R, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (H * L * L);
        KInt nh = idx / (L * L) % H;
        KInt nt1 = idx / L % L;
        KInt nt2 = idx % L;

        KInt qpos = nd * (3 * L * V) + nh * R + nt1 * (3 * V);
        KInt kpos = nd * (3 * L * V) + nh * R + nt2 * (3 * V) + V;

        KFloat sum = 0;

        for (KInt nv = 0; nv < R; nv++) {
            sum = __fadd_rn(sum, __fmul_rn(qkv[qpos++], qkv[kpos++]));
        }

        att[idx] = sum * coef;
    }
}

__global__ void kai_ker_attention_backprop_mult_kv(KInt size, KFloat* gscore, KFloat* qkv, KFloat* gqkv, KInt B, KInt L, KInt V, KInt H, KInt R, KFloat coef) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (L * V);
        KInt nh = idx / R % H;
        KInt nt = idx / V % L;
        KInt nr = idx % R;

        KInt qspos = nd * (H * L * L) + nh * (L * L) + nt * L;
        KInt kspos = nd * (H * L * L) + nh * (L * L) + nt;

        KInt qkpos = nd * (L * 3 * V) + nh * R + nr + V;
        KInt kqpos = nd * (L * 3 * V) + nh * R + nr;

        KFloat gqsum = 0, gksum = 0;

        for (KInt n = 0; n < L; n++) {
            gqsum = __fadd_rn(gqsum, __fmul_rn(gscore[qspos], qkv[qkpos]));
            gksum = __fadd_rn(gksum, __fmul_rn(gscore[kspos], qkv[kqpos]));

            qspos++, qkpos += 3 * V;
            kspos += L, kqpos += 3 * V;
        }

        KInt qpos = nd * (L * 3 * V) + nh * R + nt * (3 * V) + nr;
        KInt kpos = nd * (L * 3 * V) + nh * R + nt * (3 * V) + nr + V;

        gqkv[qpos] = __fmul_rn(gqsum, coef);
        gqkv[kpos] = __fmul_rn(gksum, coef);
    }
}
*/

/*****************************************************************************
       embed layer
*****************************************************************************/
__global__ void kai_ker_embedding_fetch(KInt size, KFloat* wiv, KFloat* wov, KInt* hint, KInt* noms, KFloat* iwdic, KFloat* owdic, KInt in_cnt, KInt out_cnt, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt vec_cnt = in_cnt + out_cnt;

        KInt nr = idx / (vec_cnt * vec_size);
        KInt nv = (idx / vec_size) % vec_cnt;
        KInt nc = idx % vec_size;

        if (nv < in_cnt) {
            KInt word_id = (KInt)hint[nr * in_cnt + nv];
            KInt didx = word_id * vec_size + nc;
            KInt widx = (nr * in_cnt + nv) * vec_size + nc;
            wiv[widx] = iwdic[didx];
        }
        else {
            nv -= in_cnt;
            KInt word_id = (KInt)noms[nr * out_cnt + nv];
            KInt didx = word_id * vec_size + nc;
            KInt widx = (nr * out_cnt + nv) * vec_size + nc;
            wov[widx] = owdic[didx];
        }
    }
}

__global__ void kai_ker_embedding_dotmul(KInt size, KFloat* y, KFloat* sub, KFloat* wov, KInt vec_cnt, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / vec_cnt;
        //KInt nc = idx % vec_cnt;

        KInt sidx = nr * vec_size;
        KInt xidx = idx * vec_size;

        KFloat sum = 0;

        for (KInt n = 0; n < vec_size; n++, xidx++) {
            sum = __fadd_rn(sum, __fmul_rn(sub[sidx], wov[xidx]));
        }

        y[idx] = sum;
    }
}

__global__ void kai_ker_embedding_dotmul_derv(KInt size, KFloat* gsub, KFloat* gwov, KFloat* gy, KFloat* sub, KFloat* wov, KInt vec_cnt, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / (vec_cnt * vec_size);
        KInt nv = (idx / vec_size) % vec_cnt;
        KInt nc = idx % vec_size;

        KInt sidx = nr * vec_size + nc;
        KInt yidx = nr * vec_cnt + nv;

        gwov[idx] = __fmul_rn(sub[sidx], gy[yidx]);

        if (nv == 0) {
            KFloat sum = 0;

            for (KInt n = 0; n < vec_cnt; n++) {
                KInt widx = (nr * vec_cnt + n) * vec_size + nc;
                KInt yidx = nr * vec_cnt + n;

                sum = __fadd_rn(sum, __fmul_rn(wov[widx], gy[yidx]));
            }
            KInt gidx = nr * vec_size + nc;
            gsub[gidx] = sum;
        }
    }
}

/*****************************************************************************
       merge layer
*****************************************************************************/
__global__ void kai_ker_merge_avg(KInt size, KFloat* m, KFloat* x, KInt vec_cnt, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / vec_size;
        KInt nc = idx % vec_size;

        KInt xidx = nr * vec_cnt * vec_size + nc;
            
        KFloat sum = 0;
        
        for (KInt n = 0; n < vec_cnt; n++, xidx += vec_size) {
            sum = __fadd_rn(sum, x[xidx]);
        }
        
        m[idx] = __fdiv_rn(sum, (KFloat)vec_cnt);
    }
}

__global__ void kai_ker_merge_avg_derv(KInt size, KFloat* gx, KFloat* gm, KInt vec_cnt, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / (vec_cnt * vec_size);
        KInt nc = idx % vec_size;

        KInt midx = nr * vec_size + nc;

        gx[idx] = __fdiv_rn(gm[midx], (KFloat)vec_cnt);
    }
}

/*****************************************************************************
       extract layer
*****************************************************************************/
__global__ void kai_ker_unextract(KInt size, KFloat* gx, KFloat* ge, KInt ax_size, KInt index, KInt nprod) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / (ax_size * nprod);
        KInt nx = (idx / nprod) % ax_size;
        KInt nc = idx % nprod;

        KInt epos = nr * nprod + nc;

        if (nx == index)
            gx[idx] = ge[epos];
        else
            gx[idx] = 0;
    }
}

/*****************************************************************************
       math
*****************************************************************************/
__global__ void kai_ker_vstack(KInt size, KFloat* vs, KFloat* x1, KFloat* x2, KInt vol1, KInt vol2, KInt rest) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / rest;
        KInt nc = idx % rest;

        if (nr < vol1)
            vs[idx] = x1[nr * rest + nc];
        else
            vs[idx] = x2[(nr - vol1) * rest + nc];
    }
}

__global__ void kai_ker_hstack(KInt size, KFloat* vs, KFloat* x1, KFloat* x2, KInt vec1, KInt vec2) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt vec_size = vec1 + vec2;

        KInt nr = idx / vec_size;
        KInt nc = idx % vec_size;

        if (nc < vec1)
            vs[idx] = x1[nr * vec1 + nc];
        else
            vs[idx] = x2[nr * vec2 + (nc - vec1)];
    }
}

__global__ void kai_ker_hsplit(KInt size, KFloat* p1, KFloat* p2, KFloat* src, KInt vec_size, KInt p1_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nrow = idx / vec_size;
        KInt ncol = idx % vec_size;

        if (ncol < p1_size) {
            KInt p1_idx = nrow * p1_size + ncol;
            p1[p1_idx] = src[idx];
        }
        else {
            KInt p2_size = vec_size - p1_size;
            KInt p2_col = ncol - p1_size;
            KInt p2_idx = nrow * p2_size + p2_col;
            p2[p2_idx] = src[idx];
        }
    }
}

__global__ void kai_ker_embed_fetch_multi_dic(KInt size, KFloat* v, KInt* wids, KFloat* dics, KInt dic_count, KInt* voc_counts, KInt vec_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nw = idx / vec_size;
        KInt nv = idx % vec_size;

        KInt wpos = nw * dic_count;
        KInt wbase = 0;

        KFloat sum = 0;

        for (KInt n = 0; n < dic_count; n++) {
            KInt wid = (KInt) wids[wpos++] + wbase;
            KInt dpos = wid * vec_size + nv;
            wbase += voc_counts[n];
            sum = __fadd_rn(sum, dics[dpos]);
        }

        v[idx] = sum;
    }
}

__global__ void kai_ker_set_row(KInt size, KFloat* pout, KInt nrest, KFloat value) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        pout[idx * nrest] = value;
    }
}

__global__ void kai_ker_extract_selected_pickup(KInt size, KFloat* dst, KFloat* arr, KInt* map, KInt drest) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nrow = idx / drest;
        KInt ncol = idx % drest;

        KInt sidx = map[nrow] * drest + ncol;
        dst[idx] = arr[sidx];
    }
}

__global__ void kai_ker_extract_selected_fill(KInt size, KFloat* arrout, KFloat* slice, KInt* map, KInt drest) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nrow = idx / drest;
        KInt ncol = idx % drest;

        KInt sidx = map[nrow] * drest + ncol;
        arrout[sidx] = slice[idx];
    }
}

__global__ void kai_ker_extract_selected_pickupt(KInt size, KInt* dst, KInt* arr, KInt* map, KInt drest) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nrow = idx / drest;
        KInt ncol = idx % drest;

        KInt sidx = map[nrow] * drest + ncol;
        dst[idx] = arr[sidx];
    }
}

__global__ void kai_ker_expand(KInt size, KFloat* y, KFloat* x, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (heights * widths * chns);
        KInt nh = idx / (widths * chns) % heights;
        KInt nw = idx / chns % widths;
        KInt nc = idx % chns;

        KInt xheights = heights / hratio;
        KInt xwidths = widths / wratio;

        KInt xh = nh / hratio;
        KInt xw = nw / wratio;

        KInt xidx = ((nd * xheights + xh) * xwidths + xw) * chns + nc;

        y[idx] = x[xidx];
    }
}

__global__ void kai_ker_expand_derv(KInt size, KFloat* gx, KFloat* gy, KInt heights, KInt widths, KInt chns, KInt hratio, KInt wratio) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (heights * widths * chns);
        KInt nh = idx / (widths * chns) % heights;
        KInt nw = idx / chns % widths;
        KInt nc = idx % chns;

        KInt yheights = heights * hratio;
        KInt ywidths = widths * wratio;

        KInt yh = nh * hratio;
        KInt yw = nw * wratio;

        KFloat sum = 0;

        KInt yidx_base = ((nd * yheights + yh) * ywidths + yw) * chns + nc;

        for (KInt h = 0; h < hratio; h++) {
            KInt yidx = yidx_base + h * ywidths * chns;
            for (KInt w = 0; w < wratio; w++) {
                sum += gy[yidx + w * chns];
            }
        }

        gx[idx] = sum;
    }
}

__global__ void kai_ker_yolo_eval_true_box_score(KInt size, KFloat* score, KFloat* boxes, KInt* anchors, KInt num_scales, KInt anchor_per_scale) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nb = idx / (num_scales * anchor_per_scale);
        KInt ns = idx / anchor_per_scale % num_scales;
        KInt na = idx % anchor_per_scale;

        KInt aidx = ((ns * anchor_per_scale) + na) * 2;

        KFloat box_width = box_wd(boxes, nb);
        KFloat box_height = box_ht(boxes, nb);

        KFloat anchor_width = (KFloat)anchors[aidx];
        KFloat anchor_height = (KFloat)anchors[aidx + 1];

        KFloat inter_width = (box_width < anchor_width) ? box_width : anchor_width;
        KFloat inter_height = (box_height < anchor_height) ? box_height : anchor_height;

        KFloat box_area = box_width * box_height;
        KFloat anchor_area = anchor_width * anchor_height;
        KFloat inter_area = inter_width * inter_height;

        KFloat union_area = __fsub_rn(__fadd_rn(box_area, anchor_area), inter_area);
        KFloat iou = __fdiv_rn(inter_area, union_area);

        score[idx] = iou;
    }
}

__global__ void kai_ker_yolo_eval_true_box_select(KInt size, KFloat* seletced, KFloat* score, KInt num_scales, KInt anchor_per_scale) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nom_cnt = num_scales * anchor_per_scale;
        KInt nom_idx = (KInt) idx * nom_cnt;
        KInt best_idx = nom_idx;

        for (KInt n = 0; n < nom_cnt; n++) {
            //seletced[nom_idx + n] = score[nom_idx + n] > 0.4f ? 1.0f : 0.0f;
            seletced[nom_idx + n] = 0.0f; // 텐서 버전 대조 결과 가장 나은 경우 하나만 선택
            if (score[nom_idx + n] > score[best_idx]) best_idx = nom_idx + n;
        }

        seletced[best_idx] = 1.0f;
    }
}

__global__ void kai_ker_yolo_eval_true_count_selected(KInt size, KInt* selected_cnt, KFloat* selected, KInt dat_size, KInt num_scales, KInt anchor_per_scale) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        for (KInt n = 0; n < dat_size; n++) {
            if (n / anchor_per_scale % num_scales != idx) continue;
            if (selected[n] <= 0.5f) continue;
            
            selected_cnt[idx]++;
        }
    }
}

__global__ void kai_ker_yolo_eval_true_lookup_scale_box(KInt size, KInt* box_scale_cood, KFloat* selected, KInt nscale, KInt dat_size, KInt num_scales, KInt anchor_per_scale) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        for (KInt n = 0, sidx=0; n < dat_size; n++) {
            //KInt nb = n / (num_scales * anchor_per_scale);
            KInt ns = n / anchor_per_scale % num_scales;
            KInt na = n % anchor_per_scale;

            if (ns != nscale) continue;
            if (selected[n] <= 0.5f) continue;

            KInt nbox = n / (num_scales * anchor_per_scale);

            set_scale_nbox(box_scale_cood, sidx, nbox);
            //set_scale_nbox(box_scale_cood, n, nbox);
            set_scale_nanchor(box_scale_cood, sidx, (KInt)na);

            sidx++;
        }
    }
}

__global__ void kai_ker_yolo_eval_true_eval_box_cood(KInt size, KInt* box_scale_coodout, KFloat* boxfo, KInt grid_size) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt bidx = scale_nbox(box_scale_coodout, idx);

        KFloat center_x = box_cx(boxfo, bidx);
        KFloat center_y = box_cy(boxfo, bidx);

        KInt nx = (KInt)(center_x / (KFloat) grid_size);
        KInt ny = (KInt)(center_y / (KFloat) grid_size);

        set_scale_nx(box_scale_coodout, idx, nx);
        set_scale_ny(box_scale_coodout, idx, ny);
    }
}

/*
__global__ void kai_ker_yolo_set_scaled_true_box(KInt size, KFloat* coods, KFloat* sizes, KFloat* confs, KInt* class, KFloat* selected, KFloat* boxes, KInt* catid,
                                             KInt nd, KInt nscale, KInt img_size, KInt grid_cnt, KInt num_scales, KInt anchor_per_scale) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (selected[idx] < 0.5f) return;

        KInt nb = idx / (num_scales * anchor_per_scale);
        KInt ns = idx / anchor_per_scale % num_scales;
        KInt na = idx % anchor_per_scale;

        if (ns != nscale) return;

        //selected[idx] = -999;

        //KInt bidx = nb * 4;

        KFloat width = box_wd(boxes, nb);
        KFloat height = box_ht(boxes, nb);

        KFloat center_x = box_cx(boxes, nb) + width / 2;
        KFloat center_y = box_cy(boxes, nb) + height / 2;

        KFloat grid_size = (KFloat)img_size / (KFloat)grid_cnt;

        KInt nx = (KInt)(center_x / grid_size);
        KInt ny = (KInt)(center_y / grid_size);

        assert(nx >= 0 && nx < grid_cnt);
        assert(ny >= 0 && ny < grid_cnt);

        KInt didx = ((nd * anchor_per_scale + na) * grid_cnt + nx) * grid_cnt + ny;

        coods[2 * didx + 0] = center_x;
        coods[2 * didx + 1] = center_y;

        sizes[2 * didx + 0] = width;
        sizes[2 * didx + 1] = height;

        confs[didx] = 1.0f;

        class[didx] = catid[nb];
    }
}
*/

__global__ void kai_ker_yolo_conv_fmap(KInt size, KFloat* pred, KFloat* fmap, KInt* anchors, KInt img_size, KInt grid_cnt, KInt anchors_cnt, KInt class_num) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ny = idx / (grid_cnt * anchors_cnt) % grid_cnt;
        KInt nx = idx / anchors_cnt % grid_cnt;
        KInt na = idx % anchors_cnt;

        KInt* anchors = anchors + (na * 2);

        KFloat rescale_ratio = __fdiv_rn((KFloat)img_size, (KFloat)grid_cnt);

        KFloat box_width = __fmul_rn(__expf(fmap_wd(fmap, idx)), (KFloat)anchors[0]);
        KFloat box_height = __fmul_rn(__expf(fmap_ht(fmap, idx)), (KFloat)anchors[1]);

        KFloat center_x = __fmul_rn(__fadd_rn(dev_sigmoid(fmap_cx(fmap, idx)), (KFloat)nx), rescale_ratio);
        KFloat center_y = __fmul_rn(__fadd_rn(dev_sigmoid(fmap_cy(fmap, idx)), (KFloat)ny), rescale_ratio);

        set_pred_xmin(pred, idx, __fsub_rn(center_x, __fdiv_rn(box_width, 2.0f)));
        set_pred_ymin(pred, idx, __fsub_rn(center_y, __fdiv_rn(box_height, 2.0f)));
        set_pred_xmax(pred, idx, __fadd_rn(center_x, __fdiv_rn(box_width, 2.0f)));
        set_pred_ymax(pred, idx, __fadd_rn(center_y, __fdiv_rn(box_height, 2.0f)));

        set_pred_conf(pred, idx, dev_sigmoid(fmap_conf(fmap, idx)));

        for (KInt n = 0; n < class_num; n++) {
            set_pred_class(pred, idx, n, dev_sigmoid(fmap_class(fmap, idx, n)));
        }
    }
}

__global__ void kai_ker_yolo_eval_iou(KInt size, KFloat* iou, KInt* boxid, KFloat* fmap, KInt* imgfo, KFloat* boxfo, KInt* scale_boxes, KInt* anchors,
                                  KInt nanchor, KInt img_size, KInt grid_cnt, KInt box_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (grid_cnt * grid_cnt * nanchor * box_cnt);
        KInt ny = idx / (grid_cnt * nanchor * box_cnt) % grid_cnt;
        KInt nx = idx / (nanchor * box_cnt) % grid_cnt;
        KInt na = idx / box_cnt % nanchor;
        KInt nb = idx % box_cnt;

        iou[idx] = -1.0f;
        boxid[idx] = -1;

        KInt nbox = scale_nbox(scale_boxes, nb);
        KInt nimg = box_nimg(boxfo, nbox);

        if (img_ndata(imgfo, nimg) != nd) return;  // different data in minibatch

        bool matched = true;

        if (scale_nanchor(scale_boxes, nb) != na) matched = false; // different anchor
        else if (scale_nx(scale_boxes, nb) != nx) matched = false; // different center_x
        else if (scale_ny(scale_boxes, nb) != ny) matched = false; // different center_y

        KInt midx = idx / box_cnt;

        KInt* anchors = anchors + (na * 2);
        KFloat rescale_ratio = __fdiv_rn((KFloat)img_size, (KFloat)grid_cnt);

        KFloat pcx = (dev_sigmoid(fmap_cx(fmap, midx)) + (KFloat)nx) * rescale_ratio;
        KFloat pcy = (dev_sigmoid(fmap_cy(fmap, midx)) + (KFloat)ny) * rescale_ratio;

        KFloat pwd = __expf(fmap_wd(fmap, midx)) * (KFloat)anchors[0];
        KFloat pht = __expf(fmap_ht(fmap, midx)) * (KFloat)anchors[1];

        KFloat pxmin = pcx - pwd / 2.0f;
        KFloat pxmax = pcx + pwd / 2.0f;
        KFloat pymin = pcy - pht / 2.0f;
        KFloat pymax = pcy + pht / 2.0f;

        KFloat tcx = box_cx(boxfo, nbox);
        KFloat tcy = box_cy(boxfo, nbox);
        KFloat twd = box_wd(boxfo, nbox);
        KFloat tht = box_ht(boxfo, nbox);

        KFloat txmin = tcx - twd / 2.0f;
        KFloat txmax = tcx + twd / 2.0f;
        KFloat tymin = tcy - tht / 2.0f;
        KFloat tymax = tcy + tht / 2.0f;

        KFloat ixmin = myfmax(pxmin, txmin);
        KFloat ixmax = myfmin(pxmax, txmax);
        KFloat iymin = myfmax(pymin, tymin);
        KFloat iymax = myfmin(pymax, tymax);

        KFloat iwd = ixmax - ixmin;
        KFloat iht = iymax - iymin;

        KFloat iou = 0.0f;

        if (iwd > 0 && iht > 0) {
            KFloat pred_box_area = pwd * pht;
            KFloat true_box_area = twd * tht;
            KFloat intersect_area = iwd * iht;
            KFloat union_area = pred_box_area + true_box_area - intersect_area;

            iou = intersect_area / union_area;
        }

        iou[idx] = iou;
        boxid[idx] = matched ? nbox : -1;
    }
}

__global__ void kai_ker_yolo_select_best_iou(KInt size, KInt* best_box, KFloat* best_iou, KFloat* iou, KInt* boxid, KInt box_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt matched_idx = -1;
        KFloat best_iou = -1.0f;

        KInt uidx = idx * box_cnt;

        for (KInt n = 0; n < box_cnt; n++) {
            if (iou[uidx + n] > best_iou) best_iou = iou[uidx + n];
            if (boxid[uidx + n] >= 0) matched_idx = boxid[uidx + n];
        }

        best_box[idx] = matched_idx;
        best_iou[idx] = best_iou;
    }
}

__global__ void kai_ker_yolo_eval_losses(KInt size, KFloat* loss, KFloat* fmap, KFloat* boxfo, KInt* best_box, KFloat* best_iou, KInt* anchors,
    KInt mb_size, KInt img_size, KInt grid_cnt, KInt nanchor, KInt class_num, bool use_focal, bool smooth_onehot) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nc = idx % FMAP_SIZE;
        KInt fidx = idx / FMAP_SIZE;

        loss[idx] = 0;

        KInt nbox = best_box[fidx];

        KFloat grid_size = (KFloat)img_size / (KFloat)grid_cnt;
        KFloat felem = fmap[idx];
        //KFloat felem = fmap_elem(fmap, fidx, nc);

        KFloat mixed = (nbox >= 0) ? box_mixed(boxfo, nbox) : 1.0f;
        KFloat coef = mixed / (KFloat)mb_size;

        if (nc < 4) {
            if (nbox < 0) return;

            KInt ny = fidx / (grid_cnt * nanchor) % grid_cnt;
            KInt nx = fidx / (nanchor) % grid_cnt;
            KInt na = fidx % nanchor;

            KFloat twidth = box_wd(boxfo, nbox);
            KFloat theight = box_ht(boxfo, nbox);

            KFloat loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (KFloat)img_size), __fdiv_rn(theight, (KFloat)img_size)));

            if (nc < 2) {
                KFloat pxy = dev_sigmoid(felem);
                KFloat tcenter = box_rect(boxfo, nbox, (KInt) nc);
                KFloat txy = tcenter / grid_size - (KFloat)((nc == 0) ? nx : ny);

                KFloat xy_diff = __fsub_rn(pxy, txy);
                KFloat dist_sq = __fmul_rn(xy_diff, xy_diff);

                loss[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, coef));
            }
            else {
                KFloat anchor_size = (KFloat)anchors[2 * na + (nc -2)];

                KFloat pwh = felem;
                KFloat tsz = box_rect(boxfo, nbox, (KInt) nc);
                KFloat twh = __logf(clip(__fdiv_rn(tsz, anchor_size), 1e-9f, 1e9f));

                KFloat wh_diff = __fsub_rn(pwh, twh);
                KFloat dist_sq = __fmul_rn(wh_diff, wh_diff);

                loss[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, coef));
            }
        }
        else if (nc == 4) {
            bool object_mask = nbox >= 0;
            bool ignore_mask = best_iou[fidx] < 0.5f;
            
            if (!object_mask && !ignore_mask) return;

            KFloat loss_conf = dev_sigmoid_cross_entropy_with_logits(felem, object_mask);

            if (use_focal) {
                KFloat focal_diff = dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask);
                KFloat focal_mask = __fmul_rn(focal_diff, focal_diff);

                loss_conf = __fmul_rn(loss_conf, focal_mask);
            }

            loss[idx] = __fmul_rn(loss_conf, coef);
        }
        else {
            if (nbox < 0) return;

            KInt pclass = (KInt) nc - 5;
            KInt tclass = box_catid(boxfo, nbox);

            KFloat z = (pclass == tclass) ? 1.0f : 0.0f;

            if (smooth_onehot) {
                KFloat delta = 0.01f;
                z += delta / (KFloat)class_num;
                if (pclass == tclass) z -= delta;
            }

            KFloat entropy = dev_sigmoid_cross_entropy_with_logits(felem, z);

            loss[idx] = __fmul_rn(entropy, coef);
        }
    }
}

__global__ void kai_ker_yolo_eval_grads(KInt size, KFloat* grad, KFloat* fmap, KFloat* boxfo, KInt* best_box, KFloat* best_iou, KInt* anchors,
    KInt mb_size, KInt img_size, KInt grid_cnt, KInt nanchor, KInt class_num, bool use_focal, bool smooth_onehot) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nc = idx % FMAP_SIZE;
        KInt fidx = idx / FMAP_SIZE;

        grad[idx] = 0;

        /*
        if (idx == 0) {
            grad[0] = dev_sigmoid(-16.5f);
            grad[1] = dev_sigmoid_cross_entropy_with_logits(-16.5f, 0.0f);
            grad[2] = dev_sigmoid_cross_entropy_with_logits_derv(-16.5f, 0.0f);
            return;
        }
        else if (idx < 4) return;
        */

        KInt nbox = best_box[fidx];

        KFloat grid_size = (KFloat)img_size / (KFloat)grid_cnt;
        KFloat felem = fmap[idx];

        KFloat mixed = (nbox >= 0) ? box_mixed(boxfo, nbox) : 1.0f;
        KFloat coef = mixed / (KFloat)mb_size;

        if (nc < 4) {
            if (nbox < 0) return;

            KInt ny = fidx / (grid_cnt * nanchor) % grid_cnt;
            KInt nx = fidx / (nanchor) % grid_cnt;
            KInt na = fidx % nanchor;

            KFloat twidth = box_wd(boxfo, nbox);
            KFloat theight = box_ht(boxfo, nbox);

            KFloat loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (KFloat)img_size), __fdiv_rn(theight, (KFloat)img_size)));

            if (nc < 2) {
                KFloat pxy = dev_sigmoid(felem);
                KFloat tcenter = box_rect(boxfo, nbox, (KInt) nc);
                KFloat txy = tcenter / grid_size - (KFloat)((nc == 0) ? nx : ny);

                KFloat xy_diff = __fsub_rn(pxy, txy);
                KFloat g_pxy = __fmul_rn(__fmul_rn(2.0f, xy_diff), __fmul_rn(loss_scale, coef));

                grad[idx] = __fmul_rn(g_pxy, dev_sigmoid_derv(felem, pxy));
            }
            else {
                KFloat anchor_size = (KFloat)anchors[2 * na + (nc -2)];

                KFloat pwh = felem;
                KFloat tsz = box_rect(boxfo, nbox, (KInt) nc);
                KFloat twh = __logf(clip(__fdiv_rn(tsz, anchor_size), 1e-9f, 1e9f));

                KFloat wh_diff = __fsub_rn(pwh, twh);

                grad[idx] = __fmul_rn(__fmul_rn(2.0f, wh_diff), __fmul_rn(loss_scale, coef));
            }
        }
        else if (nc == 4) {
            bool object_mask = nbox >= 0;
            bool ignore_mask = best_iou[fidx] < 0.5f;

            if (!object_mask && !ignore_mask) return;

            KFloat g_loss_conf = coef; // backprop for 'loss = __fmul_rn(loss, coef)'
            KFloat g_focal_x = 0;

            if (use_focal) {
                KFloat focal_diff = dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask);
                KFloat focal_mask = __fmul_rn(focal_diff, focal_diff);

                KFloat loss_conf = dev_sigmoid_cross_entropy_with_logits(felem, object_mask);

                KFloat g_focal_mask = __fmul_rn(g_loss_conf, loss_conf); // backprop for 'loss_conf = __fmul_rn(loss_conf, focal_mask)'
                KFloat g_focal_diff = __fmul_rn(g_focal_mask, __fmul_rn(2.0f, focal_diff)); // backprop for 'focal_mask = __fmul_rn(focal_diff, focal_diff)'

                g_focal_x = __fmul_rn(g_focal_diff, dev_sigmoid_derv(felem)); // backprop for 'focal_diff = dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask)'
                g_loss_conf = __fmul_rn(g_loss_conf, focal_mask); // backprop for 'loss_conf = __fmul_rn(loss_conf, focal_mask)'
            }

            KFloat g_loss_x = __fmul_rn(g_loss_conf, dev_sigmoid_cross_entropy_with_logits_derv(felem, object_mask)); // backprop for 'loss_conf = dev_sigmoid_cross_entropy_with_logits(felem, object_mask)'

            grad[idx] = __fadd_rn(g_loss_x, g_focal_x);
        }
        else {
            if (nbox < 0) return;

            KInt pclass = (KInt) nc - 5;
            KInt tclass = box_catid(boxfo, nbox);

            KFloat z = (pclass == tclass) ? 1.0f : 0.0f;

            if (smooth_onehot) {
                KFloat delta = 0.01f;
                z += delta / (KFloat)class_num;
                if (pclass == tclass) z -= delta;
            }

            KFloat sig_derv = dev_sigmoid_cross_entropy_with_logits_derv(felem, z);

            grad[idx] = __fmul_rn(sig_derv, coef);
        }
    }
}

/*
__global__ void kai_ker_yolo_loss_cood(KInt size, KFloat* loss, KFloat* fmap, KFloat* boxfo, KInt* best_box, KInt img_size, KInt nanchor, KInt grid_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt ny = idx / (grid_cnt * nanchor) % grid_cnt;
        KInt nx = idx / nanchor % grid_cnt;

        loss[idx] = 0;

        KInt nbox = best_box[idx];

        if (nbox < 0) return;

        KFloat pcx = dev_sigmoid(fmap_cx(fmap, idx));
        KFloat pcy = dev_sigmoid(fmap_cy(fmap, idx));

        KFloat grid_size = (KFloat)img_size / (KFloat)grid_cnt;

        KFloat tcenter_x = box_cx(boxfo, nbox);
        KFloat tcenter_y = box_cy(boxfo, nbox);

        KFloat tcx = tcenter_x / grid_size - (KFloat)nx;
        KFloat tcy = tcenter_y / grid_size - (KFloat)ny;

        KFloat x_diff = __fsub_rn(pcx, tcx);
        KFloat y_diff = __fsub_rn(pcy, tcy);

        KFloat dist_sq = __fadd_rn(__fmul_rn(x_diff, x_diff), __fmul_rn(y_diff, y_diff));

        KFloat twidth = box_wd(boxfo, nbox);
        KFloat theight = box_ht(boxfo, nbox);

        KFloat loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (KFloat)img_size), __fdiv_rn(theight, (KFloat)img_size)));
        KFloat mixed = box_mixed(boxfo, nbox);
         
        loss[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, mixed));
    }
}

__global__ void kai_ker_yolo_loss_size(KInt size, KFloat* loss, KFloat* fmap, KFloat* boxfo, KInt* best_box, KInt* anchors, KInt img_size, KInt nanchor, KInt grid_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt na = idx % nanchor;
        KInt nbox = best_box[idx];

        loss[idx] = 0;

        if (nbox < 0) return;

        KFloat anchor_width = (KFloat) anchors[2 * na];
        KFloat anchor_height = (KFloat)anchors[2 * na + 1];

        KFloat pwd = fmap_wd(fmap, idx);
        KFloat pht = fmap_ht(fmap, idx);

        KFloat twidth = box_wd(boxfo, nbox);
        KFloat theight = box_ht(boxfo, nbox);

        KFloat twd = __logf(clip(__fdiv_rn(twidth, anchor_width), 1e-9f, 1e9f));
        KFloat tht = __logf(clip(__fdiv_rn(theight, anchor_height), 1e-9f, 1e9f));

        KFloat w_diff = __fsub_rn(pwd, twd);
        KFloat h_diff = __fsub_rn(pht, tht);

        KFloat dist_sq = __fadd_rn(__fmul_rn(w_diff, w_diff), __fmul_rn(h_diff, h_diff));
        KFloat loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (KFloat)img_size), __fdiv_rn(theight, (KFloat)img_size)));
        KFloat mixed = box_mixed(boxfo, nbox); 

        loss[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, mixed));
    }
}

__global__ void kai_ker_yolo_loss_conf(KInt size, KFloat* loss, KFloat* fmap, KFloat* boxfo, KInt* best_box, KFloat* best_iou, bool use_focal) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        loss[idx] = 0;

        KInt nbox = best_box[idx];

        bool object_mask = nbox >= 0;
        bool ignore_mask = best_iou[idx] < 0.5f;

        bool conf_pos = object_mask;
        bool conf_neg = !object_mask && ignore_mask;

        KFloat x = fmap_conf(fmap, idx);

        KFloat loss_pos = conf_pos ? dev_sigmoid_cross_entropy_with_logits(x, object_mask) : 0;
        KFloat loss_neg = conf_neg ? dev_sigmoid_cross_entropy_with_logits(x, object_mask) : 0;
        
        KFloat conf_loss = loss_pos +loss_neg;

        if (use_focal) {
            KFloat sigmoid = dev_sigmoid(x);

            KFloat focal_diff = __fsub_rn(object_mask, sigmoid);
            KFloat focal_mask = __fmul_rn(focal_diff, focal_diff);

            conf_loss = __fmul_rn(conf_loss, focal_mask);
        }

        KFloat mixed = (nbox >= 0) ? box_mixed(boxfo, nbox) : 1.0f;

        loss[idx] = __fmul_rn(conf_loss, mixed);
    }
}

__global__ void kai_ker_yolo_loss_class(KInt size, KFloat* loss, KFloat* fmap, KFloat* boxfo, KInt* best_box, KInt class_num, bool smooth_onehot) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        loss[idx] = 0;

        KInt nbox = best_box[idx];

        if (nbox < 0) return;

        KInt target_id = box_catid(boxfo, nbox);

        KFloat loss_class = 0;

        for (KInt n = 0; n < class_num; n++) {
            KFloat x = fmap_class(fmap, idx, n);
            KFloat z = (n == target_id) ? 1 : 0;

            if (smooth_onehot) {
                KFloat delta = 0.01f;
                z += delta / (KFloat) class_num;
                if (n == target_id) z -= delta;
            }

            KFloat entropy = dev_sigmoid_cross_entropy_with_logits(x, z);

            loss_class = __fadd_rn(loss_class, entropy);
        }

        KFloat mixed = box_mixed(boxfo, nbox);

        loss[idx] = __fmul_rn(loss_class, mixed);
    }
}
*/

/*
__global__ void kai_ker_yolo_coods_derv(KInt size, KFloat* gcoods, KFloat* gmixedout, KFloat* pcood, KFloat* pmixed, KFloat* box_rect, KInt* best_box, KInt img_size, KInt nanchor, KInt grid_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt box_id = best_box[idx];
        if (box_id < 0) return;

        KInt pidx = idx * 2;
        KInt tidx = box_id * 4;

        KFloat pcenter_x = pcood[pidx];
        KFloat pcenter_y = pcood[pidx + 1];

        KFloat tcenter_x = box_rect[tidx];
        KFloat tcenter_y = box_rect[tidx + 1];
        KFloat twidth = box_rect[tidx + 2];
        KFloat theight = box_rect[tidx + 3];

        KFloat grid_size = __fdiv_rn((KFloat)img_size, (KFloat)grid_cnt);

        KFloat x_diff = __fdiv_rn(__fsub_rn(pcenter_x, tcenter_x), grid_size);
        KFloat y_diff = __fdiv_rn(__fsub_rn(pcenter_y, tcenter_y), grid_size);

        KFloat dist_sq = __fadd_rn(__fmul_rn(x_diff, x_diff), __fmul_rn(y_diff, y_diff));

        KFloat loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (KFloat)img_size), __fdiv_rn(theight, (KFloat)img_size)));
        KFloat mixed = pmixed[idx];

        KInt xidx = pidx;
        KInt yidx = pidx + 1;

        gcoods[xidx] = 2 * x_diff * loss_scale * mixed / grid_size;
        gcoods[yidx] = 2 * y_diff * loss_scale * mixed / grid_size;

        gmixedout[idx] += dist_sq * loss_scale;
    }
}

__global__ void kai_ker_yolo_sizes_derv(KInt size, KFloat* gsizes, KFloat* gmixedout, KFloat* psize, KFloat* pmixed, KFloat* box_rect, KInt* best_box, KInt* anchors, KInt img_size, KInt nanchor, KInt grid_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt na = idx / (grid_cnt * grid_cnt) % nanchor;
        KInt box_id = best_box[idx];

        if (box_id < 0) return;

        KInt pidx = idx * 2;
        KInt tidx = box_id * 4;

        KFloat anchor_width = (KFloat)anchors[2 * na];
        KFloat anchor_height = (KFloat)anchors[2 * na + 1];

        KFloat pwidth = psize[pidx];
        KFloat pheight = psize[pidx + 1];

        KFloat twidth = box_rect[tidx + 2];
        KFloat theight = box_rect[tidx + 3];

        KFloat log_pw = __logf(clip(__fdiv_rn(pwidth, anchor_width), 1e-9f, 1e9f));
        KFloat log_ph = __logf(clip(__fdiv_rn(pheight, anchor_height), 1e-9f, 1e9f));

        KFloat log_tw = __logf(clip(__fdiv_rn(twidth, anchor_width), 1e-9f, 1e9f));
        KFloat log_th = __logf(clip(__fdiv_rn(theight, anchor_height), 1e-9f, 1e9f));

        KFloat w_diff = __fsub_rn(log_pw, log_tw);
        KFloat h_diff = __fsub_rn(log_ph, log_th);

        KFloat dist_sq = __fadd_rn(__fmul_rn(w_diff, w_diff), __fmul_rn(h_diff, h_diff));
        KFloat loss_scale = __fsub_rn(2.0f, __fmul_rn(__fdiv_rn(twidth, (KFloat)img_size), __fdiv_rn(theight, (KFloat)img_size)));
        KFloat mixed = pmixed[idx];

        //loss[idx] = __fmul_rn(dist_sq, __fmul_rn(loss_scale, mixed));

        KInt widx = pidx;
        KInt hidx = pidx + 1;

        gsizes[widx] = 2 * w_diff * loss_scale * mixed / pwidth;    // anchor  크기는 미분 과정에서 상쇄되어 서라짐, 상수 나눗셈은 로그에서 상수 뺄셈으로 변환됨에 유이
        gsizes[hidx] = 2 * h_diff * loss_scale * mixed / pheight;   // anchor  크기는 미분 과정에서 상쇄되어 서라짐, 상수 나눗셈은 로그에서 상수 뺄셈으로 변환됨에 유이

        gmixedout[idx] += dist_sq * loss_scale;
    }
}

__global__ void kai_ker_yolo_confs_derv(KInt size, KFloat* gconfs, KFloat* gmixedout, KFloat* pconfs, KFloat* pmixed, KInt* best_box, KFloat* best_iou, bool use_focal) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt box_id = best_box[idx];

        bool object_mask = (box_id >= 0);
        bool ignore_mask = (best_iou[idx] < 0.5f);

        bool conf_pos = object_mask;
        bool conf_neg = ! object_mask && ignore_mask;

        KFloat x = pconfs[idx];
        KFloat z = object_mask;

        KFloat entropy = dev_sigmoid_cross_entropy_with_logits(pconfs[idx], object_mask);

        KFloat loss_pos = conf_pos ? entropy : 0;
        KFloat loss_neg = conf_neg ? entropy : 0;

        KFloat conf_loss = loss_pos + loss_neg;

        KFloat mixed = pmixed[idx];

        KFloat sigmoid = dev_sigmoid(x);

        KFloat g_conf_loss = mixed;
        KFloat g_x = 0;

        if (use_focal) {
            KFloat focal_diff = __fsub_rn(object_mask, sigmoid);
            KFloat focal_mask = __fmul_rn(focal_diff, focal_diff);

            KFloat alpha = 1.0f;

            KFloat y = sigmoid;
            KFloat sig_derv = dev_sigmoid_derv(x, y); // __fmul_rn(y, __fsub_rn(1.0f, y));

            KFloat g_focal_mask = conf_loss;
            KFloat g_focal_diff = 2 * alpha * focal_diff * g_focal_mask;
            KFloat g_sigmoid = - g_focal_diff;
            KFloat g_x_focal = sig_derv * g_sigmoid;

            g_conf_loss *= focal_mask;

            g_x += g_x_focal;
            //g_x += 2 * alpha * focal_diff * conf_loss * sig_derv * mixed;

            conf_loss *= focal_mask;
        }

        KFloat g_entropy = __fsub_rn(sigmoid, z) * g_conf_loss;

        g_x += conf_pos ? g_entropy : 0;
        g_x += conf_neg ? g_entropy : 0;

        gconfs[idx] = g_x;
        
        gmixedout[idx] += conf_loss;
    }
}

__global__ void kai_ker_yolo_probs_derv(KInt size, KFloat* gprobs, KFloat* gmbufs, KFloat* pprobs, KFloat* pmixed, KInt* best_box, KInt* boxfo, KInt class_num, bool smooth_onehot) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt midx = idx / class_num;
        KInt nclass = idx % class_num;

        KInt box_id = best_box[midx];
        if (box_id < 0) return;

        KInt target_id = boxfo[box_id * 5 + 4];

        KFloat x = pprobs[idx];
        KFloat z = (nclass == target_id) ? 1 : 0;
        
        if (smooth_onehot) {
            KFloat delta = 0.01f;
            z += delta / (KFloat) class_num;
            if (nclass == target_id) z -= delta;
        }

        KFloat sigmoid = dev_sigmoid(x);
        KFloat sig_derv = dev_sigmoid_derv(x, sigmoid);
        KFloat entropy = dev_sigmoid_cross_entropy_with_logits(x, z);

        KFloat loss_class = entropy;
        KFloat g_loss_class = sig_derv;

        KFloat mixed = pmixed[midx];

        gprobs[idx] = g_loss_class * mixed;
        
        if (gmbufs) gmbufs[idx] = loss_class;
    }
}


__global__ void kai_ker_yolo_acc_gmbufs(KInt size, KFloat* gmixedout, KFloat* gmbufs, KInt class_num) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt pidx = idx * class_num;

        for (KInt n = 0; n < class_num; n++) {
            gmixedout[idx] += gmbufs[pidx + n];
        }
    }
}

__global__ void kai_ker_yolo_fmap_derv(KInt size, KFloat* grad_fmap, KFloat* gcoods, KFloat* gsizes, KFloat* gconfs, KFloat* gprobs, KFloat* gmixed, KFloat* coods, KFloat* sizes, KFloat* confs, KFloat* probs, KFloat* mixed,
    KInt* anchors, KInt img_size, KInt grid_cnt, KInt anchors_cnt, KInt class_num, bool use_mixed) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nvec = class_num + (use_mixed ? 6 : 5);

        KInt nd = idx / (grid_cnt * grid_cnt * anchors_cnt * nvec);
        KInt nx = idx / (grid_cnt * anchors_cnt * nvec) % grid_cnt;
        KInt ny = idx / (anchors_cnt * nvec) % grid_cnt;
        KInt na = idx / nvec % anchors_cnt;
        KInt nc = idx % nvec;

        //KInt fidx = (((nd * grid_cnt + nh) * grid_cnt + nw) * anchors_cnt + na) * nvec + nc;
        //KFloat x = fmap[fidx];

        // box centers
        if (nc < 2) {
            KInt cidx = ((((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny) * 2) + nc;

            KFloat y = coods[cidx];;
            KFloat g_y = gcoods[cidx];

            // rescale backprop
            KFloat rescale_ratio = __fdiv_rn((KFloat)img_size, (KFloat)grid_cnt);
            KFloat g_offset_term = __fmul_rn(g_y, rescale_ratio);

            // add offset backprop: do nothing

            // sigmoid backprop
            KFloat g_sigmoid = dev_sigmoid_derv(0, y); // x is don't care in dev_sigmoid_derv //__fmul_rn(y, __fsub_rn(1.0f, y));
            KFloat g_x = __fmul_rn(g_offset_term, g_sigmoid);

            grad_fmap[idx] = g_x;
        }
        // box sizes
        else if (nc < 4) {
            KInt cidx = ((((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny) * 2) + (nc - 2);

            KFloat y = sizes[cidx];
            KFloat g_y = gsizes[cidx];

            KFloat g_x = __fmul_rn(g_y, y);
            grad_fmap[idx] = g_x;
        }
        // conf
        else if (nc < 5) {
            KInt cidx = ((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny;
            grad_fmap[idx] = gconfs[cidx];
        }
        // conf
        else if (nc < class_num + 5) {
            KInt pidx = (((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny) * class_num + (nc - 5);
            grad_fmap[idx] = gprobs[pidx];
        }
        // mixed: 대안 1: 사용 않음(1), 대안 2: sigmoid(x), 대안3: x => 일단 대안2로 테스트, 대안3은 비용 함수에 음수 난무 초래
        else {
            if (use_mixed) {
                KInt cidx = ((nd * anchors_cnt + na) * grid_cnt + nx) * grid_cnt + ny;

                KFloat y = mixed[cidx];
                KFloat g_y = gmixed[cidx];

                KFloat g_sigmoid = __fmul_rn(y, __fsub_rn(1.0f, y));
                KFloat g_x = __fmul_rn(g_y, g_sigmoid);

                grad_fmap[idx] = g_x;
            }
            else {
                grad_fmap[idx] = 0;
            }
        }
    }
}
*/

__global__ void kai_ker_yolo_select_pred_boxes(KInt size, unsigned char* flag, KInt* ndata, KFloat* conf_rects, KFloat* fmap, KInt* anchors, KFloat pred_conf_thr, KInt out_base, KInt img_size, KInt grid_cnt, KInt anchor_per_scale, KInt class_num) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (anchor_per_scale * grid_cnt * grid_cnt);
        KInt na = idx / (grid_cnt * grid_cnt) % anchor_per_scale;
        KInt nx = idx / (grid_cnt) % grid_cnt;
        KInt ny = idx % grid_cnt;

        KInt flag_bytes = (class_num + 7) / 8;

        KInt nidx = idx + out_base;
        KInt fidx = nidx * flag_bytes;
        KInt ridx = nidx * 5;

        KInt fmap_idx = (((nd * grid_cnt + nx) * grid_cnt + ny) * anchor_per_scale + na);

        KFloat sigmoid_conf = dev_sigmoid(pred_conf(fmap, fmap_idx));
        if (sigmoid_conf < pred_conf_thr) return;

        KInt cat_id = 0;
        KFloat max_term = fmap[fmap_idx + 5];

        for (KInt n = 1; n < class_num; n++) {
            KFloat x = fmap[fmap_idx + n + 5]; // category logits
            if (x > max_term) {
                cat_id = n;
                max_term = x;
            }
        }

        conf_rects[ridx] = sigmoid_conf;
        flag[fidx+(cat_id)/8] = (unsigned char) (1 << (cat_id % 8));
        ndata[nidx] = (KInt) nd;

        // box centers
        for (KInt n = 0; n < 2; n++) {
            KFloat x = fmap[fmap_idx + n]; // center_x or center_y
                
            // sigmoid
            KFloat term1 = (x > 0) ? 1 : __expf(x);
            KFloat term2 = __fadd_rn(1.0f, __expf((x > 0) ? -x : x));
            KFloat sigmoid_term = __fdiv_rn(term1, term2);

            // add offset
            KFloat offset = (KFloat)((n == 0) ? nx : ny);
            KFloat offset_term = __fadd_rn(sigmoid_term, offset);

            // rescale
            KFloat rescale_ratio = __fdiv_rn((KFloat)img_size, (KFloat)grid_cnt);
            KFloat rescaled_term = __fmul_rn(offset_term, rescale_ratio);

            conf_rects[ridx + n + 1] = rescaled_term;
        }

        // box sizes
        for (KInt n = 0; n < 2; n++) {
            KFloat x = fmap[fmap_idx + n + 2]; // width or height
            KFloat exp_term = __expf(x);
            KFloat anchor = (KFloat) anchors[na * 2 + n];
            KFloat rescaled_term = __fmul_rn(exp_term, anchor);

            conf_rects[ridx + n + 3] = rescaled_term;
        }
    }
}

/*
__global__ void kai_ker_yolo_eval_box_pair_ious(KInt size, KInt* pinfo, KFloat* ious, KInt* idxs, KInt* cats, KFloat* rects, KInt* boxfo, KFloat* box_rect, KInt tbox_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt npred = idx / tbox_cnt;
        KInt ntrue = idx % tbox_cnt;

        KInt pfidx = idxs[npred];
        KInt pnidx = pfidx * 2;
        KInt pridx = pfidx * 4;

        KInt tnidx = ntrue * 5;
        KInt tridx = ntrue * 4;

        KInt pred_ndata = cats[pnidx + 0];
        KInt pred_catid = cats[pnidx + 1];

        KInt true_ndata = boxfo[tnidx + 0];
        KInt true_catid = boxfo[tnidx + 4];

        KInt pidx = idx * 2;

        pinfo[pidx] = 0;  // means invalid pair
        pinfo[pidx + 1] = pred_catid;

        if (pred_ndata != true_ndata) return;
        if (pred_catid != true_catid) return;

        pinfo[pidx] = 1;  // means valid pair

        KFloat pcenter_x = rects[pridx];
        KFloat pcenter_y = rects[pridx + 1];
        KFloat pwidth = rects[pridx + 2];
        KFloat pheight = rects[pridx + 3];

        KFloat pleft = pcenter_x - pwidth / 2.0f;
        KFloat pright = pcenter_x + pwidth / 2.0f;
        KFloat ptop = pcenter_y - pheight / 2.0f;
        KFloat pbottom = pcenter_y + pheight / 2.0f;

        KFloat tcenter_x = box_rect[tridx];
        KFloat tcenter_y = box_rect[tridx + 1];
        KFloat twidth = box_rect[tridx + 2];
        KFloat theight = box_rect[tridx + 3];

        KFloat tleft = tcenter_x - twidth / 2.0f;
        KFloat tright = tcenter_x + twidth / 2.0f;
        KFloat ttop = tcenter_y - theight / 2.0f;
        KFloat tbottom = tcenter_y + theight / 2.0f;

        if (pleft > tright) return;
        if (tleft > pright) return;
        if (ptop > tbottom) return;
        if (ttop > pbottom) return;

        KFloat ileft = (pleft > tleft) ? pleft : tleft;
        KFloat iright = (pright < tright) ? pright : tright;
        KFloat itop = (ptop > ttop) ? ptop : ttop;
        KFloat ibottom = (pbottom < tbottom) ? pbottom : tbottom;

        KFloat iwidth = iright - ileft;
        KFloat iheight = ibottom - itop;

        KFloat pred_box_area = pwidth * pheight;
        KFloat true_box_area = twidth * theight;
        KFloat intersect_area = iwidth * iheight;
        KFloat union_area = pred_box_area + true_box_area - intersect_area;

        KFloat iou = intersect_area / union_area;
        
        ious[idx] = iou;
    }
}

__global__ void kai_ker_yolo_count_true_boxes(KInt size, KInt* tbox_cnt, KInt* boxfo, KInt tbox_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nclass = idx;

        KInt found = 0;

        for (KInt ntrue = 0; ntrue < tbox_cnt; ntrue++) {
            KInt tnidx = ntrue * 5;
            KInt true_catid = boxfo[tnidx + 4];
            if (true_catid == nclass) found++;
        }

        tbox_cnt[idx] = found;
    }
}

__global__ void kai_ker_yolo_count_pred_boxes(KInt size, KInt* pbox_cnt, KInt* pairfo, KInt pbox_cnt, KInt tbox_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nclass = idx;

        KInt found = 0;

        for (KInt npred = 0; npred < pbox_cnt; npred++) {
            KInt pnidx = npred * tbox_cnt;
            KInt pred_catid = pairfo[pnidx * 2 + 1];
            if (pred_catid == nclass) found++;
        }

        pbox_cnt[idx] = found;
    }
}

__global__ void kai_ker_yolo_count_matched_box_pairs(KInt size, KInt* match_cnt, KInt* pairfo, KFloat* ious, KInt iou_thr_cnt, KFloat iou_thr_from, KFloat iou_thr_step, KInt pbox_cnt, KInt tbox_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nclass = idx / iou_thr_cnt;
        KInt niouthr = idx % iou_thr_cnt;

        KFloat iou_thr = iou_thr_from + iou_thr_step * (KFloat) niouthr;

        KInt found = 0;

        for (KInt npred = 0; npred < pbox_cnt; npred++) {
            KInt pnidx = npred * tbox_cnt;
            
            KInt pred_catid = pairfo[pnidx * 2+ 1];
            if (pred_catid != nclass) continue;

            for (KInt ntrue = 0; ntrue < tbox_cnt; ntrue++, pnidx++) {
                KInt is_valid_pair = pairfo[pnidx * 2];
                if (!is_valid_pair) continue;
                KFloat iou = ious[pnidx];
                if (iou >= iou_thr) {
                    found++;
                    break;
                }
            }
        }

        match_cnt[idx] = found;
    }
}

__global__ void kai_ker_yolo_eval_prec_recall(KInt size, KFloat* precision, KFloat* recall, KInt* tbox_cnt, KInt* pbox_cnt, KInt* match_cnt, KInt iou_thr_cnt) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nclass = idx / iou_thr_cnt;
        KInt niouthr = idx % iou_thr_cnt;

        KInt tbox_cnt = tbox_cnt[idx];
        KInt pbox_cnt = pbox_cnt[idx];

        KInt match_cnt = match_cnt[idx];

        KFloat precision = 0;
        KFloat recall = 0;

        if (match_cnt > 0) {
            precision = (KFloat) match_cnt / (KFloat)pbox_cnt;
            recall = (KFloat)match_cnt / (KFloat)tbox_cnt;
        }

        precision[idx] = precision;
        recall[idx] = recall;
    }
}
*/

__global__ void kai_ker_yolo_eval_predict_score(KInt size, KFloat* score, KFloat* pred, KInt class_num, KFloat score_thresh) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / class_num;
        KInt nc = idx % class_num;

        KFloat conf_prob = pred_conf(pred, nr);
        KFloat class_prof = pred_class(pred, nr, nc);

        KFloat score = __fmul_rn(conf_prob, class_prof);
        score[idx] = (score >= score_thresh) ? score : 0;
    }
}

__global__ void kai_ker_yolo_get_boxes(KInt size, KFloat* boxes, KFloat* old_boxes, KInt* idxs, KFloat* score, KFloat* pred, KInt old_count, KInt grid_cnt, KInt anchors_cnt, KInt class_num) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        //KInt bidx = idx * 8;

        if (idx < old_count) {
            memcpy(boxes + idx * RES_BOX_SIZE, old_boxes + idx * RES_BOX_SIZE, RES_BOX_SIZE * sizeof(KFloat));
        }
        else {
            KInt sidx = idxs[idx - old_count];

            KInt nd = sidx / (grid_cnt * grid_cnt * anchors_cnt * class_num);
            KInt pidx = sidx / class_num;

            set_res_xmin(boxes, idx, pred_xmin(pred, pidx));
            set_res_ymin(boxes, idx, pred_ymin(pred, pidx));
            set_res_xmax(boxes, idx, pred_xmax(pred, pidx));
            set_res_ymax(boxes, idx, pred_ymax(pred, pidx));

            set_res_score(boxes, idx, score[sidx]);
            set_res_class(boxes, idx, (KInt) sidx % class_num);
            set_res_ndata(boxes, idx, (KInt)nd);
            set_res_flag(boxes, idx, 0);
        }
    }
}

__global__ void kai_ker_yolo_non_max_suppression(KInt size, KFloat* boxesout, KInt box_cnt, KInt max_boxes, KFloat iou_thr) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {   // idx: class index
        //KInt step = 0;

        while (true) {
            //step++;

            KInt max_id = -1;
            KFloat max_score = 0;
            for (KInt n = 0; n < box_cnt; n++) {
                if (res_flag(boxesout, n) != 0) continue;
                if (res_class(boxesout, n) != (KFloat)idx) continue;
                if (res_score(boxesout, n) > max_score) {
                    max_id = n;
                    max_score = res_score(boxesout, n);
                }
            }
            if (max_id < 0) break;
            
            set_res_flag(boxesout, max_id, 1);

            KFloat width = res_xmax(boxesout, max_id) - res_xmin(boxesout, max_id);
            KFloat height = res_ymax(boxesout, max_id) - res_ymin(boxesout, max_id);

            KFloat box1_area = width * height;

            for (KInt n = 0; n < box_cnt; n++) {
                if (res_flag(boxesout, n) != 0) continue;
                if (res_class(boxesout, n) != (KFloat)idx) continue;
                
                //KFloat box2_area = boxesout[n * 7 + 2] * boxesout[n * 7 + 3];

                KFloat left   = myfmax(res_xmin(boxesout, max_id), res_xmin(boxesout, n));
                KFloat right  = myfmin(res_xmax(boxesout, max_id), res_xmax(boxesout, n)); 
                KFloat top    = myfmax(res_ymin(boxesout, max_id), res_ymin(boxesout, n));
                KFloat bottom = myfmin(res_ymax(boxesout, max_id), res_ymax(boxesout, n));

                KFloat width = (left < right) ? (right - left) : 0;
                KFloat height = (top < bottom) ? (bottom - top) : 0;
                
                KFloat inter_area = width * height;
                //KFloat union_area = box1_area + box2_area - inter_area;

                //KFloat iou = inter_area / union_area;
                KFloat iou = inter_area / box1_area;

                if (iou >= iou_thr) {
                    set_res_flag(boxesout, n, 2);
                }
            }
        }
    }
}

__global__ void kai_ker_real_to_complex(KInt size, KFloat* c, KFloat* f) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        c[idx] = (idx % 2 == 0) ? f[idx / 2] : 0;
    }
}

__global__ void kai_ker_short_to_complex(KInt size, KFloat* c, short* f) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        c[idx] = (idx % 2 == 0) ? (KFloat) f[idx / 2] : 0;
    }
}

__global__ void kai_ker_complex_to_abs(KInt size, KFloat* a, KFloat* c) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KFloat real = c[idx * 2];
        KFloat image = c[idx * 2 + 1];
        a[idx] = __fsqrt_rn(real * real + image * image);
    }
}

__global__ void kai_ker_fft_step(KInt size, KFloat* dst, KFloat* src, KInt data_num, KInt step) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nd = idx / (data_num * 2);
        KInt nx = (idx / 2) % data_num;
        bool is_real = (idx % 2) == 0;

        KInt stride = data_num / step;

        KInt pos1 = (nx / stride * (2 * stride)) % data_num + nx % stride;
        KInt pos2 = pos1 + stride;

        KFloat x1_real = src[(nd * data_num + pos1) * 2];
        KFloat x1_image = src[(nd * data_num + pos1) * 2 + 1];

        KFloat x2_real = src[(nd * data_num + pos2) * 2];
        KFloat x2_image = src[(nd * data_num + pos2) * 2 + 1];

        KFloat theta = -2 * CUDART_PI_F * (nx / stride * stride) / data_num;

        KFloat t_real = __cosf(theta);
        KFloat t_image = __sinf(theta);

        if (is_real)
            dst[idx] = x1_real + x2_real * t_real - x2_image * t_image;
        else
            dst[idx] = x1_image + x2_real * t_image + x2_image * t_real;
    }
}

__global__ void kai_ker_eveal_hash_match_point(KInt size, KFloat* p, KFloat* c1, KFloat* c2, KInt nrow, KInt ncol, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / ncol;
        KInt nc = idx % ncol;

        KFloat* c1 = c1 + nr * nvec;
        KFloat* c2 = c2 + nc * nvec;

        KFloat point = 0;

        for (KInt n = 0; n < nvec; n++) {
            if ((c1[n] > 0.5f) == (c2[n] > 0.5f)) point = point + 1;
        }
        
        p[idx] = point;
    }
}

__global__ void kai_ker_eveal_vector_dist(KInt size, KFloat* d, KFloat* c1, KFloat* c2, KInt nrow, KInt ncol, KInt nvec) {
    KInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        KInt nr = idx / ncol;
        KInt nc = idx % ncol;

        KFloat* c1 = c1 + nr * nvec;
        KFloat* c2 = c2 + nc * nvec;

        KFloat sq_sum = 0;

        for (KInt n = 0; n < nvec; n++) {
            sq_sum += (c1[n] - c2[n]) * (c1[n] - c2[n]);
        }
        
        d[idx] = __fsqrt_rn(sq_sum / nvec);
    }
}
#endif