/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <assert.h>

#include "../../core/dataset.h"

#include "../cuda_conn.cuh"
#include "../cuda_kernels.h"
#include "../cuda_math.h"
#include "../cuda_note.h"

#include "../../core/func_timer.h"
#include "../../core/log.h"

float Dataset::m_forward_postproc_cuda(Dict y, Dict out, enum loss_mode enumMode) {
	CudaConn cuda("Dataset::forward", NULL);

	Array<float> est = out["data"];

	float* cuda_est = CudaConn::GetCudaMem(est, "est");
	float* cuda_res = cuda.alloc_float_mem(est.shape(), "res");
	float* cuda_ans = NULL;

	int64 size = est.total_size();

	if (enumMode != loss_mode::classify_1st && enumMode != loss_mode::classify_idx) {
		Array<float> ans = y["data"];
		cuda_ans = CudaConn::GetCudaMem(ans, "ans");
	}

	if (enumMode == loss_mode::regression) {
		cu_call(ker_mse, size, (size, cuda_res, cuda_est, cuda_ans));
	}
	else if (enumMode == loss_mode::binary) {
		cu_call(ker_sigmoid_cross_entropy_with_logits, size, (size, cuda_res, cuda_est, cuda_ans));
	}
	else if (enumMode == loss_mode::classify) {
		int64 nvec = est.axis_size(-1);
		size /= nvec;

		cu_call(ker_softmax_cross_entropy_with_logits, size, (size, cuda_res, cuda_est, cuda_ans, nvec));
	}
	else if (enumMode == loss_mode::classify_idx) {
		Array<int64> ans = y["wids"];
		int64* cuda_ans = CudaConn::GetCudaMem(ans, "ans");
		int64 nvec = est.axis_size(-1);
		size /= nvec;

		cu_call(ker_softmax_cross_entropy_with_logits_idx, size, (size, cuda_res, cuda_est, cuda_ans, nvec));
	}
	else if (enumMode == loss_mode::classify_1st) {
		int64 nvec = est.axis_size(-1);
		size /= nvec;

		cu_call(ker_softmax_cross_entropy_with_logits_1st, size, (size, cuda_res, cuda_est, nvec));
	}
	else if (enumMode == loss_mode::autoencode) {
		cu_call(ker_mse, size, (size, cuda_res, cuda_est, cuda_ans));
	}

	for (int64 range = ADD_RANGE, ssize = size; ssize > 1; range *= ADD_RANGE) {
		ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
		cu_call(ker_sum, ssize, (ssize, cuda_res, size, range));
	}

	float loss = cuda.get_nth_element(cuda_res, 0) / (float)size;

	return loss;
}

Dict Dataset::m_backprop_postproc_cuda(Dict y, Dict out, enum loss_mode enumMode) {
	CudaConn cuda("Dataset::backprop", NULL);

	Array<float> est = out["data"];

	int64 mb_size = est.axis_size(0);


	float* cuda_gh = cuda.alloc_float_mem(est.shape(), "G_hidden");

	float* cuda_est = CudaConn::GetCudaMem(est, "est");
	float* cuda_ans = NULL;

	if (enumMode != loss_mode::classify_1st && enumMode != loss_mode::classify_idx) {
		Array<float> ans = y["data"];
		cuda_ans = CudaConn::GetCudaMem(ans, "ans");
	}

	int64 hsize = est.total_size();
	int64 nvec = est.axis_size(-1);

	if (enumMode == loss_mode::regression) {
		float coef = 2.0f / (float)hsize;
		cu_call(ker_mult_diff_coef, hsize, (hsize, cuda_gh, cuda_est, cuda_ans, coef));
	}
	else if (enumMode == loss_mode::binary) {
		float coef = 1.0f / (float)hsize;
		cu_call(ker_sigmoid_cross_entropy_with_logits_derv, hsize, (hsize, cuda_gh, cuda_est, cuda_ans, coef));
	}
	else if (enumMode == loss_mode::classify) {
		int64 mb_size = hsize / nvec;
		float coef = 1.0f / (float)mb_size;
		cu_call(ker_softmax_cross_entropy_with_logits_derv, hsize, (hsize, cuda_gh, cuda_est, cuda_ans, nvec, coef));
	}
	else if (enumMode == loss_mode::classify_idx) {
		Array<int64> ans = y["wids"];
#ifdef NO_LENGTH_TEST
		if (output_seq()) ans = ans.merge_time_axis();
#else
#endif
		int64* cuda_ans = CudaConn::GetCudaMem(ans, "ans");
		int64 mb_size = hsize / nvec;
		float coef = 1.0f / (float)mb_size;
		cu_call(ker_softmax_cross_entropy_with_logits_idx_derv, hsize, (hsize, cuda_gh, cuda_est, cuda_ans, nvec, coef));
	}
	else if (enumMode == loss_mode::classify_1st) {
		int64 mb_size = hsize / nvec;
		float coef = 1.0f / (float)mb_size;
		cu_call(ker_softmax_cross_entropy_with_logits_1st_derv, hsize, (hsize, cuda_gh, cuda_est, nvec, coef));
	}
	else if (enumMode == loss_mode::autoencode) {
		float coef = 2.0f / (float)hsize;
		cu_call(ker_mult_diff_coef, hsize, (hsize, cuda_gh, cuda_est, cuda_ans, coef));
		/*
		Array<float> diff = aux["diff"];
		Shape shape = diff.shape();
		Array<float> g_loss_square = kmath->ones(shape) / (float)shape.total_size();
		Array<float> g_square_diff = diff * 2;
		Array<float> G_square = g_loss_square * G_loss;
		G_output = g_square_diff * G_square;
		*/
	}

	Array<float> G_out_data = cuda.detach(cuda_gh, "G_output");

	Dict G_out;

	G_out["data"] = G_out_data;

	return G_out;
}

float Dataset::m_eval_accuracy_cuda(Dict x, Dict y, Dict out, enum loss_mode enumMode) {
	CudaConn cuda("Dataset::eval_accuracy", NULL);

	Array<float> est = out["data"];

	Shape eshape = est.shape();

	float* cuda_est = cuda.GetCudaMem(est, "est");
	float* cuda_buf = cuda.alloc_float_mem(eshape, "buf");
	float* cuda_ans = NULL;

	if (enumMode != loss_mode::classify_1st && enumMode != loss_mode::classify_idx) {
		Array<float> ans = y["data"];
		cuda_ans = cuda.copy_to_buffer(ans, "ans");
	}

	int64 size = eshape.total_size();
	int64 nvec = eshape[-1];
	int64 nrow = size / nvec;

	if (enumMode == loss_mode::regression || enumMode == loss_mode::autoencode) {
		cu_call(ker_mse_diff_sq, size, (size, cuda_buf, cuda_est, cuda_ans));

		for (int64 range = ADD_RANGE, ssize = size; ssize > 1; range *= ADD_RANGE) {
			ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
			cu_call(ker_sum, ssize, (ssize, cuda_buf, size, range));
			cu_call(ker_sum, ssize, (ssize, cuda_ans, size, range));
		}

		float mse = cuda.get_nth_element(cuda_buf, 0) / (float)size;
		float mean = cuda.get_nth_element(cuda_ans, 0) / (float)size;

		float accuracy = 1.0f - (float)::sqrt(mse) / mean;

		return accuracy;
	}
	else if (enumMode == loss_mode::binary) {
		cu_call(ker_bin_acc, size, (size, cuda_buf, cuda_est, cuda_ans));

		for (int64 range = ADD_RANGE, ssize = size; ssize > 1; range *= ADD_RANGE) {
			ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
			cu_call(ker_sum, ssize, (ssize, cuda_buf, size, range));
		}

		float accuracy = cuda.get_nth_element(cuda_buf, 0) / (float)size;

		return accuracy;
	}
	else if (enumMode == loss_mode::classify) {
		cu_call(ker_class_acc, nrow, (nrow, cuda_buf, cuda_est, cuda_ans, nvec));

		for (int64 range = ADD_RANGE, ssize = nrow; ssize > 1; range *= ADD_RANGE) {
			ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
			cu_call(ker_sum, ssize, (ssize, cuda_buf, nrow, range));
		}

		float accuracy = cuda.get_nth_element(cuda_buf, 0) / (float)nrow;

		return accuracy;
	}
	else if (enumMode == loss_mode::classify_idx) {
		Array<int64> ans = y["wids"];
		int64* cuda_ans = CudaConn::GetCudaMem(ans, "ans"); 

		cu_call(ker_class_idx_acc, nrow, (nrow, cuda_buf, cuda_est, cuda_ans, nvec));

		for (int64 range = ADD_RANGE, ssize = nrow; ssize > 1; range *= ADD_RANGE) {
			ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
			cu_call(ker_sum, ssize, (ssize, cuda_buf, nrow, range));
		}

		float accuracy = cuda.get_nth_element(cuda_buf, 0) / (float)nrow;

		return accuracy;
	}
	else if (enumMode == loss_mode::classify_1st) {
		cu_call(ker_class_1st_acc, nrow, (nrow, cuda_buf, cuda_est, nvec));

		for (int64 range = ADD_RANGE, ssize = nrow; ssize > 1; range *= ADD_RANGE) {
			ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
			cu_call(ker_sum, ssize, (ssize, cuda_buf, nrow, range));
		}

		float accuracy = cuda.get_nth_element(cuda_buf, 0) / (float)nrow;

		return accuracy;
	}
	else {
		throw KaiException(KERR_ASSERT);
	}
	return 0;
}

/*
Value Dataset::m_get_estimate_cuda(Array<float> output, enum loss_mode enumMode) {
	CudaConn cuda("Dataset::get_estimate", NULL);

	Array<float> estimate;

	if (enumMode == loss_mode::regression) {
		estimate = output;
	}
	else if (enumMode == loss_mode::binary) {
		estimate = kmath->sigmoid(output);
	}
	else if (enumMode == loss_mode::classify) {
		estimate = kmath->softmax(output);
	}
	else if (enumMode == loss_mode::classify_idx) {
		estimate = kmath->softmax(output);
	}
	else if (enumMode == loss_mode::autoencode) {
		estimate = output;
	}

	return estimate;
}
*/