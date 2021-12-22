/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kcudamath.h"
#include "../gpu_cuda/device_manager.h"
#include "../gpu_cuda/kai_kernels.cuh"
#include "../math/karray.h"
#include "../exec/exec_context.h"
#include "../utils/kutil.h"

KaiCudaMath::KaiCudaMath(KaiDeviceManager* pDevManager) : KaiHostMath() {
	m_pDevManager = pDevManager;
	m_block_size = 512;
}

KaiCudaMath::~KaiCudaMath() {
}

KInt KaiCudaMath::m_getMaxBatchSize(KInt nNeedSizePerbatch) {
	return m_pDevManager->getMaxBatchSize(nNeedSizePerbatch);
}

void KaiCudaMath::m_lock() {
	m_pDevManager->lock();
}

void KaiCudaMath::m_unlock() {
	m_pDevManager->unlock();
}

void KaiCudaMath::cudaErrorCheck() {
	m_pDevManager->errorCheck();
}

KaiArray<KFloat> KaiCudaMath::copy(KaiArray<KFloat> arr) {
	if (arr.is_cuda()) {
		KInt nAllocSize;
		KFloat* dp = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
		KInt ssize = arr.total_size();
		cudaMemcpy(dp, arr.data_ptr(), ssize * sizeof(KFloat), cudaMemcpyDeviceToDevice);
		KaiArray<KFloat> result(dp, arr.shape(), nAllocSize, m_pDevManager);
		return result;
	}
	else {
		return KaiHostMath::copy(arr);
	}
}

KaiArray<KFloat> KaiCudaMath::zeros(KaiShape shape) {
	KInt nAllocSize;
	KFloat* dp = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KInt ssize = shape.total_size();
	kcuda_call(kai_ker_set, ssize, (ssize, dp, 0));
	KaiArray<KFloat> result(dp, shape, nAllocSize, m_pDevManager);
	return result;
}

KaiArray<KFloat> KaiCudaMath::ones(KaiShape shape, KFloat fill) {
	KInt nAllocSize;

	KFloat* dp = m_pDevManager->allocFloatMemory(shape, &nAllocSize);

	KInt asize = shape.total_size();

	kcuda_call(kai_ker_set, asize, (asize, dp, fill));

	KaiArray<KFloat> result(dp, shape, nAllocSize, m_pDevManager);

	return result;
}

KaiArray<KFloat> KaiCudaMath::random_uniform(KaiShape shape) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "should be called only by the hostmath");
}

KaiArray<KFloat> KaiCudaMath::random_normal(KaiShape shape, KFloat mean, KFloat std, KBool adapt) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "should be called only by the hostmath");
}

KaiArray<KInt> KaiCudaMath::to_cuda(KaiArray<KInt> arr) {
	if (arr.is_cuda()) return arr;

	KaiShape shape = arr.shape();
	KInt size = arr.total_size();
	KInt* pHost = arr.data_ptr();
	KInt mem_size = 0;

	KInt* pCuda = (KInt*)m_pDevManager->allocMemory(size * sizeof(KInt), &mem_size);
	cudaMemcpy(pCuda, pHost, size * sizeof(KInt), cudaMemcpyHostToDevice);
	KaiArray<KInt> cuda_arr(pCuda, shape, mem_size, m_pDevManager);

	return cuda_arr;
}

KaiArray<KFloat> KaiCudaMath::to_cuda(KaiArray<KFloat> arr) {
	if (arr.is_cuda()) return arr;

	KaiShape shape = arr.shape();
	KInt size = arr.total_size();
	KFloat* pHost = arr.data_ptr();
	KInt mem_size = 0;

	KFloat* pCuda = (KFloat*)m_pDevManager->allocMemory(size * sizeof(KFloat), &mem_size);
	cudaMemcpy(pCuda, pHost, size * sizeof(KFloat), cudaMemcpyHostToDevice);
	KaiArray<KFloat> cuda_arr(pCuda, shape, mem_size, m_pDevManager);

	return cuda_arr;
}

KaiArray<KInt> KaiCudaMath::to_host(KaiArray<KInt> arr) {
	if (!arr.is_cuda()) return arr;

	KaiArray<KInt> host_arr(arr.shape());

	KInt size = arr.total_size();
	KInt* pCuda = arr.data_ptr();
	KInt* pHost = host_arr.data_ptr();

	cudaMemcpy(pHost, pCuda, size * sizeof(KInt), cudaMemcpyDeviceToHost);

	return host_arr;
}

KaiArray<KFloat> KaiCudaMath::to_host(KaiArray<KFloat> arr) {
	if (!arr.is_cuda()) return arr;

	KaiArray<KFloat> host_arr(arr.shape());

	KInt size = arr.total_size();
	KFloat* pCuda = arr.data_ptr();
	KFloat* pHost = host_arr.data_ptr();

	cudaMemcpy(pHost, pCuda, size * sizeof(KFloat), cudaMemcpyDeviceToHost);

	return host_arr;
}

// The following 4 methods have been added by Hyung-jae, Son (2021-08-20)

void KaiCudaMath::to_cuda(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	if (!arrDst.is_cuda())
		throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "target array is not in device memory");

	if (arrSrc.total_size() < nSrcStart + nCount || arrDst.total_size() < nDstStart + nCount)
		throw KaiException(KERR_INDEX_OUT_OF_RANGE, "invalid start index or copy size");

	enum cudaMemcpyKind copy_method = cudaMemcpyDefault;

	if (arrSrc.is_cuda()) copy_method = cudaMemcpyDeviceToDevice;
	else copy_method = cudaMemcpyHostToDevice;

	cudaMemcpy(arrDst.data_ptr()+nDstStart, arrSrc.data_ptr()+nSrcStart, sizeof(KInt)*nCount, copy_method);
}

void KaiCudaMath::to_cuda(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	if (!arrDst.is_cuda())
		throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "target array is not in device memory");

	if (arrSrc.total_size() < nSrcStart + nCount || arrDst.total_size() < nDstStart + nCount)
		throw KaiException(KERR_INDEX_OUT_OF_RANGE, "invalid start index or copy size");

	enum cudaMemcpyKind copy_method = cudaMemcpyDefault;

	if (arrSrc.is_cuda()) copy_method = cudaMemcpyDeviceToDevice;
	else copy_method = cudaMemcpyHostToDevice;

	cudaMemcpy(arrDst.data_ptr()+nDstStart, arrSrc.data_ptr()+nSrcStart, sizeof(KFloat)*nCount, copy_method);
}

void KaiCudaMath::to_host(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	if (arrDst.is_cuda())
		throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "target array is not in host memory");

	if (arrSrc.total_size() < nSrcStart + nCount || arrDst.total_size() < nDstStart + nCount)
		throw KaiException(KERR_INDEX_OUT_OF_RANGE, "invalid start index or copy size");

	enum cudaMemcpyKind copy_method = cudaMemcpyDefault;

	if (arrSrc.is_cuda()) copy_method = cudaMemcpyDeviceToHost;
	else copy_method = cudaMemcpyHostToHost;

	cudaMemcpy(arrDst.data_ptr()+nDstStart, arrSrc.data_ptr()+nSrcStart, sizeof(KInt)*nCount, copy_method);
}

void KaiCudaMath::to_host(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	if (arrDst.is_cuda())
		throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "target array is not in host memory");

	if (arrSrc.total_size() < nSrcStart + nCount || arrDst.total_size() < nDstStart + nCount)
		throw KaiException(KERR_INDEX_OUT_OF_RANGE, "invalid start index or copy size");

	enum cudaMemcpyKind copy_method = cudaMemcpyDefault;

	if (arrSrc.is_cuda()) copy_method = cudaMemcpyDeviceToHost;
	else copy_method = cudaMemcpyHostToHost;

	cudaMemcpy(arrDst.data_ptr()+nDstStart, arrSrc.data_ptr()+nSrcStart, sizeof(KFloat)*nCount, copy_method);
}

KaiArray<KInt> KaiCudaMath::arange(KInt nCount) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "should be called only by the hostmath");
}

KaiArray<KInt> KaiCudaMath::subrange(KaiArray<KInt> arr, KInt nStart, KInt nCount) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "should be called only by the hostmath");
}

void KaiCudaMath::shuffle(KaiArray<KInt> arr) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "should be called only by the hostmath");
}

void KaiCudaMath::shuffle(KaiArray<KInt> arr, KInt nStart, KInt nCount) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "should be called only by the hostmath");
}

KaiArray<KFloat> KaiCudaMath::matmul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	KaiShape shape1 = arr1.shape();
	KaiShape shape2 = arr2.shape();

	if (shape1.size() < 2 || shape2.size() < 2) throw KaiException(KERR_MATMUL_SHAPE_MISMATCH);
	if (shape1.size() > 2 && shape2.size() > 2) throw KaiException(KERR_MATMUL_SHAPE_MISMATCH);
	
	KInt nvecs = (shape1[-1] > shape2[0]) ? shape1[-1] : shape2[0];

	if (shape1.total_size() % nvecs != 0) throw KaiException(KERR_MATMUL_SHAPE_MISMATCH);
	if (shape2.total_size() % nvecs != 0) throw KaiException(KERR_MATMUL_SHAPE_MISMATCH);

	KInt nrows = shape1.total_size() / nvecs;
	KInt ncols = shape2.total_size() / nvecs;

	KaiShape yshape = shape1.remove_tail_by_size(nvecs).append(shape2.remove_head_by_size(nvecs));

	KInt nAllocSize;

	KFloat* p_mul = m_pDevManager->allocFloatMemory(yshape, &nAllocSize);
	KFloat* p_arr1 = arr1.data_ptr();
	KFloat* p_arr2 = arr2.data_ptr();

	KInt ysize = yshape.total_size();

	kcuda_call(kai_ker_matmul, ysize, (ysize, p_mul, p_arr1, p_arr2, nrows, nvecs, ncols));

	KaiArray<KFloat> mul(p_mul, yshape, nAllocSize, m_pDevManager);

	return mul;
}

KaiArray<KFloat> KaiCudaMath::add_bias(KaiArray<KFloat> arr, KaiArray<KFloat> bias) {
	KaiShape ashape = arr.shape();
	KaiShape bshape = bias.shape();

	if (ashape.size() <= 0 || bshape.size() != 1) throw KaiException(KERR_ADD_BIAS_SHAPE_MISMATCH);

	KInt ncols = bshape[0];
	KInt asize = arr.total_size();

	if (ncols != ashape[ashape.size() - 1]) throw KaiException(KERR_ADD_BIAS_SHAPE_MISMATCH);

	KInt nAllocSize;

	KFloat* p_sum = m_pDevManager->allocFloatMemory(ashape, &nAllocSize);
	KFloat* ap = arr.data_ptr();
	KFloat* bp = bias.data_ptr();

	kcuda_call(kai_ker_add_bias, asize, (asize, p_sum, ap, bp, ncols));

	KaiArray<KFloat> sum(p_sum, ashape, nAllocSize, m_pDevManager);

	return sum;
}

KaiArray<KFloat> KaiCudaMath::transpose(KaiArray<KFloat> arr) {
	if (arr.dim() < 2) throw KaiException(KERR_BAD_DIMENSION_FOR_ARRAY_TRANSPOSE);

	KaiShape ashape = arr.shape();

	KInt vec_size = ashape[-1];
	KInt mb_size = ashape.total_size() / vec_size;

	KaiShape tshape = KaiShape{ vec_size , mb_size };

	KInt nAllocSize;

	KFloat* p_trans = m_pDevManager->allocFloatMemory(tshape, &nAllocSize);
	KFloat* p_arr = arr.data_ptr();

	KInt tsize = tshape.total_size();

	kcuda_call(kai_ker_transpose, tsize, (tsize, p_trans, p_arr, vec_size, mb_size));

	KaiArray<KFloat> trans(p_trans, tshape, nAllocSize, m_pDevManager);

	return trans;
}

KaiArray<KFloat> KaiCudaMath::sum_on_column(KaiArray<KFloat> arr) {
	if (arr.dim() < 2) throw KaiException(KERR_BAD_DIMENSION_FOR_ARRAY_SUM_ON_COL);

	KaiShape ashape = arr.shape();

	KInt cols = ashape[-1];
	KInt rows = ashape.total_size() / cols;

	KaiShape sshape = KaiShape{ cols };

	KInt nAllocSize;

	KFloat* p_sum = m_pDevManager->allocFloatMemory(sshape, &nAllocSize);
	KFloat* p_arr = arr.data_ptr();


	kcuda_call(kai_ker_sum_on_column, cols, (cols, p_sum, p_arr, rows, cols));

	KaiArray<KFloat> sum(p_sum, sshape, nAllocSize, m_pDevManager);

	return sum;
}

KaiArray<KFloat> KaiCudaMath::acivate(KaiArray<KFloat> arr, KInt nActFuncID, KaiExecContext* pContext) {
	KaiShape ashape = arr.shape();

	KInt nAllocSize;

	KFloat* dp = m_pDevManager->allocFloatMemory(ashape, &nAllocSize);
	KFloat* ap = arr.data_ptr();

	KInt asize = arr.total_size();

	if ((Ken_actfunc)nActFuncID == Ken_actfunc::custom) throw KaiException(KERR_UNIMPEMENTED_YET, "custom activate function");

	KFloat leaky_alpha = 0;

	if ((Ken_actfunc)nActFuncID != Ken_actfunc::leaky_relu) leaky_alpha = pContext->get_property("leaky_alpha", 0.1f);

	kcuda_call(kai_ker_activate, asize, (asize, dp, ap, nActFuncID, leaky_alpha));

	KaiArray<KFloat> result(dp, ashape, nAllocSize, m_pDevManager);

	return result;
}

KaiArray<KFloat> KaiCudaMath::acivate_backprop(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat> y, KInt nActFuncID, KaiExecContext* pContext) {
	KaiShape ashape = garr.shape();

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(ashape, &nAllocSize);
	KFloat* pa = garr.data_ptr();
	KFloat* px = x.data_ptr();
	KFloat* py = y.data_ptr();

	KInt asize = garr.total_size();

	if ((Ken_actfunc)nActFuncID == Ken_actfunc::custom) throw KaiException(KERR_UNIMPEMENTED_YET, "custom activate function");

	KFloat leaky_alpha = 0;

	if ((Ken_actfunc)nActFuncID != Ken_actfunc::leaky_relu) leaky_alpha = pContext->get_property("leaky_alpha", 0.1f);

	kcuda_call(kai_ker_activate_backprop, asize, (asize, pd, pa, px, py, nActFuncID, leaky_alpha));

	KaiArray<KFloat> result(pd, ashape, nAllocSize, m_pDevManager);

	return result;
}

KaiArray<KFloat> KaiCudaMath::minus(KaiArray<KFloat> arr) {
	KaiShape ashape = arr.shape();

	if (ashape.size() == 0) return arr;

	KInt nAllocSize;

	KFloat* dp = m_pDevManager->allocFloatMemory(ashape, &nAllocSize);
	KFloat* ap = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_minus, asize, (asize, dp, ap));

	KaiArray<KFloat> result(dp, ashape, nAllocSize, m_pDevManager);

	return result;
}

KaiArray<KFloat> KaiCudaMath::filter(KaiArray<KFloat> xarr, KaiArray<KInt> mask) {
	KaiShape xshape = xarr.shape();
	KaiShape mshape = mask.shape();

	if (xshape.total_size() % mshape.total_size() != 0) throw KaiException(KERR_UNMATCHING_SHAPE_FOR_FILTER_OPERATION);
	if (xshape.total_size() / mshape.total_size() != xshape[-1]) throw KaiException(KERR_UNMATCHING_SHAPE_FOR_FILTER_OPERATION);

	KInt msize = mshape.total_size();
	KInt nAlloc1, nAlloc2;

	KFloat* px = xarr.data_ptr();
	KInt* pm = mask.data_ptr();
	KInt* p_map = m_pDevManager->allocIntMemory(KaiShape{ msize + 1 }, &nAlloc1);

	kcuda_call(kai_ker_mask_to_idx, 1, (1, p_map, pm, msize));

	KInt mask_cnt = fetch(p_map, msize);
	KInt vec_size = xshape[-1];

	KaiShape fshape{ mask_cnt, xshape[-1] };

	KInt fsize = fshape.total_size();

	KFloat* p_filtered = m_pDevManager->allocFloatMemory(fshape, &nAlloc2);

	kcuda_call(kai_ker_filter, fsize, (fsize, p_filtered, px, p_map, vec_size));

	m_pDevManager->freeMemory(p_map, nAlloc1);

	KaiArray<KFloat> filtered(p_filtered, fshape, nAlloc2, m_pDevManager);

	return filtered;
}

KaiArray<KFloat> KaiCudaMath::eval_binary_op(exp_op op_code, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr2.total_size() == 1) {
		return eval_binary_op(op_code, arr1, fetch(arr2));
	}

	if (op_code == exp_op::softmax_cross_entropy_with_logits) {
		if (arr1.total_size() != arr2.total_size()) {
			throw KaiException(KERR_UNIMPEMENTED_YET);
		}
		return softmax_cross_entropy_with_logits(arr1, arr2);
	}

	KaiShape mshape = arr1.shape();
	KInt vec_size1 = 1;
	KInt vec_size2 = 1;

	if (arr1.total_size() > arr2.total_size()) {
		vec_size1 = arr1.total_size() / arr2.total_size();
		mshape.remove_tail_by_size(vec_size1); // to check shape error
	}
	else if (arr1.total_size() < arr2.total_size()) {
		mshape = arr2.shape();
		vec_size2 = arr2.total_size() / arr1.total_size();
		mshape.remove_tail_by_size(vec_size2); // to check shape error
	}

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(mshape, &nAllocSize);
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt msize = mshape.total_size();

	kcuda_call(kai_ker_binary_op, msize, (msize, pd, p1, p2, op_code, vec_size1, vec_size2));

	KaiArray<KFloat> dst(pd, mshape, nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::add(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::add, arr1, arr2);
}

KaiArray<KFloat> KaiCudaMath::sub(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::sub, arr1, arr2);
}

KaiArray<KFloat> KaiCudaMath::mul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::mult, arr1, arr2);
}

KaiArray<KFloat> KaiCudaMath::div(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::div, arr1, arr2);
}

KaiArray<KFloat> KaiCudaMath::gt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::gt, arr1, arr2);
}

KaiArray<KFloat> KaiCudaMath::lt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::lt, arr1, arr2);
}

KaiArray<KFloat> KaiCudaMath::equal(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::equal, arr1, arr2);
}

/*
KaiArray<KFloat> KaiCudaMath::add(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) {
		if (arr2.total_size() != 1) throw KaiException(KERR_UNIMPEMENTED_YET); 
		return add(arr1, fetch(arr2));
	}

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr1.shape(), &nAllocSize);
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_add, asize, (asize, pd, p1, p2));

	KaiArray<KFloat> dst(pd, arr1.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::sub(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) {
		if (arr2.total_size() != 1) throw KaiException(KERR_UNIMPEMENTED_YET);
		return sub(arr1, fetch(arr2));
	}

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr1.shape(), &nAllocSize);
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_sub, asize, (asize, pd, p1, p2));

	KaiArray<KFloat> dst(pd, arr1.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::mul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) {
		if (arr2.total_size() != 1) throw KaiException(KERR_UNIMPEMENTED_YET);
		return mul(arr1, fetch(arr2));
	}

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr1.shape(), &nAllocSize);
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_mul, asize, (asize, pd, p1, p2));

	KaiArray<KFloat> dst(pd, arr1.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::div(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) {
		if (arr2.total_size() != 1) throw KaiException(KERR_UNIMPEMENTED_YET);
		return div(arr1, fetch(arr2));
	}

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr1.shape(), &nAllocSize);
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_div, asize, (asize, pd, p1, p2));

	KaiArray<KFloat> dst(pd, arr1.shape(), nAllocSize, m_pDevManager);

	return dst;
}
*/

void KaiCudaMath::add_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr2.is_empty()) return; // get 처리에 의해 parallel layer의 입력이 이용되지 않아 grad를 empty로 처리한 경우
	if (arr1.shape() != arr2.shape()) {
		logger.Print("add_on: %s vs %s", arr1.shape().desc().c_str(), arr2.shape().desc().c_str());
		throw KaiException(KERR_UNIMPEMENTED_YET);
	}

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_add, asize, (asize, p1, p1, p2));
}

void KaiCudaMath::sub_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_sub, asize, (asize, p1, p1, p2));
}

void KaiCudaMath::mul_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_mul, asize, (asize, p1, p1, p2));
}

void KaiCudaMath::div_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt asize = arr1.total_size();

	kcuda_call(kai_ker_div, asize, (asize, p1, p1, p2));
}

KaiArray<KFloat> KaiCudaMath::eval_binary_op(exp_op op_code, KaiArray<KFloat> arr, KFloat term) {
	if (arr.shape().size() == 0) return arr;

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_binary_op, asize, (asize, pd, pa, term, op_code));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::add(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::add, arr, term);
}

KaiArray<KFloat> KaiCudaMath::sub(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::sub, arr, term);
}

KaiArray<KFloat> KaiCudaMath::mul(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::mult, arr, term);
}

KaiArray<KFloat> KaiCudaMath::div(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::div, arr, term);
}

KaiArray<KFloat> KaiCudaMath::sigmoid_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::sigmoid_cross_entropy_with_logits, arr1, arr2);
}

KaiArray<KFloat> KaiCudaMath::softmax_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.total_size() != arr2.total_size()) throw KaiException(KERR_UNIMPEMENTED_YET);
	if (arr1.shape()[-1] != arr2.shape()[-1]) throw KaiException(KERR_UNIMPEMENTED_YET);

	KInt nAllocSize;
	KInt nvec = arr1.axis_size(-1);
	KInt nrow = arr1.total_size() / nvec;

	KaiShape shape = arr1.shape().cut_tail(1);
	
	KFloat* pd = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	kcuda_call(kai_ker_softmax_cross_entropy_with_logits, nrow, (nrow, pd, p1, p2, nvec));

	KaiArray<KFloat> entropy(pd, shape, nAllocSize, m_pDevManager);

	return entropy;
}

KaiArray<KFloat> KaiCudaMath::softmax_cross_entropy_with_logits_idx(KaiArray<KFloat> arr1, KaiArray<KInt> arr2) {
	KInt nvec = arr1.axis_size(-1);
	KInt nrow = arr1.total_size() / nvec;

	if (arr2.total_size() != nrow) throw KaiException(KERR_UNIMPEMENTED_YET);

	KInt nAllocSize;
	KaiShape shape = arr1.shape().cut_tail(1);

	KFloat* y_out = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* est_in = arr1.data_ptr();
	KInt* ans_in = arr2.data_ptr();

	kcuda_call(kai_ker_softmax_cross_entropy_with_logits_idx, nrow, (nrow, y_out, est_in, ans_in, nvec));

	KaiArray<KFloat> entropy(y_out, shape, nAllocSize, m_pDevManager);

	return entropy;
}

KaiArray<KFloat> KaiCudaMath::softmax_cross_entropy_with_logits_idx_derv(KaiArray<KFloat> arr1, KaiArray<KInt> arr2) {
	KInt nvec = arr1.axis_size(-1);
	KInt nrow = arr1.total_size() / nvec;

	KaiShape shape = arr1.shape();

	if (arr2.total_size() != nrow) throw KaiException(KERR_UNIMPEMENTED_YET);

	KInt nAllocSize;

	KFloat* y_out = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* est_in = arr1.data_ptr();
	KInt* ans_in = arr2.data_ptr();

	KInt ysize = shape.total_size();

	kcuda_call(kai_ker_softmax_cross_entropy_with_logits_idx_derv, ysize, (ysize, y_out, est_in, ans_in, nvec));

	KaiArray<KFloat> entropy_derv(y_out, shape, nAllocSize, m_pDevManager);

	return entropy_derv;
}

KaiArray<KFloat> KaiCudaMath::equal_col(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.total_size() != arr2.total_size()) throw KaiException(KERR_UNIMPEMENTED_YET);
	if (arr1.shape()[-1] != arr2.shape()[-1]) throw KaiException(KERR_UNIMPEMENTED_YET);

	KInt nAllocSize;
	KInt nvec = arr1.axis_size(-1);
	KInt nrow = arr1.total_size() / nvec;

	KaiShape shape = arr1.shape().cut_tail(1);
	
	KFloat* pd = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	kcuda_call(kai_ker_equal_row, nrow, (nrow, pd, p1, p2, nvec));

	KaiArray<KFloat> dst(pd, shape, nAllocSize, m_pDevManager);

	KString sArr3 = dst.get_core()->desc();

	return dst;
}

KaiArray<KFloat> KaiCudaMath::max_col(KaiArray<KFloat> arr) {
	KInt nAllocSize;
	KInt nvec = arr.axis_size(-1);
	KInt nrow = arr.total_size() / nvec;

	KaiShape shape = arr.shape().cut_tail(1);
	
	KFloat* pd = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* pa = arr.data_ptr();

	kcuda_call(kai_ker_max_row, nrow, (nrow, pd, pa, nvec));

	KaiArray<KFloat> dst(pd, shape, nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::max_col_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr) {
	KInt nAllocSize;
	KInt nvec = farr.axis_size(-1);
	KInt nrow = farr.total_size() / nvec;

	KaiShape shape = farr.shape();

	KFloat* pd = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* pa = farr.data_ptr();
	KFloat* pg = grad_y.data_ptr();

	KInt asize = shape.total_size();

	kcuda_call(kai_ker_max_row_derv, asize, (asize, pd, pa, pg, nvec));

	KaiArray<KFloat> dst(pd, shape, nAllocSize, m_pDevManager);

	return dst;
}

void KaiCudaMath::vstack(KaiArray<KFloat> arr_on, KaiArray<KFloat> arr, KInt nFrom) {
	KInt ncol = arr_on.axis_size(-1);
	KInt nvec = arr.axis_size(-1);
	KInt nrow = arr.total_size() / nvec;

	KFloat* pd = arr_on.data_ptr();
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_vstack, asize, (asize, pd, pa, ncol, nFrom, nvec));
}

KaiArray<KFloat> KaiCudaMath::vstack_grad(KaiArray<KFloat> grad, KInt nStart, KInt nCount) {
	KInt nAllocSize;
	KInt nvec = grad.axis_size(-1);
	KInt nrow = grad.total_size() / nvec;

	KaiShape shape = grad.shape().replace_end(nCount);

	KFloat* pd = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* pg = grad.data_ptr();

	KInt asize = shape.total_size();

	kcuda_call(kai_ker_vstack_derv, asize, (asize, pd, pg, nvec, nStart, nCount));

	KaiArray<KFloat> dst(pd, shape, nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::iou_yolo(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	KaiShape shape1 = arr1.shape();
	KaiShape shape2 = arr2.shape();

	if (shape1.size() != 3) throw KaiException(KERR_BAD_SHAE_FOR_IOU_OP);
	if (shape2.size() != 3) throw KaiException(KERR_BAD_SHAE_FOR_IOU_OP);
	if (shape1[0] != shape2[0]) throw KaiException(KERR_BAD_SHAE_FOR_IOU_OP);
	if (shape1[2] != 4) throw KaiException(KERR_BAD_SHAE_FOR_IOU_OP);
	if (shape2[2] != 4) throw KaiException(KERR_BAD_SHAE_FOR_IOU_OP);

	KaiShape ishape = shape1.replace_end(shape2[1]);

	KInt nAllocSize;
	KInt isize = ishape.total_size();

	KFloat* piou = m_pDevManager->allocFloatMemory(ishape, &nAllocSize);
	KFloat* pa1 = arr1.data_ptr();
	KFloat* pa2 = arr2.data_ptr();

	kcuda_call(kai_ker_iou, isize, (isize, piou, pa1, pa2, shape1[1], shape2[1]));

	KaiArray<KFloat> iou(piou, ishape, nAllocSize, m_pDevManager);

	return iou;
}

KaiArray<KFloat> KaiCudaMath::iou_yolo_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2, KInt nth) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	if (nth != 0) throw KaiException(KERR_UNIMPEMENTED_YET);

	KaiShape xshape = arr1.shape();
	KaiShape shape2 = arr2.shape();

	KaiShape ishape = grad_y.shape();

	KInt nAllocSize;
	KInt xsize = xshape.total_size();

	KFloat* pgiou = grad_y.data_ptr();
	KFloat *pgx = m_pDevManager->allocFloatMemory(xshape, &nAllocSize);
	KFloat* pa2 = arr2.data_ptr();

	kcuda_call(kai_ker_iou_grad, xsize, (xsize, pgx, pgiou, xshape[1], shape2[1]));

	KaiArray<KFloat> gx(pgx, xshape, nAllocSize, m_pDevManager);

	return gx;
	*/
}

KaiArray<KFloat> KaiCudaMath::argmax(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KaiShape ashape = arr.shape();
	KaiShape mshape = ashape.cut_tail(1);

	KInt nvec = ashape[-1];
	KInt nrow = ashape.total_size() / nvec;

	KFloat* pd = m_pDevManager->allocFloatMemory(mshape, &nAllocSize);
	KFloat* pa = arr.data_ptr();

	kcuda_call(kai_ker_argmax, nrow, (nrow, pd, pa, nvec));

	KaiArray<KFloat> dst(pd, mshape, nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::max(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KaiShape ashape = arr.shape();
	KaiShape mshape = ashape.cut_tail(1);

	KInt nvec = ashape[-1];
	KInt nrow = ashape.total_size() / nvec;

	KFloat* pd = m_pDevManager->allocFloatMemory(mshape, &nAllocSize);
	KFloat* pa = arr.data_ptr();

	kcuda_call(kai_ker_maxcol, nrow, (nrow, pd, pa, nvec));

	KaiArray<KFloat> dst(pd, mshape, nAllocSize, m_pDevManager);

	return dst;
}

/*
KaiArray<KFloat> KaiCudaMath::add(KaiArray<KFloat> arr, KFloat term) {
	if (arr.shape().size() == 0) return arr;

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_add, asize, (asize, pd, pa, term));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::sub(KaiArray<KFloat> arr, KFloat term) {
	if (arr.shape().size() == 0) return arr;

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_sub, asize, (asize, pd, pa, term));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::mul(KaiArray<KFloat> arr, KFloat term) {
	if (arr.shape().size() == 0) return arr;

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_mul, asize, (asize, pd, pa, term));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::div(KaiArray<KFloat> arr, KFloat term) {
	if (arr.shape().size() == 0) return arr;

	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_div, asize, (asize, pd, pa, term));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}
*/

void KaiCudaMath::mul_on(KaiArray<KFloat> arr, KFloat term) {
	KFloat* pa = arr.data_ptr();
	KInt asize = arr.total_size();

	kcuda_call(kai_ker_mul, asize, (asize, pa, pa, term));
}

KaiArray<KFloat> KaiCudaMath::sign(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_sign, asize, (asize, pd, pa));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::square(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_square, asize, (asize, pd, pa));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::sqrt(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_sqrt, asize, (asize, pd, pa));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::log(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_log, asize, (asize, pd, pa));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::exp(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_exp, asize, (asize, pd, pa));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::sum(KaiArray<KFloat> arr) {
	KInt nAllocSize1, nAllocSize2;

	KaiShape scalarShape{ 1 };

	KFloat* pd = m_pDevManager->allocFloatMemory(scalarShape, &nAllocSize1);
	KFloat* pb = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize2);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	cudaMemcpy(pb, pa, asize * sizeof(KFloat), cudaMemcpyDeviceToDevice);

	for (KInt range = KAI_ADD_RANGE, ssize = asize; ssize > 1; range *= KAI_ADD_RANGE) {
		ssize = (ssize + KAI_ADD_RANGE - 1) / KAI_ADD_RANGE;
		kcuda_call(kai_ker_sum, ssize, (ssize, pb, asize, range));
	}

	cudaMemcpy(pd, pb, 1 * sizeof(KFloat), cudaMemcpyDeviceToDevice);

	m_pDevManager->freeMemory(pb, nAllocSize2);

	KaiArray<KFloat> dst(pd, scalarShape, nAllocSize1, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::sum_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr) {
	if (grad_y.total_size() != 1) throw KaiException(KERR_UNIMPEMENTED_YET);
	KFloat fGrad = fetch(grad_y);
	KaiArray<KFloat> grad = ones(farr.shape(), fGrad);
	return grad;
}

KaiArray<KFloat> KaiCudaMath::mean(KaiArray<KFloat> arr) {
	KInt nAllocSize1, nAllocSize2;

	KaiShape scalarShape{ 1 };

	KFloat* pd = m_pDevManager->allocFloatMemory(scalarShape, &nAllocSize1);
	KFloat* pb = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize2);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	cudaMemcpy(pb, pa, asize * sizeof(KFloat), cudaMemcpyDeviceToDevice);

	for (KInt range = KAI_ADD_RANGE, ssize = asize; ssize > 1; range *= KAI_ADD_RANGE) {
		ssize = (ssize + KAI_ADD_RANGE - 1) / KAI_ADD_RANGE;
		kcuda_call(kai_ker_sum, ssize, (ssize, pb, asize, range));
	}

	kcuda_call(kai_ker_mul_scalar_on, 1, (1, pb, 1.0f / (KFloat)asize));
	cudaMemcpy(pd, pb, 1 * sizeof(KFloat), cudaMemcpyDeviceToDevice);

	m_pDevManager->freeMemory(pb, nAllocSize2);

	KaiArray<KFloat> dst(pd, scalarShape, nAllocSize1, m_pDevManager);

	return dst;
}

KInt KaiCudaMath::fetch(KInt* arr, KInt nIndex) {
	KInt value;
	cudaMemcpy(&value, arr + nIndex, 1 * sizeof(KInt), cudaMemcpyDeviceToHost);
	return value;
}

KFloat KaiCudaMath::fetch(KFloat* arr, KInt nIndex) {
	KFloat value;
	cudaMemcpy(&value, arr + nIndex, 1 * sizeof(KFloat), cudaMemcpyDeviceToHost);
	return value;
}

KFloat KaiCudaMath::fetch(KaiArray<KFloat> arr, KInt nIndex) {
	KInt size = arr.total_size();

	if (nIndex < 0 || nIndex >= size) throw KaiException(KERR_BAD_INDEX_ON_FARRAY_FETCH);

	KFloat* ap = arr.data_ptr();
	KFloat value;

	cudaMemcpy(&value, ap + nIndex, 1 * sizeof(KFloat), cudaMemcpyDeviceToHost);

	return value;
}

KInt KaiCudaMath::fetch(KaiArray<KInt> arr, KInt nIndex) {
	KInt size = arr.total_size();

	if (nIndex < 0 || nIndex >= size) throw KaiException(KERR_BAD_INDEX_ON_FARRAY_FETCH);

	KInt* ap = arr.data_ptr();
	KInt value;

	cudaMemcpy(&value, ap + nIndex, 1 * sizeof(KInt), cudaMemcpyDeviceToHost);

	return value;
}

KaiArray<KFloat> KaiCudaMath::sigmoid(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();

	kcuda_call(kai_ker_sigmoid, asize, (asize, pd, pa));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::sigmoid_derv_grad(KaiArray<KFloat> gsig, KaiArray<KFloat> sig) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(sig.shape(), &nAllocSize);
	KFloat* pa = gsig.data_ptr();
	KFloat* pb = sig.data_ptr();

	KInt ssize = sig.total_size();

	kcuda_call(kai_ker_sigmoid_derv, ssize, (ssize, pd, pa, pb));

	KaiArray<KFloat> dst(pd, sig.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::softmax(KaiArray<KFloat> arr) {
	KInt nAllocSize;

	KFloat* pd = m_pDevManager->allocFloatMemory(arr.shape(), &nAllocSize);
	KFloat* pa = arr.data_ptr();

	KInt asize = arr.total_size();
	KInt nvec = arr.axis_size(-1);

	kcuda_call(kai_ker_softmax, asize, (asize, pd, pa, nvec));

	KaiArray<KFloat> dst(pd, arr.shape(), nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::softmax_derv(KaiArray<KFloat> gyarr, KaiArray<KFloat> yarr) {
	KaiShape xshape = gyarr.shape();

	KInt nAllocSize;

	KFloat* gx_out = m_pDevManager->allocFloatMemory(xshape, &nAllocSize);
	KFloat* gy_in = gyarr.data_ptr();
	KFloat* y_in = yarr.data_ptr();

	KInt xsize = xshape.total_size();
	KInt nvec = xshape[-1];

	kcuda_call(kai_ker_softmax_derv, xsize, (xsize, gx_out, gy_in, y_in, nvec));

	KaiArray<KFloat> gxarr(gx_out, xshape, nAllocSize, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::eval_adam_delta(KaiArray<KFloat> grad, KaiArray<KFloat> pm_s, KaiArray<KFloat> pm_t, KFloat pm_n, KFloat ro1, KFloat ro2, KFloat epsilon) {
	KInt nAllocSize;

	KaiShape shape = grad.shape();

	KFloat* pd = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* pg = grad.data_ptr();
	KFloat* ps = pm_s.data_ptr();
	KFloat* pt = pm_t.data_ptr();

	KInt gsize = grad.total_size();

	kcuda_call(kai_ker_eval_adam_delta, gsize, (gsize, pd, pg, ps, pt, pm_n, ro1, ro2, epsilon));

	KaiArray<KFloat> dst(pd, shape, nAllocSize, m_pDevManager);

	return dst;
}

KaiArray<KFloat> KaiCudaMath::apply_decay(KaiArray<KFloat> pm, KaiArray<KFloat> grad, KFloat l2_decay, KFloat l1_decay) {
	KInt nAllocSize;

	KaiShape shape = grad.shape();

	KFloat* pd = m_pDevManager->allocFloatMemory(shape, &nAllocSize);
	KFloat* pp = pm.data_ptr();
	KFloat* pg = grad.data_ptr();

	KInt gsize = grad.total_size();

	kcuda_call(kai_ker_apply_decay, gsize, (gsize, pd, pp, pg, l2_decay, l1_decay));

	KaiArray<KFloat> dst(pd, shape, nAllocSize, m_pDevManager);

	return dst;
}

KaiList KaiCudaMath::to_host(KaiList list) {
	KaiList host_list;

	for (auto& it : list) {
		host_list.push_back(m_to_host(it));
	}

	return host_list;
}

KaiDict KaiCudaMath::to_host(KaiDict dict) {
	KaiDict host_dict;

	for (auto& it : dict) {
		host_dict[it.first] = m_to_host(it.second);
	}

	return host_dict;
}

KaiValue KaiCudaMath::m_to_host(KaiValue value) {
	KaiArray<KFloat> farr;
	KaiArray<KInt> narr;

	switch (value.type()) {
	case Ken_value_type::object:
		switch (((KHObject)value)->get_type()) {
		case Ken_object_type::farray:
			farr = FARRAY(value);
			farr = to_host(farr);
			return farr.get_core();
		case Ken_object_type::narray:
			narr = NARRAY(value);
			narr = to_host(narr);
			return narr.get_core();
		default:
			return value;
		}
	case Ken_value_type::dict:
		return to_host((KaiDict)value);
	case Ken_value_type::list:
		return to_host((KaiList)value);
	default:
		return value;
	}
}

KFloat KaiCudaMath::mean(KaiList list) {
	KFloat count = (KFloat)list.size();
	return sum(list) / count;
}

KFloat KaiCudaMath::sum(KaiList list) {
	KFloat count = (KFloat)list.size();
	KFloat sum = 0;

	for (auto& it : list) {
		sum += (KFloat)it;
	}

	return sum;
}


KaiArray<KFloat> KaiCudaMath::convolution(KaiArray<KFloat> xarr, KaiArray<KFloat> kernel) {
	KaiShape xshape = xarr.shape();
	KaiShape kshape = kernel.shape();

	if(xshape.size() < 4) throw KaiException(KERR_BAD_SHAPE_FOR_CONVOLUTION);

	KInt xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
	KInt kh = kshape[0], kw = kshape[1], ychn = kshape[3];

	KaiShape yshape = xshape.replace_end(ychn);
	KaiShape cshape = yshape.append(xchn);

	KInt xsize = xshape.total_size();
	KInt ysize = yshape.total_size();
	KInt csize = cshape.total_size();

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_k = kernel.data_ptr();

	KInt nyAlloc, ncAlloc;

	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nyAlloc);
	KFloat* cuda_c = m_pDevManager->allocFloatMemory(cshape, &ncAlloc);

	kcuda_call(kai_ker_conv_kernel, csize, (csize, cuda_c, cuda_x, cuda_k, xh, xw, kh, kw, xchn, ychn));
	kcuda_call(kai_ker_conv_sum, ysize, (ysize, cuda_y, cuda_c, xchn));

	m_pDevManager->freeMemory(cuda_c, ncAlloc);

	KaiArray<KFloat> yarr(cuda_y, yshape, nyAlloc, m_pDevManager);

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::convolution_derv_x(KaiArray<KFloat> gyarr, KaiArray<KFloat> kernel) {
	KaiShape yshape = gyarr.shape();
	KaiShape kshape = kernel.shape();

	if (yshape.size() < 4) throw KaiException(KERR_BAD_SHAPE_FOR_CONVOLUTION);

	KInt yh = yshape[-3], yw = yshape[-2], ychn = yshape[-1];
	KInt kh = kshape[0], kw = kshape[1], xchn = kshape[2];
	KInt xh = yh, xw = yw;

	KaiShape xshape = yshape.replace_end(xchn);
	KaiShape cshape = xshape.append(ychn);

	KInt xsize = xshape.total_size();
	KInt ysize = yshape.total_size();
	KInt csize = cshape.total_size();

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_k = kernel.data_ptr();

	KInt nxAlloc, ncAlloc;

	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nxAlloc);
	KFloat* cuda_c = m_pDevManager->allocFloatMemory(cshape, &ncAlloc);

	kcuda_call(kai_ker_conv_derv_x_kernel, csize, (csize, cuda_c, cuda_gy, cuda_k, xh, xw, kh, kw, xchn, ychn));
	kcuda_call(kai_ker_conv_derv_x_sum, xsize, (xsize, cuda_gx, cuda_c, ychn));

	m_pDevManager->freeMemory(cuda_c, ncAlloc);

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nxAlloc, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::convolution_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> xarr, KaiShape kshape) {
	KaiShape xshape = xarr.shape();
	KaiShape yshape = gyarr.shape();

	KInt xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
	KInt kh = kshape[0], kw = kshape[1], ychn = kshape[3];
	KInt mb_size = xshape.total_size() / (xh * xw * xchn);
	
	KaiShape sshape = kshape.append(mb_size);
	KaiShape dshape = sshape.append(xh);

	KInt xsize = xshape.total_size();
	KInt ysize = yshape.total_size();
	KInt ksize = kshape.total_size();
	KInt dsize = dshape.total_size();
	KInt ssize = sshape.total_size();

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_x = xarr.data_ptr();

	KInt nkAlloc, ndAlloc;

	KFloat* cuda_gk = m_pDevManager->allocFloatMemory(kshape, &nkAlloc);
	KFloat* cuda_d = m_pDevManager->allocFloatMemory(dshape, &ndAlloc);

	kcuda_call(kai_ker_conv_derv_kw_x, dsize, (dsize, cuda_d, cuda_gy, cuda_x, mb_size, xh, xw, kh, kw, xchn, ychn));
	kcuda_call(kai_ker_conv_derv_kw_sum1, ssize, (ssize, cuda_d, xw));
	kcuda_call(kai_ker_conv_derv_kw_sum2, ksize, (ksize, cuda_gk, cuda_d, mb_size, xw));

	m_pDevManager->freeMemory(cuda_d, ndAlloc);

	KaiArray<KFloat> gkarr(cuda_gk, kshape, nkAlloc, m_pDevManager);

	return gkarr;
}

KaiArray<KFloat> KaiCudaMath::subrange(KaiArray<KFloat> xarr, KInt nth_ax, KInt nFrom, KInt nCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiCudaMath::subrange_derv(KaiArray<KFloat> garr, KInt nth_ax, KInt nFrom, KInt nCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiCudaMath::stride(KaiArray<KFloat> xarr, KaiShape stride) {
	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.copy();

	KInt xh = xshape[-3], xw = xshape[-2], chn = xshape[-1];
	KInt sh = stride[0], sw = stride[1];

	KInt yh = (xh + sh / 2) / sh;
	KInt yw = (xw + sw / 2) / sw;

	yshape[-3] = yh;
	yshape[-2] = yw;

	KFloat* cuda_x = xarr.data_ptr();

	KInt nyAlloc;

	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nyAlloc);

	KInt ysize = yshape.total_size();

	kcuda_call(kai_ker_stride, ysize, (ysize, cuda_y, cuda_x, xh, xw, yh, yw, chn, sh, sw));

	KaiArray<KFloat> yarr(cuda_y, yshape, nyAlloc, m_pDevManager);

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::stride_derv(KaiArray<KFloat> gyarr, KaiShape stride, KaiShape xshape) {
	KaiShape yshape = gyarr.shape().copy();

	KInt xh = xshape[-3], xw = xshape[-2], chn = xshape[-1];
	KInt yh = yshape[-3], yw = yshape[-2];
	KInt sh = stride[0], sw = stride[1];

	KInt nAlloc;

	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);
	KFloat* cuda_gy = gyarr.data_ptr();

	KInt xsize = xshape.total_size();

	kcuda_call(kai_ker_stride_derv, xsize, (xsize, cuda_gx, cuda_gy, xh, xw, yh, yw, chn, sh, sw));

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nAlloc, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::max_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pMaxMap, KaiShape kshape) {
	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape;

	assert(xshape.size() >= 4);

	KInt xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
	KInt kh = kshape[0], kw = kshape[1], ychn = kshape[3];

	KInt ysize = yshape.total_size();

	KFloat* cuda_x = xarr.data_ptr();

	KInt nyAlloc, nmAlloc;

	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nyAlloc);
	KInt* cuda_m = m_pDevManager->allocIntMemory(yshape, &nmAlloc);

	kcuda_call(kai_ker_max, ysize, (ysize, cuda_y, cuda_m, cuda_x, xh, xw, xchn, kh, kw));

	KaiArray<KFloat> yarr(cuda_y, yshape, nyAlloc, m_pDevManager);
	*pMaxMap = KaiArray<KInt>(cuda_m, yshape, nmAlloc, m_pDevManager);

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::max_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> maxMap, KaiShape kshape) {
	KaiShape yshape = gyarr.shape();
	KaiShape xshape = yshape;

	KInt xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
	KInt kh = kshape[0], kw = kshape[1];

	KInt xsize = xshape.total_size();

	KFloat* cuda_gy = gyarr.data_ptr();
	KInt* cuda_n = maxMap.data_ptr();

	KInt nxAlloc;

	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nxAlloc);

	kcuda_call(kai_ker_max_derv, xsize, (xsize, cuda_gx, cuda_n, cuda_gy, xh, xw, xchn, kh, kw));

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nxAlloc, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::avg_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pAvgCnt, KaiShape kshape) {
	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape;

	assert(xshape.size() >= 4);

	KInt xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
	KInt kh = kshape[0], kw = kshape[1], ychn = kshape[3];

	KInt ysize = yshape.total_size();

	KFloat* cuda_x = xarr.data_ptr();

	KInt nyAlloc, nmAlloc;

	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nyAlloc);
	KInt* cuda_m = m_pDevManager->allocIntMemory(yshape, &nmAlloc);

	kcuda_call(kai_ker_avg, ysize, (ysize, cuda_y, cuda_m, cuda_x, xh, xw, xchn, kh, kw));

	KaiArray<KFloat> yarr(cuda_y, yshape, nyAlloc, m_pDevManager);
	*pAvgCnt = KaiArray<KInt>(cuda_m, yshape, nmAlloc, m_pDevManager);

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::avg_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> avgCnt, KaiShape kshape) {
	KaiShape yshape = gyarr.shape();
	KaiShape xshape = yshape;

	KInt xh = xshape[-3], xw = xshape[-2], xchn = xshape[-1];
	KInt kh = kshape[0], kw = kshape[1];

	KInt xsize = xshape.total_size();

	KFloat* cuda_gy = gyarr.data_ptr();
	KInt* cuda_n = avgCnt.data_ptr();

	KInt nxAlloc;

	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nxAlloc);

	kcuda_call(kai_ker_avg_derv, xsize, (xsize, cuda_gx, cuda_n, cuda_gy, xh, xw, xchn, kh, kw));

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nxAlloc, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::globalavg(KaiArray<KFloat> xarr) {
	KaiShape xshape = xarr.shape();
	KInt xh = xshape[-3], xw = xshape[-2], chn = xshape[-1];

	KaiShape yshape = xshape.cut_tail(3).append(chn);

	KInt nyAlloc;

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nyAlloc);

	KInt ysize = yshape.total_size();

	kcuda_call(kai_ker_globalavg, ysize, (ysize, cuda_y, cuda_x, xh, xw, chn));

	KaiArray<KFloat> yarr(cuda_y, yshape, nyAlloc, m_pDevManager);

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::globalavg_derv(KaiArray<KFloat> garr, KaiShape xshape) {
	KaiShape yshape = garr.shape();
	KInt xh = xshape[-3], xw = xshape[-2], chn = xshape[-1];

	KInt nAlloc;

	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);
	KFloat* cuda_gy = garr.data_ptr();

	KInt xsize = xshape.total_size();

	kcuda_call(kai_ker_globalavg_derv, xsize, (xsize, cuda_gx, cuda_gy, xh, xw, chn));

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nAlloc, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::BNCollectNorm(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KaiArray<KFloat>& var, KFloat momentum, KFloat epsilon) {
	KaiShape xshape = xarr.shape();
	KaiShape bshape = KaiShape { xshape[-1] };

	KInt xsize = xshape.total_size();
	KInt bsize = bshape.total_size();

	KInt nAlloc,nAlloc1, nAlloc2;

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	KFloat* cuda_avg = m_pDevManager->allocFloatMemory(bshape, &nAlloc1);
	KFloat* cuda_var = m_pDevManager->allocFloatMemory(bshape, &nAlloc2);

	KFloat* cuda_mavg = mavg.data_ptr();
	KFloat* cuda_mvar = mvar.data_ptr();

    kcuda_call(kai_ker_bn_collect, bsize, (bsize, cuda_avg, cuda_var, cuda_mavg, cuda_mvar, cuda_x, xsize, momentum));
	kcuda_call(kai_ker_bn_normalize, xsize, (xsize, cuda_y, cuda_x, cuda_avg, cuda_var, bsize, epsilon));

	m_pDevManager->freeMemory(cuda_avg, nAlloc1);

	KaiArray<KFloat> yarr(cuda_y, xshape, nAlloc, m_pDevManager);
	KaiArray<KFloat> temp_var(cuda_var, bshape, nAlloc2, m_pDevManager);

	var = temp_var;

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::BnNormalize(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KFloat epsilon) {
	KaiShape xshape = xarr.shape();
	KaiShape bshape = KaiShape { xshape[-1] };

	KInt xsize = xshape.total_size();
	KInt bsize = bshape.total_size();

	KInt nAlloc;

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	KFloat* cuda_mavg = mavg.data_ptr();
	KFloat* cuda_mvar = mvar.data_ptr();

	kcuda_call(kai_ker_bn_normalize, xsize, (xsize, cuda_y, cuda_x, cuda_mavg, cuda_mvar, bsize, epsilon));

	KaiArray<KFloat> yarr(cuda_y, xshape, nAlloc, m_pDevManager);

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::BnScale(KaiArray<KFloat> xarr, KaiArray<KFloat> scale, KaiArray<KFloat> shift) {
	KaiShape xshape = xarr.shape();
	KaiShape bshape = KaiShape { xshape[-1] };

	KInt xsize = xshape.total_size();
	KInt bsize = bshape.total_size();

	KInt nAlloc;

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	KFloat* cuda_scale = scale.data_ptr();
	KFloat* cuda_shift = shift.data_ptr();

	kcuda_call(kai_ker_bn_rescale, xsize, (xsize, cuda_y, cuda_x, cuda_scale, cuda_shift, bsize));

	KaiArray<KFloat> yarr(cuda_y, xshape, nAlloc, m_pDevManager);

	return yarr;
}

void KaiCudaMath::rescale_derv_pm(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat>* p_grad_scale, KaiArray<KFloat>* p_grad_shift) {
	KaiShape xshape = x.shape();
	KaiShape bshape = KaiShape { xshape[-1] };

	KInt xsize = xshape.total_size();
	KInt bsize = bshape.total_size();

	KInt nAlloc1, nAlloc2;

	KFloat* cuda_x = x.data_ptr();
	KFloat* cuda_gy = garr.data_ptr();

	KFloat* cuda_gscale = m_pDevManager->allocFloatMemory(bshape, &nAlloc1);
	KFloat* cuda_gshift = m_pDevManager->allocFloatMemory(bshape, &nAlloc2);

	kcuda_call(kai_ker_bn_rescale_derv_pm, bsize, (bsize, cuda_gscale, cuda_gshift, cuda_gy, cuda_x, xsize));

	KaiArray<KFloat> grad_scale(cuda_gscale, bshape, nAlloc1, m_pDevManager);
	KaiArray<KFloat> grad_shift(cuda_gshift, bshape, nAlloc2, m_pDevManager);

	*p_grad_scale = grad_scale;
	*p_grad_shift = grad_shift;
}

KaiArray<KFloat> KaiCudaMath::rescale_derv_x(KaiArray<KFloat> garr, KaiArray<KFloat> scale) {
	KaiShape xshape = garr.shape();
	KaiShape bshape = KaiShape { xshape[-1] };

	KInt xsize = xshape.total_size();
	KInt bsize = bshape.total_size();

	KInt nAlloc;

	KFloat* cuda_gy = garr.data_ptr();
	KFloat* cuda_scale = scale.data_ptr();

	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_bn_rescale_derv_x, xsize, (xsize, cuda_gx, cuda_gy, cuda_scale, bsize));

	KaiArray<KFloat> grad_x(cuda_gx, xshape, nAlloc, m_pDevManager);

	return grad_x;
}

KaiArray<KFloat> KaiCudaMath::BnNormDerv(KaiArray<KFloat> garr, KaiArray<KFloat> var, KFloat epsilon) {
	KaiShape xshape = garr.shape();
	KaiShape bshape = KaiShape { xshape[-1] };

	KInt xsize = xshape.total_size();
	KInt bsize = bshape.total_size();

	KFloat* cuda_gx = garr.data_ptr();
	KFloat* cuda_var = var.data_ptr();

	kcuda_call(kai_ker_bn_norm_derv, xsize, (xsize, cuda_gx, cuda_var, bsize, epsilon));
	
	return garr;
}

void KaiCudaMath::CopyIntoSlice(KaiArray<KFloat> yarr, KaiArray<KFloat> barr, KInt& chn_from) {
	KaiShape yshape = yarr.shape();
	KaiShape bshape = barr.shape();

	KInt ychn = yshape[-1];
	KInt bchn = bshape[-1];

	KInt ysize = yshape.total_size();
	KInt bsize = bshape.total_size();

	if (ysize * bchn != bsize * ychn || chn_from + bchn > ychn) throw KaiException(KERR_INTERNAL_LOGIC_ERROR);

	KFloat* cuda_y = yarr.data_ptr();
	KFloat* cuda_b = barr.data_ptr();

	kcuda_call(kai_ker_put_branch, bsize, (bsize, cuda_y, cuda_b, ychn, bchn, chn_from));

	chn_from += bchn;
}

KaiArray<KFloat> KaiCudaMath::CopyFromSlice(KaiArray<KFloat> garr, KInt& chn_from, KInt bchn) {
	KaiShape yshape = garr.shape();
	KaiShape bshape = yshape.replace_end(bchn);

	KInt ysize = yshape.total_size();
	KInt bsize = bshape.total_size();

	KInt ychn = yshape[-1];

	KInt nAlloc;

	KFloat* cuda_gy = garr.data_ptr();
	KFloat* cuda_gb = m_pDevManager->allocFloatMemory(bshape, &nAlloc);

	kcuda_call(kai_ker_get_branch, bsize, (bsize, cuda_gb, cuda_gy, ychn, bchn, chn_from));

	KaiArray<KFloat> barr(cuda_gb, bshape, nAlloc, m_pDevManager);

	chn_from += bchn;

	return barr;
}

KaiArray<KFloat> KaiCudaMath::random_bernoulli(KaiShape xshape, KFloat one_ratio) {
	KaiArray<KFloat> yhost = KaiHostMath::random_bernoulli(xshape, one_ratio);
	return to_cuda(yhost);
}

KaiArray<KFloat> KaiCudaMath::dropout(KaiArray<KFloat> xarr, KaiArray<KFloat> mask, KFloat keep_raio) {
	KaiShape xshape = xarr.shape();

	KInt xsize = xshape.total_size();

	KInt nAlloc;

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_m = mask.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_dropout, xsize, (xsize, cuda_y, cuda_x, cuda_m, keep_raio));

	KaiArray<KFloat> yarr(cuda_y, xshape, nAlloc, m_pDevManager);

	return yarr;
}

KaiArray<KFloat> KaiCudaMath::dropout_derv(KaiArray<KFloat> gyarr, KaiArray<KFloat> mask, KFloat keep_raio) {
	KaiShape yshape = gyarr.shape();

	KInt ysize = yshape.total_size();

	KInt nAlloc;

	KFloat* cuda_y = gyarr.data_ptr();
	KFloat* cuda_m = mask.data_ptr();
	KFloat* cuda_x = m_pDevManager->allocFloatMemory(yshape, &nAlloc);

	kcuda_call(kai_ker_dropout_derv, ysize, (ysize, cuda_x, cuda_y, cuda_m, keep_raio));

	KaiArray<KFloat> gxarr(cuda_x, yshape, nAlloc, m_pDevManager);

	return gxarr;
}

void KaiCudaMath::residual_add(KaiArray<KFloat> yarr, KaiArray<KFloat> xarr) {
	KaiShape yshape = yarr.shape();
	KaiShape xshape = xarr.shape();

	KInt ysize = yshape.total_size();

	KInt ychn = yshape[-1];
	KInt xchn = xshape[-1];

	KFloat* cuda_y = yarr.data_ptr();
	KFloat* cuda_x = xarr.data_ptr();

	kcuda_call(kai_ker_tile_chn, ysize, (ysize, cuda_y, cuda_x, ychn, xchn));
}

KaiArray<KFloat> KaiCudaMath::residual_add_derv(KaiArray<KFloat> gyarr, KInt xchn) {
	KaiShape yshape = gyarr.shape();
	KInt ychn = yshape[-1];

	if (xchn == ychn) return gyarr;

	KaiShape xshape = yshape.replace_end(xchn);
	KInt xsize = xshape.total_size();

	KInt nAlloc;

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_untile_chn, xsize, (xsize, cuda_gx, cuda_gy, ychn, xchn));

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nAlloc, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::CombineExtendedInput(KaiArray<KFloat> recurrent, KBool isSeq, KaiArray<KFloat> xarr, KInt nth) {
	KInt mb_size= xarr.shape()[0];
	KInt timesteps = isSeq ? xarr.shape()[-2] : 0;
	KInt timefeats = xarr.shape()[-1];
	KInt recur_size = recurrent.shape()[-1];

	KaiShape eshape{ mb_size, timefeats + recur_size };

	KInt esize = eshape.total_size();

	KInt nAlloc;

	KFloat* cuda_ex_inp = m_pDevManager->allocFloatMemory(eshape, &nAlloc);
	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_rec = recurrent.data_ptr();

	kcuda_call(kai_ker_rnn_combine_ex_inp, esize, (esize, cuda_ex_inp, cuda_x, cuda_rec, timesteps, timefeats, recur_size, isSeq, nth));

	KaiArray<KFloat> ext_input(cuda_ex_inp, eshape, nAlloc, m_pDevManager);

	return ext_input;
}

KaiArray<KFloat> KaiCudaMath::SplitExtendedInputGrad(KaiArray<KFloat> g_exp_input, KBool isSeq, KaiArray<KFloat> g_x, KInt nth) {
	KInt mb_size = g_exp_input.shape()[0];
	KInt timesteps = isSeq ? g_x.shape()[-2] : 0;
	KInt timefeats = g_x.shape()[-1];
	KInt recur_size = g_exp_input.shape()[-1] - timefeats;

	KaiShape eshape = g_exp_input.shape();
	KaiShape rshape{ mb_size , recur_size };

	KInt esize = eshape.total_size();

	KInt nAlloc;

	KFloat* cuda_gex_inp = g_exp_input.data_ptr();
	KFloat* cuda_gx = g_x.data_ptr();
	KFloat* cuda_grec = m_pDevManager->allocFloatMemory(rshape, &nAlloc);

	kcuda_call(kai_ker_rnn_split_ex_inp, esize, (esize, cuda_gx, cuda_grec, cuda_gex_inp, timesteps, timefeats, recur_size, isSeq, nth));

	KaiArray<KFloat> grec(cuda_grec, rshape, nAlloc, m_pDevManager);

	return grec;
}

void KaiCudaMath::CopyIntoTimeSlice(KaiArray<KFloat> whole, KaiArray<KFloat> slice, KInt nth) {
	KInt wsize = whole.total_size();
	KInt ssize = slice.total_size();

	KInt timesteps = wsize / ssize;
	KInt recur_size = slice.shape()[-1];

	KFloat* cuda_w = whole.data_ptr();
	KFloat* cuda_s = slice.data_ptr();

	kcuda_call(kai_ker_rnn_fill_output_slice, ssize, (ssize, cuda_w, cuda_s, timesteps, recur_size, nth));
}

void KaiCudaMath::add_time_slice_on_dest(KaiArray<KFloat> dest, KaiArray<KFloat> whole, KInt nth) {
	KInt wsize = whole.total_size();
	KInt dsize = dest.total_size();

	KInt timesteps = wsize / dsize;
	KInt recur_size = dest.shape()[-1];

	KFloat* cuda_w = whole.data_ptr();
	KFloat* cuda_d = dest.data_ptr();

	kcuda_call(kai_ker_rnn_add_time_slice, dsize, (dsize, cuda_d, cuda_w, timesteps, recur_size, nth));
}

KaiArray<KFloat> KaiCudaMath::lstm_gates(KaiArray<KFloat> affine) {
	KaiShape ashape = affine.shape();

	KInt asize = ashape.total_size();

	KInt nAlloc;

	KFloat* cuda_a = affine.data_ptr();
	KFloat* cuda_g = m_pDevManager->allocFloatMemory(ashape, &nAlloc);

	kcuda_call(kai_ker_lstm_gate, asize, (asize, cuda_g, cuda_a));

	KaiArray<KFloat> gates(cuda_g, ashape, nAlloc, m_pDevManager);

	return gates;
}

KaiArray<KFloat> KaiCudaMath::lstm_proc(KaiArray<KFloat> gates, KaiArray<KFloat>& state, KBool use_state) {
	KaiShape sshape = state.shape();

	KInt ssize = sshape.total_size();

	KInt nAlloc1, nAlloc2;

	KFloat* cuda_g = gates.data_ptr();
	KFloat* cuda_s = state.data_ptr();
	KFloat* cuda_r = m_pDevManager->allocFloatMemory(sshape, &nAlloc1);
	KFloat* cuda_n = m_pDevManager->allocFloatMemory(sshape, &nAlloc2);

	kcuda_call(kai_ker_lstm_proc, ssize, (ssize, cuda_r, cuda_n, cuda_s, cuda_g));

	KaiArray<KFloat> recurrent(cuda_r, sshape, nAlloc1, m_pDevManager);
	KaiArray<KFloat> new_state(cuda_n, sshape, nAlloc2, m_pDevManager);

	state = new_state;

	return recurrent;
}

KaiArray<KFloat> KaiCudaMath::lstm_gates_derv(KaiArray<KFloat> g_gates, KaiArray<KFloat> gates) {
	KaiShape gshape = g_gates.shape();

	KInt gsize = gshape.total_size();

	KInt nAlloc;

	KFloat* cuda_gg = g_gates.data_ptr();
	KFloat* cuda_gt = gates.data_ptr();
	KFloat* cuda_ga = m_pDevManager->allocFloatMemory(gshape, &nAlloc);

	kcuda_call(kai_ker_lstm_gate_derv, gsize, (gsize, cuda_ga, cuda_gg, cuda_gt));

	KaiArray<KFloat> g_affine(cuda_ga, gshape, nAlloc, m_pDevManager);

	return g_affine;
}

KaiArray<KFloat> KaiCudaMath::lstm_proc_derv(KaiArray<KFloat>& g_state, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_state, KaiArray<KFloat> post_recur, KBool use_state) {
	KaiShape sshape = g_state.shape();
	KaiShape gshape = gates.shape();

	KInt ssize = sshape.total_size();

	KInt nAlloc1, nAlloc2;

	KFloat* cuda_gs = g_state.data_ptr();
	KFloat* cuda_gr = g_recurrent.data_ptr();
	KFloat* cuda_gt = gates.data_ptr();
	KFloat* cuda_st = pre_state.data_ptr();
	KFloat* cuda_rc = post_recur.data_ptr();
	KFloat* cuda_gg = m_pDevManager->allocFloatMemory(gshape, &nAlloc1);
	KFloat* cuda_gn = m_pDevManager->allocFloatMemory(sshape, &nAlloc2);

	kcuda_call(kai_ker_lstm_proc_derv, ssize, (ssize, cuda_gg, cuda_gn, cuda_gs, cuda_gr, cuda_gt, cuda_st, cuda_rc));

	KaiArray<KFloat> g_gates(cuda_gg, gshape, nAlloc1, m_pDevManager);
	KaiArray<KFloat> g_state_new(cuda_gn, sshape, nAlloc2, m_pDevManager);

	g_state = g_state_new;

	return g_gates;
}

KaiArray<KFloat> KaiCudaMath::gru_combine_extra(KaiArray<KFloat> exp_input, KaiArray<KFloat> gates) {
	KaiShape xshape = exp_input.shape();
	KaiShape gshape = gates.shape();

	KInt xsize = xshape.total_size();
	KInt ext_size = xshape[-1];
	KInt rec_size = gshape[-1] / 2;
	KInt inp_size = ext_size - rec_size;

	KInt nAlloc;

	KFloat* p_input = exp_input.data_ptr();
	KFloat* p_gates = gates.data_ptr();
	KFloat* p_new_input = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_gru_combine_extra, xsize, (xsize, p_new_input, p_input, p_gates, ext_size, inp_size, rec_size));

	KaiArray<KFloat> ext_extra(p_new_input, xshape, nAlloc, m_pDevManager);

	return ext_extra;
}

void KaiCudaMath::gru_combine_extra_derv(KaiArray<KFloat> g_exp_input, KaiArray<KFloat> g_gates, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> exp_input) {
	KaiShape xshape = exp_input.shape();
	KaiShape gshape = gates.shape();

	KInt xsize = xshape.total_size();
	KInt ext_size = xshape[-1];
	KInt rec_size = gshape[-1] / 2;
	KInt inp_size = ext_size - rec_size;

	KFloat* p_ext_in = exp_input.data_ptr();
	KFloat* pg_ext_inout = g_exp_input.data_ptr();
	KFloat* p_gates = gates.data_ptr();
	KFloat* pg_gates = g_gates.data_ptr();

	kcuda_call(kai_ker_gru_combine_extra_derv, xsize, (xsize, pg_gates, pg_ext_inout, p_ext_in, p_gates, ext_size, inp_size, rec_size));
}

KaiArray<KFloat> KaiCudaMath::gru_proc(KaiArray<KFloat> gates, KaiArray<KFloat> recurrent, KaiArray<KFloat> extra_affine) {
	KaiShape rshape = recurrent.shape();

	KInt rsize = rshape.total_size();

	KInt nAlloc;

	KFloat* cuda_r1 = recurrent.data_ptr();
	KFloat* cuda_gt = gates.data_ptr();
	KFloat* cuda_in = extra_affine.data_ptr();
	KFloat* cuda_r2 = m_pDevManager->allocFloatMemory(rshape, &nAlloc);

	kcuda_call(kai_ker_gru_proc, rsize, (rsize, cuda_r2, cuda_r1, cuda_gt, cuda_in));

	KaiArray<KFloat> new_recurrent(cuda_r2, rshape, nAlloc, m_pDevManager);

	return new_recurrent;
}

KaiArray<KFloat> KaiCudaMath::gru_proc_derv(KaiArray<KFloat>& g_gates, KaiArray<KFloat>& g_new_rec, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_recur, KaiArray<KFloat> extra_affine) {
	KaiShape rshape = pre_recur.shape();
	KaiShape gshape = gates.shape();

	KInt rsize = rshape.total_size();

	KInt nAlloc1, nAlloc2, nAlloc3;

	KFloat* p_pre_rec = pre_recur.data_ptr();
	KFloat* p_gates = gates.data_ptr();
	KFloat* p_inp = extra_affine.data_ptr();
	KFloat* pg_rec_in = g_recurrent.data_ptr();
	KFloat* pg_rec_out = m_pDevManager->allocFloatMemory(rshape, &nAlloc1);
	KFloat* pg_gates = m_pDevManager->allocFloatMemory(gshape, &nAlloc2);
	KFloat* pg_affine = m_pDevManager->allocFloatMemory(rshape, &nAlloc3);

	kcuda_call(kai_ker_gru_proc_derv, rsize, (rsize, pg_affine, pg_gates, pg_rec_out, pg_rec_in, p_pre_rec, p_gates, p_inp));

	g_new_rec = KaiArray<KFloat>(pg_rec_out, rshape, nAlloc1, m_pDevManager);
	g_gates = KaiArray<KFloat>(pg_gates, gshape, nAlloc2, m_pDevManager);

	KaiArray<KFloat> g_affine(pg_affine, rshape, nAlloc3, m_pDevManager);

	return g_affine;
}

void KaiCudaMath::add_embed_dict(KaiArray<KFloat> yarr, KaiArray<KInt> tokens, KaiArray<KFloat> wdic, KInt axis) {
	KaiShape xshape = tokens.shape();
	KaiShape yshape = yarr.shape();
	KaiShape dshape = wdic.shape();

	KInt ysize = yshape.total_size();
	KInt dic_kind = xshape[-1];
	KInt vec_size = yshape[-1];

	KFloat* cuda_y = yarr.data_ptr();
	KFloat* cuda_d = wdic.data_ptr();
	KInt* cuda_t = tokens.data_ptr();

	kcuda_call(kai_ker_add_embed_dict, ysize, (ysize, cuda_y, cuda_d, cuda_t, vec_size, dic_kind, axis));
}

#define MAX_PIECE_CNT_FOR_SPLIT 4

KaiList KaiCudaMath::split_array(KaiArray<KFloat> arr, KInt piece_cnt) {
	KaiShape xshape = arr.shape();
	KaiShape pshape = xshape.replace_end(xshape[-1] / piece_cnt);

	KInt vec_size = xshape[-1];
	KInt xsize = xshape.total_size();
	KInt nAlloc;

	if (piece_cnt > MAX_PIECE_CNT_FOR_SPLIT) throw KaiException(KERR_TOO_LARGE_SPLIT_COUNT);
	if (vec_size % piece_cnt != 0) throw KaiException(KERR_INVALID_PIECE_CNT_FOR_SPLIT_ARRAY);

	KFloat* cuda_x = arr.data_ptr();
	KFloat* cuda_p[MAX_PIECE_CNT_FOR_SPLIT];

	for (KInt n = 0; n < piece_cnt; n++) {
		cuda_p[n] = m_pDevManager->allocFloatMemory(pshape, &nAlloc);
	}

	kcuda_call(kai_ker_split_array, xsize, (xsize, cuda_p[0], cuda_p[1], cuda_p[2], cuda_p[3], cuda_x, piece_cnt));

	KaiList result;

	for (KInt n = 0; n < piece_cnt; n++) {
		KaiArray<KFloat> piece(cuda_p[n], pshape, nAlloc, m_pDevManager);
		result.push_back(piece.get_core());
	}

	return result;
}

KaiArray<KFloat> KaiCudaMath::merge_array(KaiList arrs) {
	KInt piece_cnt = arrs.size();

	if (piece_cnt > MAX_PIECE_CNT_FOR_SPLIT) throw KaiException(KERR_TOO_LARGE_SPLIT_COUNT);

	KFloat* cuda_p[MAX_PIECE_CNT_FOR_SPLIT];

	KaiShape pshape;

	for (KInt n = 0; n < piece_cnt; n++) {
		KaiArray<KFloat> piece = FARRAY(arrs[n]);
		cuda_p[n] = piece.data_ptr();
		if (n == 0) pshape = piece.shape();
		else if (pshape != piece.shape()) throw KaiException(KERR_SHAPE_TOO_MERGE_SHAPE_NOT_UNIQUE);
	}

	KaiShape xshape = pshape.replace_end(pshape[-1] * piece_cnt);

	KInt xsize = xshape.total_size();
	KInt nAlloc;

	KFloat* cuda_x = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_merge_array, xsize, (xsize, cuda_x, cuda_p[0], cuda_p[1], cuda_p[2], cuda_p[3], piece_cnt));

	KaiArray<KFloat> result(cuda_x, xshape, nAlloc, m_pDevManager);

	return result;
}

KaiArray<KFloat> KaiCudaMath::multi_head_matmul_qk(KaiArray<KFloat> query, KaiArray<KFloat> key, KInt head_cnt) {
	KaiShape qshape = query.shape();
	KaiShape kshape = key.shape();

	if (qshape != kshape) throw KaiException(KERR_NEED_CODE_MODIFICATION);

	KInt timesteps = qshape[-2];
	KInt vec_size = qshape[-1];
	KInt vec_per_head = vec_size / head_cnt;
	KInt mb_size = qshape.total_size() / (timesteps * vec_size);

	if (vec_size % head_cnt != 0) throw KaiException(KERR_NEED_CODE_MODIFICATION);

	KaiShape rshape{ mb_size, head_cnt, timesteps, timesteps };

	KInt rsize = rshape.total_size();
	KInt nAlloc;

	KFloat* cuda_q = query.data_ptr();
	KFloat* cuda_k = key.data_ptr();
	KFloat* cuda_r = m_pDevManager->allocFloatMemory(rshape, &nAlloc);

	kcuda_call(kai_ker_multi_head_matmul_qk, rsize, (rsize, cuda_r, cuda_q, cuda_k, timesteps, head_cnt, vec_per_head));

	KaiArray<KFloat> result(cuda_r, rshape, nAlloc, m_pDevManager);

	return result;
}

KaiArray<KFloat> KaiCudaMath::multi_head_matmul_qk_derv_q(KaiArray<KFloat> gyarr, KaiArray<KFloat> key, KInt head_cnt) {
	KaiShape kshape = key.shape();
	KaiShape qshape = kshape;

	KInt timesteps = qshape[-2];
	KInt vec_size = qshape[-1];
	KInt vec_per_head = vec_size / head_cnt;

	KInt qsize = qshape.total_size();
	KInt nAlloc;

	KFloat* gy_in = gyarr.data_ptr();
	KFloat* k_in = key.data_ptr();
	KFloat* gq_out = m_pDevManager->allocFloatMemory(qshape, &nAlloc);

	kcuda_call(kai_ker_multi_head_matmul_qk_derv_q, qsize, (qsize, gq_out, gy_in, k_in, timesteps, head_cnt, vec_per_head));

	KaiArray<KFloat> gquery(gq_out, qshape, nAlloc, m_pDevManager);

	return gquery;
}

KaiArray<KFloat> KaiCudaMath::multi_head_matmul_qk_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> query, KInt head_cnt) {
	KaiShape qshape = query.shape();
	KaiShape kshape = qshape;

	KInt timesteps = qshape[-2];
	KInt vec_size = qshape[-1];
	KInt vec_per_head = vec_size / head_cnt;

	KInt ksize = kshape.total_size();
	KInt nAlloc;

	KFloat* gy_in = gyarr.data_ptr();
	KFloat* q_in = query.data_ptr();
	KFloat* gk_out = m_pDevManager->allocFloatMemory(kshape, &nAlloc);

	kcuda_call(kai_ker_multi_head_matmul_qk_dev_k, ksize, (ksize, gk_out, gy_in, q_in, timesteps, head_cnt, vec_per_head));

	KaiArray<KFloat> gkey(gk_out, kshape, nAlloc, m_pDevManager);

	return gkey;
}

KaiArray<KFloat> KaiCudaMath::multi_head_matmul_pv(KaiArray<KFloat> probs, KaiArray<KFloat> value) {
	KaiShape pshape = probs.shape();
	KaiShape vshape = value.shape();

	if (pshape[0] != vshape[0]) throw KaiException(KERR_NEED_CODE_MODIFICATION);
	if (pshape[3] != vshape[1]) throw KaiException(KERR_NEED_CODE_MODIFICATION);
	if (pshape[2] != pshape[3]) throw KaiException(KERR_NEED_CODE_MODIFICATION);

	KInt mb_size = pshape[0];
	KInt head_cnt = pshape[1];
	KInt timesteps = pshape[2];
	KInt vec_size = vshape[-1];
	KInt vec_per_head = vec_size / head_cnt;

	if (vec_size % head_cnt != 0) throw KaiException(KERR_NEED_CODE_MODIFICATION);

	KaiShape rshape{ mb_size, timesteps, vec_size };

	KInt rsize = rshape.total_size();
	KInt nAlloc;

	KFloat* cuda_p = probs.data_ptr();
	KFloat* cuda_v = value.data_ptr();
	KFloat* cuda_r = m_pDevManager->allocFloatMemory(rshape, &nAlloc);

	kcuda_call(kai_ker_multi_head_matmul_pv, rsize, (rsize, cuda_r, cuda_p, cuda_v, timesteps, head_cnt, vec_per_head));

	KaiArray<KFloat> result(cuda_r, rshape, nAlloc, m_pDevManager);

	return result;
}

KaiArray<KFloat> KaiCudaMath::multi_head_matmul_pv_derv_p(KaiArray<KFloat> gyarr, KaiArray<KFloat> value, KInt head_cnt) {
	KaiShape yshape = gyarr.shape();
	KaiShape vshape = value.shape();

	KInt mb_size = yshape[0];
	KInt timesteps = yshape[1];
	KInt vec_size = yshape[2];
	KInt vec_per_head = vec_size / head_cnt;

	if (vec_size % head_cnt != 0) throw KaiException(KERR_NEED_CODE_MODIFICATION);

	KaiShape pshape{ mb_size, head_cnt, timesteps, timesteps };

	KInt psize = pshape.total_size();
	KInt nAlloc;

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_v = value.data_ptr();
	KFloat* cuda_gp = m_pDevManager->allocFloatMemory(pshape, &nAlloc);

	kcuda_call(kai_ker_multi_head_matmul_pv_derv_p, psize, (psize, cuda_gp, cuda_gy, cuda_v, timesteps, head_cnt, vec_per_head));

	KaiArray<KFloat> gprob(cuda_gp, pshape, nAlloc, m_pDevManager);

	return gprob;
}

KaiArray<KFloat> KaiCudaMath::multi_head_matmul_pv_derv_v(KaiArray<KFloat> gyarr, KaiArray<KFloat> probs) {
	KaiShape yshape = gyarr.shape();
	KaiShape pshape = probs.shape();

	KInt mb_size = yshape[0];
	KInt timesteps = yshape[1];
	KInt vec_size = yshape[2];
	KInt head_cnt = pshape[1];
	KInt vec_per_head = vec_size / head_cnt;

	if (vec_size % head_cnt != 0) throw KaiException(KERR_NEED_CODE_MODIFICATION);

	KaiShape vshape{ mb_size, timesteps, vec_size};

	KInt vsize = vshape.total_size();
	KInt nAlloc;

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_p = probs.data_ptr();
	KFloat* cuda_gv = m_pDevManager->allocFloatMemory(vshape, &nAlloc);

	kcuda_call(kai_ker_multi_head_matmul_pv_derv_v, vsize, (vsize, cuda_gv, cuda_gy, cuda_p, timesteps, head_cnt, vec_per_head));

	KaiArray<KFloat> gvalue(cuda_gv, vshape, nAlloc, m_pDevManager);

	return gvalue;
}

KaiArray<KFloat> KaiCudaMath::extract(KaiArray<KFloat> xarr, KInt axis, KInt index, KInt count, KBool reduce_seq) {
	if (count == 0) count = 1;

	KaiShape xshape = xarr.shape();
	KaiShape yshape = xshape.replace_nth(axis, count);

	if (count == 1 && reduce_seq) yshape = yshape.remove_nth(axis);

	KInt ysize = yshape.total_size();
	KInt nAlloc;
	KInt nProd = 1;

	for (KInt n = axis + 1; n < xshape.size(); n++) nProd *= xshape[n];

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nAlloc);

	kcuda_call(kai_ker_extract, ysize, (ysize, cuda_y, cuda_x, xshape[axis], index, count, nProd));

	KaiArray<KFloat> extracted(cuda_y, yshape, nAlloc, m_pDevManager);

	return extracted;
}

KaiArray<KFloat> KaiCudaMath::extract_derv(KaiArray<KFloat> gyarr, KaiShape xshape, KInt axis, KInt index, KInt count, KBool reduce_seq) {
	if (count == 0) count = 1;

	KaiShape yshape = gyarr.shape();

	KInt ysize = yshape.total_size();
	KInt nAlloc;
	KInt nProd = 1;

	for (KInt n = axis + 1; n < xshape.size(); n++) nProd *= xshape[n];

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_extract_derv, ysize, (ysize, cuda_gx, cuda_gy, xshape[axis], index, count, nProd));

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nAlloc, m_pDevManager);

	return gxarr;
}

KaiArray<KFloat> KaiCudaMath::select(KaiArray<KFloat> xarr, KaiArray<KInt> selector, KaiShape vshape) {
	if (xarr.total_size() % vshape.total_size() != 0) throw KaiException(KERR_BAD_SHAPE_PAIR_FOR_SELECT_OP);

	KInt mb_size = selector.total_size();

	KaiShape yshape = vshape.insert_head(mb_size);

	KInt ysize = yshape.total_size();
	KInt vsize = vshape.total_size();
	KInt nAlloc;

	KInt* cuda_s = selector.data_ptr();
	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nAlloc);

	kcuda_call(kai_ker_filter, ysize, (ysize, cuda_y, cuda_x, cuda_s, vsize));

	KaiArray<KFloat> selected(cuda_y, yshape, nAlloc, m_pDevManager);

	return selected;
}

KaiArray<KFloat> KaiCudaMath::select_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> selector, KaiShape xshape, KaiShape vshape) {
	if (xshape.total_size() % vshape.total_size() != 0) throw KaiException(KERR_BAD_SHAPE_PAIR_FOR_SELECT_OP);

	KInt mb_size = selector.total_size();
	KaiShape yshape = vshape.insert_head(mb_size);

	if (yshape != gyarr.shape()) throw KaiException(KERR_BAD_SHAPE_PAIR_FOR_SELECT_OP);

	KInt ysize = yshape.total_size();
	KInt vsize = vshape.total_size();
	KInt nAlloc;

	KInt* cuda_s = selector.data_ptr();
	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_filter_derv, ysize, (ysize, cuda_gx, cuda_gy, cuda_s, vsize));

	KaiArray<KFloat> gxarr(cuda_gx, xshape, nAlloc, m_pDevManager);

	return gxarr;
}

void KaiCudaMath::update_dic_weight_sgd(KaiArray<KFloat> weight, KaiArray<KFloat> grad, KaiArray<KInt> tokens, KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
	KInt gsize = grad.total_size();

	KInt dic_cnt = tokens.shape()[-1];
	KInt dic_size = weight.shape()[0];
	KInt vec_size = weight.shape()[1];

	KFloat* cuda_w = weight.data_ptr();
	KFloat* cuda_g = grad.data_ptr();
	KInt* cuda_t = tokens.data_ptr();

	kcuda_call(kai_kernel_sgd_update_embed, gsize, (gsize, cuda_w, cuda_g, cuda_t, dic_cnt, nth, dic_size, vec_size, learning_rate, l2_decay, l1_decay));
}

void KaiCudaMath::update_dic_weight_adam(KaiArray<KFloat> weight, KaiArray<KFloat> s, KaiArray<KFloat> t, KaiArray<KFloat> n, KaiArray<KFloat> grad, KaiArray<KInt> tokens,
	KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay, KFloat ro1, KFloat ro2, KFloat epsilon) {
	KInt gsize = grad.total_size();

	KInt dic_cnt = tokens.shape()[-1];
	KInt dic_size = weight.shape()[0];
	KInt vec_size = weight.shape()[1];

	KFloat* cuda_w = weight.data_ptr();
	KFloat* cuda_s = s.data_ptr();
	KFloat* cuda_t = t.data_ptr();
	KFloat* cuda_n = n.data_ptr();
	KFloat* cuda_g = grad.data_ptr();
	KInt* cuda_tk = tokens.data_ptr();

	kcuda_call(kai_kernel_adam_update_embed, gsize, (gsize, cuda_w, cuda_s, cuda_t, cuda_n, cuda_g, cuda_tk,
		dic_cnt, nth, dic_size, vec_size, learning_rate, l2_decay, l1_decay, ro1, ro2, epsilon));
}

KaiArray<KFloat> KaiCudaMath::expand(KaiArray<KFloat> xarr, KaiShape ratio) {
	KaiShape xshape = xarr.shape();
	if (xshape.size() != 4) throw KaiException(KERR_BAD_DIMENSION_FOR_EXPAND_LAYER);
	
	KInt mb_size = xshape[0], xheight = xshape[1], xwidth = xshape[2], chn = xshape[3];
	KInt hratio = ratio[0], wratio = ratio[1];
	KInt yheight = xheight * hratio, ywidth = xwidth * wratio;

	KaiShape yshape { mb_size, yheight, ywidth, chn };

	KInt ysize = yshape.total_size();
	KInt nAlloc;

	KFloat* cuda_x = xarr.data_ptr();
	KFloat* cuda_y = m_pDevManager->allocFloatMemory(yshape, &nAlloc);

	kcuda_call(kai_ker_expand, ysize, (ysize, cuda_y, cuda_x, yheight, ywidth, chn, hratio, wratio));

	KaiArray<KFloat> expanded(cuda_y, yshape, nAlloc, m_pDevManager);

	return expanded;
}

KaiArray<KFloat> KaiCudaMath::expand_derv(KaiArray<KFloat> gyarr, KaiShape ratio) {
	KaiShape yshape = gyarr.shape();

	KInt mb_size = yshape[0], yheight = yshape[1], ywidth = yshape[2], chn = yshape[3];
	KInt hratio = ratio[0], wratio = ratio[1];
	KInt xheight = yheight / hratio, xwidth = ywidth / wratio;

	KaiShape xshape{ mb_size, xheight, xwidth, chn };

	KInt xsize = xshape.total_size();
	KInt nAlloc;

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	kcuda_call(kai_ker_expand_derv, xsize, (xsize, cuda_gx, cuda_gy, xheight, xwidth, chn, hratio, wratio));

	KaiArray<KFloat> gx(cuda_gx, xshape, nAlloc, m_pDevManager);

	return gx;
}

KInt KaiCudaMath::stack_on(KaiArray<KFloat> dest, KaiArray<KFloat> src, KInt tail_size, KInt nFrom, KInt nTo) {
	KInt mb_size = dest.shape()[0];

	KInt ssize = src.total_size();
	KInt bsize = ssize / mb_size;
	KInt rsize = dest.total_size() / mb_size;
	KInt copy_cnt = bsize / tail_size;

	assert(nFrom + copy_cnt <= nTo);

	KFloat* cuda_d = dest.data_ptr();
	KFloat* cuda_s = src.data_ptr();

	kcuda_call(kai_ker_stack_on, ssize, (ssize, cuda_d, cuda_s, bsize, rsize, tail_size, nFrom));

	return nFrom + copy_cnt;
}

KaiArray<KFloat> KaiCudaMath::stack_on_grad(KaiArray<KFloat> gyarr, KaiShape xshape, KInt tail_size, KInt& nFrom, KInt nTo) {
	KInt mb_size = gyarr.shape()[0];

	KInt ysize = gyarr.total_size();
	KInt rsize = ysize / mb_size;
	KInt bsize = xshape.total_size() / mb_size;
	KInt copy_cnt = bsize / tail_size;

	assert(nFrom + copy_cnt <= nTo);

	KInt nAlloc;

	KFloat* cuda_gy = gyarr.data_ptr();
	KFloat* cuda_gx = m_pDevManager->allocFloatMemory(xshape, &nAlloc);

	KInt xsize = xshape.total_size();

	kcuda_call(kai_ker_stack_on_grad, xsize, (xsize, cuda_gx, cuda_gy, bsize, rsize, tail_size, nFrom));

	nFrom += copy_cnt;

	KaiArray<KFloat> gx(cuda_gx, xshape, nAlloc, m_pDevManager);

	return gx;
}

KaiArray<KFloat> KaiCudaMath::get_subvector(KaiArray<KFloat> arr, KInt nStart, KInt nCount) {
	KaiShape ashape = arr.shape();
	KaiShape sshape = ashape.replace_end(nCount);

	KInt vec_size = ashape[-1];
	KInt ssize = sshape.total_size();

	KInt nAlloc;

	KFloat* cuda_a = arr.data_ptr();
	KFloat* cuda_s = m_pDevManager->allocFloatMemory(sshape, &nAlloc);

	kcuda_call(kai_ker_get_subvector, ssize, (ssize, cuda_s, cuda_a, vec_size, nStart, nCount));

	KaiArray<KFloat> subvector(cuda_s, sshape, nAlloc, m_pDevManager);

	return subvector;
}

void KaiCudaMath::get_subvector_derv_acc(KaiArray<KFloat> grad, KaiArray<KFloat> grad_subvec, KInt nStart, KInt nCount) {
	KaiShape ashape = grad.shape();
	KaiShape sshape = ashape.replace_end(nCount);

	assert(sshape == grad_subvec.shape());

	KInt vec_size = ashape[-1];
	KInt ssize = sshape.total_size();

	KFloat* cuda_gy = grad.data_ptr();
	KFloat* cuda_gs = grad_subvec.data_ptr();

	kcuda_call(kai_ker_acc_grad_subvector, ssize, (ssize, cuda_gy, cuda_gs, vec_size, nStart, nCount));
}

void KaiCudaMath::fft(KFloat* pWave, KFloat* pFFT, KInt mb_size, KInt fetch_width, KInt step_width, KInt step_cnt, KInt fft_width, KInt freq_cnt) {
	KInt piece_size = m_getMaxBatchSize(sizeof(float) * (fetch_width + step_cnt * (4 * fft_width + freq_cnt)));
	KInt split = 1;

	KaiShape wavShape{ piece_size, fetch_width };
	KaiShape bufShape{ piece_size, step_cnt, fft_width, 2 };
	KaiShape fftShape{ piece_size, step_cnt, freq_cnt };

	KInt nAlloc1, nAlloc2, nAlloc3, nAlloc4;

	KFloat* cuda_wave = m_pDevManager->allocFloatMemory(wavShape, &nAlloc1);
	KFloat* cuda_buf1 = m_pDevManager->allocFloatMemory(bufShape, &nAlloc2);
	KFloat* cuda_buf2 = m_pDevManager->allocFloatMemory(bufShape, &nAlloc3);
	KFloat* cuda_fft = m_pDevManager->allocFloatMemory(fftShape, &nAlloc4);

	KInt bsize = bufShape.total_size();
	KInt fsize = fftShape.total_size();

	KInt rest_size = mb_size;

	while (rest_size > 0) {
		KInt slice_size = (rest_size >= piece_size) ? piece_size : rest_size;

		cudaMemcpy(cuda_wave, pWave, sizeof(KFloat) * slice_size * fetch_width, cudaMemcpyHostToDevice);
		kcuda_call(kai_ker_wave_slices_to_complex, bsize, (bsize, cuda_buf1, cuda_wave, step_width, step_cnt, fft_width, fetch_width));
		m_fft_core_split(cuda_buf1, cuda_buf2, cuda_fft, fft_width, freq_cnt, fsize, bsize, split);
		cudaMemcpy(pFFT, cuda_fft, sizeof(float) * slice_size * step_cnt * freq_cnt, cudaMemcpyDeviceToHost);

		pWave += piece_size * fetch_width;
		pFFT += piece_size * step_cnt * freq_cnt;

		rest_size -= slice_size;
	}

	m_pDevManager->freeMemory(cuda_wave, nAlloc1);
	m_pDevManager->freeMemory(cuda_buf1, nAlloc2);
	m_pDevManager->freeMemory(cuda_buf2, nAlloc3);
	m_pDevManager->freeMemory(cuda_fft, nAlloc4);
}

void KaiCudaMath::m_fft_core_split(KFloat* cuda_buf1, KFloat* cuda_buf2, KFloat* cuda_y, KInt data_num, KInt freq_cnt, KInt dsize, KInt bsize, KInt split) {
	KFloat* src = cuda_buf1;
	KFloat* dst = cuda_buf2;

	KInt mb_size = bsize / (data_num * 2);
	KInt sp_size = mb_size / split;
	KInt ssize = bsize / split;

	KInt step = 1;

	while (step < data_num) {
		step = step * 2;
		for (KInt n = 0; n < split; n++) {
			KInt nd_base = n * sp_size;
			kcuda_call(kai_ker_fft_step_split, ssize, (ssize, dst, src, data_num, step, nd_base));
		}
		float* tmp = dst;
		dst = src;
		src = tmp;
	}

	kcuda_call(kai_ker_complex_to_abs_mean, dsize, (dsize, cuda_y, src, data_num, freq_cnt));
}

#include "../../src2020/core/array.h"
#include "../../src2020/core/host_math.h"

List KaiCudaMath::lstm_forward_test(KaiArray<KFloat> xarr, KaiArray<KFloat> yarr, KaiDict pm) {
	int m_timesteps = 200;
	int mb_size = 10;
	int m_recur_size = 64;
	int m_timefeats = 128;

	bool m_inseq = true;

	Array<float> recurrent = hmath.zeros(Shape(mb_size, m_recur_size));
	Array<float> state = hmath.zeros(Shape(mb_size, m_recur_size));
	Array<float> outputs = hmath.zeros(Shape(mb_size, m_timesteps, m_recur_size));

	Array<float> hidden = hmath.zeros(Shape(mb_size, m_timesteps, m_timefeats));

	Array<float> weight = hmath.zeros(Shape(m_timefeats + m_recur_size, 4 * m_recur_size));
	Array<float> bias = hmath.zeros(Shape(4 * m_recur_size));

	float* pSrc = xarr.data_ptr();
	float* pDst = hidden.data_ptr();

	KInt xsize = xarr.total_size();

	cudaMemcpy(pDst, pSrc, sizeof(float) * xsize, cudaMemcpyDeviceToHost);

	KaiDict pm_w = pm["w"], pm_b = pm["b"];
	KaiArray<KFloat> pm_weight = FARRAY(pm_w["_pm_"]);
	KaiArray<KFloat> pm_bias = FARRAY(pm_b["_pm_"]);

	pSrc = pm_weight.data_ptr();
	pDst = weight.data_ptr();

	xsize = pm_weight.total_size();

	cudaMemcpy(pDst, pSrc, sizeof(float) * xsize, cudaMemcpyDeviceToHost);

	pSrc = pm_bias.data_ptr();
	pDst = bias.data_ptr();

	xsize = pm_bias.total_size();

	cudaMemcpy(pDst, pSrc, sizeof(float) * xsize, cudaMemcpyDeviceToHost);

	Array<float> x_slice = hmath.zeros(Shape(mb_size, m_timefeats));
	Array<float> x_slice_buf, output_buf;

	if (!m_inseq) x_slice = hidden;

	List m_rnn_haux;

	for (int tn = 0; tn < m_timesteps; tn++) {
		if (m_inseq) {
			x_slice_buf = hidden[Axis(_all_, tn, _all_)];
			x_slice = x_slice_buf.reshape(Shape(mb_size, m_timefeats));
		}

		Dict aux_step;

		Array<float> ex_inp = hmath.hstack(x_slice, recurrent);

		//Array<float> affine = forward_affine(m_param, ex_inp);
		Array<float> affine = hmath.matmul(ex_inp, weight) + bias;

		aux_step["ex_inp"] = ex_inp;

		Array<float> forget_gate, input_gate, output_gate, block_input;

		/*
		forget_gate = affine[Axis(_all_, Ax(0 * m_recur_size, 1 * m_recur_size))];
		input_gate = affine[Axis(_all_, Ax(1 * m_recur_size, 2 * m_recur_size))];
		output_gate = affine[Axis(_all_, Ax(2 * m_recur_size, 3 * m_recur_size))];
		block_input = affine[Axis(_all_, Ax(3 * m_recur_size, 4 * m_recur_size))];
		*/

		affine = affine.reshape(Shape(10, 64, 4));

		forget_gate = affine[Axis(_all_, _all_, Ax(0))];
		input_gate = affine[Axis(_all_, _all_, Ax(1))];
		output_gate = affine[Axis(_all_, _all_, Ax(2))];
		block_input = affine[Axis(_all_, _all_, Ax(3))];

		forget_gate = forget_gate.reshape(Shape(10,64));
		input_gate = input_gate.reshape(Shape(10, 64));
		output_gate = output_gate.reshape(Shape(10, 64));
		block_input = block_input.reshape(Shape(10, 64));

		forget_gate = hmath.sigmoid(forget_gate);
		input_gate = hmath.sigmoid(input_gate);
		output_gate = hmath.sigmoid(output_gate);
		block_input = hmath.tanh(block_input);

		aux_step["forget_gate"] = forget_gate;
		aux_step["input_gate"] = input_gate;
		aux_step["output_gate"] = output_gate;
		aux_step["block_input"] = block_input;

		aux_step["state_tmp"] = state;
		state = state * forget_gate + block_input * input_gate;

		recurrent = hmath.tanh(state);
		aux_step["recur_tmp"] = recurrent;
		recurrent = recurrent * output_gate;

		output_buf = recurrent;

		m_rnn_haux.push_back(aux_step);

		outputs[Axis(_all_, tn, _all_)] = output_buf.reshape(Shape(mb_size, 1, m_recur_size));
	}

	Array<float> output = outputs;
	output = hmath.select_rnn_last_col(outputs);

	output.print("nocuda forward output");
	yarr.dump("yarr");

	return m_rnn_haux;
}

void KaiCudaMath::lstm_backprop_test(KaiArray<KFloat> gyarr, KaiArray<KFloat> gxarr, KaiDict pm, List aux) {
	int m_timesteps = 200;
	int mb_size = 10;
	int m_recur_size = 64;
	int m_timefeats = 128;

	bool m_inseq = true;

	Array<float> G_hidden = hmath.zeros(Shape(mb_size, m_recur_size));

	float* pSrc = gyarr.data_ptr();
	float* pDst = G_hidden.data_ptr();

	KInt xsize = gyarr.total_size();

	cudaMemcpy(pDst, pSrc, sizeof(float) * xsize, cudaMemcpyDeviceToHost);

	Array<float> weight = hmath.zeros(Shape(m_timefeats + m_recur_size, 4 * m_recur_size));
	Array<float> bias = hmath.zeros(Shape(4 * m_recur_size));

	KaiDict pm_w = pm["w"], pm_b = pm["b"];
	KaiArray<KFloat> pm_weight = FARRAY(pm_w["_pm_"]);
	KaiArray<KFloat> pm_bias = FARRAY(pm_b["_pm_"]);

	pSrc = pm_weight.data_ptr();
	pDst = weight.data_ptr();

	xsize = pm_weight.total_size();

	cudaMemcpy(pDst, pSrc, sizeof(float) * xsize, cudaMemcpyDeviceToHost);

	pSrc = pm_bias.data_ptr();
	pDst = bias.data_ptr();

	xsize = pm_bias.total_size();

	cudaMemcpy(pDst, pSrc, sizeof(float) * xsize, cudaMemcpyDeviceToHost);

	Array<float> G_weight = hmath.zeros(weight.shape());
	Array<float> G_bias = hmath.zeros(bias.shape());

	Shape xshape = Shape(mb_size, m_timesteps, m_timefeats);

	Array<float> G_xs = hmath.zeros(xshape);
	Array<float> G_recurrent = hmath.zeros(Shape(mb_size, m_recur_size));
	Array<float> G_state = hmath.zeros(Shape(mb_size, m_recur_size));

	Array<float> G_y_slice, G_affine, G_ex_inp;

	G_recurrent.copy(G_hidden);

	for (int tn = m_timesteps - 1; tn >= 0; tn--) {
		Dict aux_step = aux.back();
		aux.pop_back();

		Array<float> forget_gate, input_gate, output_gate, block_input, recur_tmp, state_tmp;

		forget_gate = aux_step["forget_gate"];
		input_gate = aux_step["input_gate"];
		output_gate = aux_step["output_gate"];
		block_input = aux_step["block_input"];
		recur_tmp = aux_step["recur_tmp"];
		state_tmp = aux_step["state_tmp"];

		Array<float> G_recur_tmp = G_recurrent * output_gate;
		Array<float> G_output_gate = G_recurrent * recur_tmp;

		G_state += hmath.tanh_derv(recur_tmp) * G_recur_tmp;

		Array<float> G_input_gate = G_state * block_input;
		Array<float> G_block_input = G_state * input_gate;

		Array<float> G_forget_gate = G_state * state_tmp;
		G_state = G_state * forget_gate;

		G_affine = hmath.zeros(Shape(mb_size, m_recur_size, 4));

		G_affine[Axis(_all_, _all_, 0)] = (hmath.sigmoid_derv(forget_gate) * G_forget_gate).reshape(Shape(10, 64, 1));
		G_affine[Axis(_all_, _all_, 1)] = (hmath.sigmoid_derv(input_gate) * G_input_gate).reshape(Shape(10, 64, 1));
		G_affine[Axis(_all_, _all_, 2)] = (hmath.sigmoid_derv(output_gate) * G_output_gate).reshape(Shape(10, 64, 1));
		G_affine[Axis(_all_, _all_, 3)] = (hmath.tanh_derv(block_input) * G_block_input).reshape(Shape(10, 64, 1));

		G_affine = G_affine.reshape(Shape(mb_size, 256));

		Array<float> ex_inp = aux_step["ex_inp"];

		Array<float> g_affine_weight = ex_inp.transpose();
		Array<float> g_affine_input = weight.transpose();

		G_weight += hmath.matmul(g_affine_weight, G_affine);
		G_bias += hmath.sum(G_affine, -1);
		G_ex_inp = hmath.matmul(G_affine, g_affine_input);

		Array<float> piece1, piece2;

		piece1 = G_ex_inp[Axis(_all_, Ax(0, m_timefeats))];
		piece2 = G_ex_inp[Axis(_all_, Ax(m_timefeats, m_timefeats+m_recur_size))];

		if (m_inseq)
			G_xs[Axis(_all_, tn, _all_)] = piece1.reshape(Shape(mb_size, 1, m_timefeats));
		else
			G_xs += piece1;

		G_recurrent = piece2.reshape(Shape(mb_size, m_recur_size));
	}

	KaiArray<KFloat> pm_weight_grad = FARRAY(pm_w["_grad_"]);
	KaiArray<KFloat> pm_bias_grad = FARRAY(pm_b["_grad_"]);

	pm_weight_grad.dump("pm_weight_grad");
	G_weight.print("nocuda backprop G_weight");

	pm_bias_grad.dump("pm_bias_grad");
	G_bias.print("nocuda backprop G_bias");

	gxarr.dump("gxarr");
	G_xs.print("nocuda backprop output");
}
