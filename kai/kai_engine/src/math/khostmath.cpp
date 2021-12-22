/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "khostmath.h"
#include "karray.h"
#include "../exec/exec_context.h"
#include "../../../kai_engine/src/nightly/nightly_utils.h"


#ifdef NORANDOM
#include <cmath>
static long seed = 1234;
static long norandom_coin = seed;
float norandom_normal(float mean, float std){
	norandom_coin = ((((norandom_coin * 214013L + 2531011L) >> 16) & 0x7fff)); // Dewdney Algorithm
	double temp = (double)(int)norandom_coin / 32768.0 + 0.0000001;
	norandom_coin = ((((norandom_coin * 214013L + 2531011L) >> 16) & 0x7fff));
	double temp2 = (double)(int)norandom_coin / 32768.0;
	float temp3 = (::sqrt(-2.0 * ::log(temp)) * cos(2.0 * 3.141592 * temp2))*std+mean; //Box-Muller Transform
	return temp3;
}
template<typename T> T norandom_uniform(T start, T end) {
	//norandom_coin = (norandom_coin << 7) % 65521;
	norandom_coin = ((((norandom_coin * 214013L + 2531011L) >> 16) & 0x7fff) % 32768);
	float temp = (float)(int)norandom_coin / 32768.0f;
	float temp2 = temp * ((float)end - (float)start) + (float)start;
	if (typeid(T) == typeid(int)) {
		temp2 += 0.4999f;
		temp2 = (float)floor(temp2);
	}
	return (T)temp2;
}

#endif

KaiHostMath hostmath; 

KaiHostMath::KaiHostMath() : KaiMath() {
	m_randGen.seed((unsigned int)time(NULL));
	//m_randGen.seed(1234);
}

KaiHostMath::~KaiHostMath() {
}

void KaiHostMath::cudaErrorCheck() {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::copy(KaiArray<KFloat> arr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::zeros(KaiShape shape) {
	return KaiArray<KFloat>::zeros(shape);
}

KaiArray<KFloat> KaiHostMath::ones(KaiShape shape, KFloat fill) {
	return KaiArray<KFloat>::ones(shape, fill);
}

KaiArray<KFloat> KaiHostMath::random_uniform(KaiShape shape) {
	KaiArray<KFloat> arr(shape);

	std::uniform_real_distribution<float> coin(0.0f, 1.0f);
	KFloat* pData = arr.data_ptr();
	KInt size = arr.total_size();

	for (KInt n = 0; n < size; n++) {
#ifdef NORANDOM
		*pData++ = norandom_uniform<float>(0.0f, 1.0f);
#else
		*pData++ = coin(m_randGen);
#endif
	}
	return arr;
}

KaiArray<KFloat> KaiHostMath::random_normal(KaiShape shape, KFloat mean, KFloat std, KBool adapt) {
	
	////////////////////////////////////////////////////////////////

	// Before bugfix on 2021-09-02 :
	// When the array size is too small,
	// the array may not be initialized to a normal distribution.
	
	KaiArray<KFloat> arr(shape);

	std::normal_distribution<float> coin(mean, std);


	KFloat* pData = arr.data_ptr();
	KInt size = arr.total_size();
	
	for (KInt n = 0; n < size; n++) {
#ifdef NORANDOM
		
		* pData++ = norandom_normal(mean, std);

#else
		* pData++ = coin(m_randGen);
#endif	


	}
	pData -= size;
	KFloat sum = 0.0f;
	if (adapt) {
		for (KInt n = 0; n < size; n++) {
			sum += *pData++;
		}
		mean = sum / (KFloat)size;
		pData -= size;
		for (KInt n = 0; n < size; n++) {
			*pData++ -= mean;
			//printf("[%d] Random number : %.9lf\n", (int)n, random_number);
		}
	}
	return arr;

	////////////////////////////////////////////////////////////////

//#if (ACTIVATE_TEST && TEST_NORMAL_DISTRIBUTION_WITH_FIXED_VALUES)
//	// Test code for debugging : Generate normal distribution with fixed values
//	KaiArray<KFloat> test_arr(shape);
//	KFloat* pArray = test_arr.data_ptr();
//	KInt array_size = test_arr.total_size();
//
//	if (array_size == 1) {
//		pArray[0] = 0.0f;
//		return test_arr;
//	}
//
//	for (KInt i=0; i<array_size; ++i) {
//		pArray[i] = -std + i*((2.0f * std) / (array_size-1));
//	}
//
//	return test_arr;
//
//#else
//	// Bug fixed by Hyung-jae, Son on 2021-09-02 :
//	// The mean of generated random numbers was modified
//	// to always be 0.
//
//	KaiArray<KFloat> arr(shape);
//	//printf("[DEBUG] %s(%u): shape.desc() : %s\n", __FUNCTION__, __LINE__, shape.desc().c_str());
//
//	std::normal_distribution<float> coin(mean, std);
//
//	KFloat* pData = arr.data_ptr();
//	KInt size = arr.total_size();
//
//	KFloat sum = 0.0f;
//	for (KInt n = 0; n < size; n++) {
//		sum += (*pData++ = coin(m_randGen));
//	}
//
//	// Adaptive distribution : Adjust so that the mean is zero
//	if (adapt) {
//		pData = arr.data_ptr();
//		mean = sum / (KFloat)size;
//
//		for (KInt n = 0; n < size; n++) {
//			KFloat random_number = ((*pData++) -= mean);
//			//printf("[%d] Random number : %.9lf\n", (int)n, random_number);
//		}
//	}
//
//	return arr;
//
//#endif

}

KaiArray<KInt> KaiHostMath::to_cuda(KaiArray<KInt> arr) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "to_cuda() should be called only by the cudamath");
}

KaiArray<KFloat> KaiHostMath::to_cuda(KaiArray<KFloat> arr) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "to_cuda() should be called only by the cudamath");
}

KaiArray<KInt> KaiHostMath::to_host(KaiArray<KInt> arr) {
	return arr;
}

KaiArray<KFloat> KaiHostMath::to_host(KaiArray<KFloat> arr) {
	return arr;
}

// The following 4 methods have been added by Hyung-jae, Son (2021-08-20)

void KaiHostMath::to_cuda(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "to_cuda() should be called only by the cudamath");
}

void KaiHostMath::to_cuda(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "to_cuda() should be called only by the cudamath");
}

void KaiHostMath::to_host(KaiArray<KInt>& arrSrc, KaiArray<KInt>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	if (arrSrc.is_cuda() || arrDst.is_cuda())
		throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "device array should be called only by the cudamath");

	if (arrSrc.total_size() < nSrcStart + nCount || arrDst.total_size() < nDstStart + nCount)
		throw KaiException(KERR_INDEX_OUT_OF_RANGE, "invalid start index or copy size");

	memcpy(arrDst.data_ptr()+nDstStart, arrSrc.data_ptr()+nSrcStart, sizeof(KInt)*nCount);
}

void KaiHostMath::to_host(KaiArray<KFloat>& arrSrc, KaiArray<KFloat>& arrDst, KInt nSrcStart, KInt nDstStart, KInt nCount) {
	if (arrSrc.is_cuda() || arrDst.is_cuda())
		throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "device array should be called only by the cudamath");

	if (arrSrc.total_size() < nSrcStart + nCount || arrDst.total_size() < nDstStart + nCount)
		throw KaiException(KERR_INDEX_OUT_OF_RANGE, "invalid start index or copy size");

	memcpy(arrDst.data_ptr()+nDstStart, arrSrc.data_ptr()+nSrcStart, sizeof(KFloat)*nCount);
}

KaiArray<KInt> KaiHostMath::arange(KInt nCount) {
	KaiArray<KInt> arr(KaiShape{ nCount });
	KInt* pData = arr.data_ptr();
	for (KInt n = 0; n < nCount; n++) pData[n] = n;
	return arr;
}

KaiArray<KInt> KaiHostMath::subrange(KaiArray<KInt> arr, KInt nStart, KInt nCount) {
	KInt nSize = arr.total_size();

	if (nStart < 0 || nStart + nCount > nSize) throw KaiException(KERR_BAD_INDEX_ON_ARRAY_SUBRANGE);

	KaiArray<KInt> dst(KaiShape{ nCount });

	KInt* pSrc = arr.data_ptr();
	KInt* pDst = dst.data_ptr();

	memcpy(pDst, pSrc + nStart, nCount * sizeof(KInt));

	return dst;

}
 
void KaiHostMath::shuffle(KaiArray<KInt> arr) {
	KInt* pData = arr.data_ptr();
	KInt nSize = arr.total_size();

	for (KInt n = 0; n < nSize - 1; n++) {
		KInt nth=0;
#ifdef NORANDOM
		nth = norandom_uniform<int>(0, nSize - n - 1);
#else
		std::uniform_int_distribution<KInt> coin(0, nSize - n - 1);
		nth = coin(m_randGen);
#endif
		std::swap<KInt>(pData[n], pData[n + nth]);
	}
}


void KaiHostMath::shuffle(KaiArray<KInt> arr, KInt nStart, KInt nCount) {
//#ifndef NORANDOM
	KInt* pData = arr.data_ptr();
	for (KInt n = 0; n < nCount - 1; n++) {
		KInt nth=0;
#ifdef NORANDOM
		nth = norandom_uniform<int>(0, nCount - n - 1);
#else
		std::uniform_int_distribution<KInt> coin(0, nCount - n - 1);
		nth = coin(m_randGen);
#endif
		std::swap<KInt>(pData[nStart + n], pData[nStart + n + nth]);
	}
//#endif
}

KaiArray<KFloat> KaiHostMath::matmul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	// 2020 시제품 버전은 복수의 matmul을 한 방에 처리할 수 있게 확장된 내용임, attention layer 지원 시점에 비슷한 확장 여부를 검토해야 함
	KaiShape shape1 = arr1.shape();
	KaiShape shape2 = arr2.shape();

	if (shape1.size() <= 1 || shape2.size() != 2) throw KaiException(KERR_MATMUL_SHAPE_MISMATCH);

	KInt nvecs = shape2[0];
	KInt ncols = shape2[1];
	KInt nrows = arr1.total_size() / nvecs;

	if (nvecs != shape1[shape1.size() - 2]) throw KaiException(KERR_MATMUL_SHAPE_MISMATCH);

	KaiShape yshape = shape1.replace_end(ncols);

	KaiArray<KFloat> mul = zeros(yshape);

	KFloat* p_base1 = arr1.data_ptr();
	KFloat* p_base2 = arr2.data_ptr();
	KFloat* mp = mul.data_ptr();

	for (KInt n = 0; n < nrows; n++) {
		for (KInt m = 0; m < ncols; m++) {
			KFloat dot_sum = 0;
			KFloat* p1 = p_base1 + n * nvecs;
			KFloat* p2 = p_base2 + m;
			for (KInt k = 0; k < nvecs; k++) {
				dot_sum += *p1 * *p2;
				p1++;
				p2 += ncols;
			}
			*mp++ = dot_sum;
		}
	}

	return mul;
}

KaiArray<KFloat> KaiHostMath::add_bias(KaiArray<KFloat> arr, KaiArray<KFloat> bias) {
	KaiShape ashape = arr.shape();
	KaiShape bshape = bias.shape();

	if (ashape.size() <= 0 || bshape.size() != 1) throw KaiException(KERR_ADD_BIAS_SHAPE_MISMATCH);

	KInt nvecs = bshape[0];
	KInt nsize = arr.total_size();

	if (nvecs != ashape[ashape.size() - 2]) throw KaiException(KERR_ADD_BIAS_SHAPE_MISMATCH);

	KaiArray<KFloat> sum = zeros(ashape);

	KFloat* ap = arr.data_ptr();
	KFloat* bp = bias.data_ptr();
	KFloat* sp = sum.data_ptr();

	for (KInt n = 0; n < nsize ; n++) {
		*sp++ = *ap++ + bp[n % nvecs];
	}

	return sum;
}

KaiArray<KFloat> KaiHostMath::transpose(KaiArray<KFloat> arr) {
	if (arr.dim() != 2) throw KaiException(KERR_BAD_DIMENSION_FOR_ARRAY_TRANSPOSE);

	KaiShape ashape = arr.shape();
	
	KaiArray<KFloat> trans(KaiShape{ ashape[1], ashape[0] });

	float* bp_base = trans.data_ptr();
	float* bp_end = bp_base + arr.total_size();

	float* bp = bp_base++;
	float* ap = arr.data_ptr();

	KInt asize = arr.total_size();
	KInt rows = ashape[0];

	for (KInt n = 0; n < asize; n++, ap++) {
		*bp = *ap;
		bp += rows;
		if (bp >= bp_end) bp = bp_base++;
	}

	return trans;
}

KaiArray<KFloat> KaiHostMath::sum_on_column(KaiArray<KFloat> arr) {
	if (arr.dim() != 2) throw KaiException(KERR_BAD_DIMENSION_FOR_ARRAY_SUM_ON_COL);

	KaiShape ashape = arr.shape();

	KaiArray<KFloat> sumarr = zeros(KaiShape{ ashape[1] });

	KFloat* ap = arr.data_ptr();
	KFloat* bp = sumarr.data_ptr();

	KInt asize = arr.total_size();
	KInt cols = ashape[1];

	for (KInt n = 0; n < asize; n++) {
		bp[n % cols] += ap[n];
	}

	return sumarr;
}

KaiArray<KFloat> KaiHostMath::sum_on_row(KaiArray<KFloat> arr) {
	if (arr.dim() != 2) throw KaiException(KERR_BAD_DIMENSION_FOR_ARRAY_SUM_ON_COL);

	KaiShape ashape = arr.shape();

	KaiArray<KFloat> sumarr = zeros(KaiShape{ ashape[0] });

	KFloat* ap = arr.data_ptr();
	KFloat* bp = sumarr.data_ptr();

	KInt asize = arr.total_size();
	KInt cols = ashape[1];

	for (KInt n = 0; n < asize; n++) {
		bp[n / cols] += ap[n];
	}

	return sumarr;
}

KaiArray<KFloat> KaiHostMath::acivate(KaiArray<KFloat> arr, KInt nActFuncID, KaiExecContext* pContext) {
	switch ((Ken_actfunc)nActFuncID) {
	case Ken_actfunc::none:
		return arr;
	case Ken_actfunc::relu:
		return relu(arr);
	case Ken_actfunc::sigmoid:
		return sigmoid(arr);
	case Ken_actfunc::tanh:
		return tanh(arr);
	case Ken_actfunc::leaky_relu:
		return leaky_relu(arr, pContext->get_property("leaky_alpha", 0.1f));
	case Ken_actfunc::gelu:
		return gelu(arr);
	case Ken_actfunc::custom:
		throw KaiException(KERR_UNIMPEMENTED_YET, "custom activate function");
	default:
		throw KaiException(KERR_UNKNOWN_ACTFUNCNAME);
	}
}

KaiArray<KFloat> KaiHostMath::acivate_backprop(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat> y, KInt nActFuncID, KaiExecContext* pContext) {
	switch ((Ken_actfunc)nActFuncID) {
	case Ken_actfunc::none:
		return garr;
	case Ken_actfunc::relu:
		return mul(relu_derv(x), garr);
	case Ken_actfunc::sigmoid:
		return mul(sigmoid_derv(y), garr);
	case Ken_actfunc::tanh:
		return mul(tanh_derv(y), garr);
	case Ken_actfunc::leaky_relu:
		return mul(leaky_relu_derv(x, pContext->get_property("leaky_alpha", 0.1f)), garr);
	case Ken_actfunc::gelu:
		return mul(gelu_derv(x), garr);
	case Ken_actfunc::custom:
		throw KaiException(KERR_UNIMPEMENTED_YET, "custom activate function");
	default:
		throw KaiException(KERR_UNKNOWN_ACTFUNCNAME);
	}
}

KaiArray<KFloat> KaiHostMath::relu(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();
	
	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = (ap[n] > 0) ? ap[n] : 0;
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::sigmoid(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		KFloat x = ap[n];
		dp[n] = (KFloat)((x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (::expf(x) + 1.0)));
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::sigmoid_derv_grad(KaiArray<KFloat> gsig, KaiArray<KFloat> sig) {
	KaiArray<KFloat> dst(sig.shape());

	KFloat* dp = dst.data_ptr();
	KFloat* ap = gsig.data_ptr();
	KFloat* bp = sig.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = ap[n] * bp[n] * (1 - bp[n]);
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::tanh(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		KFloat x = ap[n] * 2;
		dp[n] = (KFloat) ((x > 0) ? ((1.0 - ::expf(-x)) / (1.0 + ::expf(-x))) : ((::expf(x) - 1.0) / (::expf(x) + 1.0)));
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::leaky_relu(KaiArray<KFloat> arr, KFloat leaky_alpha) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = (ap[n] > 0) ? ap[n] : ap[n] * leaky_alpha;
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::gelu(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		KFloat x = ap[n];
		KFloat t = x * 0.797885f + x * x * x * 0.035677f;
		KFloat u = t * 2;
		KFloat v = (KFloat)((u > 0) ? ((1.0 - ::expf(-u)) / (1.0 + ::expf(-u))) : ((::expf(u) - 1.0) / (::expf(u) + 1.0)));

		dp[n] = x * 0.5f * v;
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::relu_derv(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = (ap[n] > 0) ? 1.0f : 0.0f;
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::sigmoid_derv(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		KFloat y = ap[n];
		dp[n] = y * (1.0f - y);
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::tanh_derv(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		KFloat y = ap[n];
		dp[n] = 1.0f - y * y;
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::leaky_relu_derv(KaiArray<KFloat> arr, KFloat leaky_alpha) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = (ap[n] > 0) ? 1 : leaky_alpha;
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::gelu_derv(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		KFloat x = ap[n];
		KFloat t = x * 0.797885f + x * x * x * 0.035677f;
		KFloat u = (KFloat)((t > 0) ? ((1.0 - ::expf(-t)) / (1.0 + ::expf(-t))) : ((::expf(t) - 1.0) / (::expf(t) + 1.0)));

		KFloat v = x * x * 2 * 0.035677f + 0.797885f;
		KFloat w = -0.5f * (u + 1) * ((u - 1) * x * v + 1.0f);

		dp[n] = w;
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::minus(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = -ap[n];
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::eval_binary_op(exp_op op_code, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) {
		if (arr2.total_size() != 1) throw KaiException(KERR_UNIMPEMENTED_YET);
		return eval_binary_op(op_code, arr1, fetch(arr2));
	}

	if (op_code == exp_op::softmax_cross_entropy_with_logits) {
		return softmax_cross_entropy_with_logits(arr1, arr2);
	}

	KaiArray<KFloat> dst(arr1.shape());

	KFloat* pd = dst.data_ptr();
	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		pd[n] = m_eval_binary_op(op_code, p1[n],  p2[n]);
	}

	return dst;
}

KFloat KaiHostMath::m_eval_binary_op(exp_op op_code, KFloat elem1, KFloat elem2) {
	switch (op_code) {
	case exp_op::_and:
		return (elem1 != 0.0f && elem2 != 0.0f) ? 1.0f : 0.0f;
	case exp_op::_or:
		return (elem1 != 0.0f || elem2 != 0.0f) ? 1.0f : 0.0f;
	case exp_op::add:
		return elem1 + elem2;
	case exp_op::sub:
		return elem1 - elem2;
	case exp_op::mult:
		return elem1 * elem2;
	case exp_op::div:
		return (elem2 != 0.0f) ? elem1 / elem2 : elem2;
	case exp_op::gt:
		return (elem1 > elem2) ? 1.0f : 0.0f;
	case exp_op::lt:
		return (elem1 < elem2) ? 1.0f : 0.0f;
	case exp_op::ge:
		return (elem1 >= elem2) ? 1.0f : 0.0f;
	case exp_op::le:
		return (elem1 <= elem2) ? 1.0f : 0.0f;
	case exp_op::equal:
		return (elem1 == elem2) ? 1.0f : 0.0f;
	case exp_op::sigmoid_cross_entropy_with_logits:
		return MAX(elem1,0) - elem1 * elem2 + ::logf(1.0f + ::expf(-::fabs(elem1)));
	default:
		throw KaiException(KERR_NEED_CODE_MODIFICATION, "missing exp_op case");
	}
}

KaiArray<KFloat> KaiHostMath::add(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::add, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::sub(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::sub, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::mul(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::mult, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::div(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::div, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::gt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::gt, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::lt(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::lt, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::equal(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::equal, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::filter(KaiArray<KFloat> xarr, KaiArray<KInt> mask) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::sigmoid_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	return eval_binary_op(exp_op::sigmoid_cross_entropy_with_logits, arr1, arr2);
}

KaiArray<KFloat> KaiHostMath::softmax_cross_entropy_with_logits(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);
	if (arr1.dim() != 2) throw KaiException(KERR_UNIMPEMENTED_YET);

	KaiArray<KFloat> probs = softmax(arr1);
	KaiArray<KFloat> logs = log(add(probs, 1.0e-10f));
	KaiArray<KFloat> log_ans = mul(arr2, logs);
	KaiArray<KFloat> sums = sum_on_row(log_ans);

	return minus(sums);
}

KaiArray<KFloat> KaiHostMath::softmax_cross_entropy_with_logits_idx(KaiArray<KFloat> arr1, KaiArray<KInt> arr2) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::softmax_cross_entropy_with_logits_idx_derv(KaiArray<KFloat> arr1, KaiArray<KInt> arr2) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::equal_col(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::max_col(KaiArray<KFloat> arr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::max_col_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::vstack(KaiArray<KFloat> arr_on, KaiArray<KFloat> arr, KInt nFrom) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::vstack_grad(KaiArray<KFloat> grad, KInt nStart, KInt nCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::iou_yolo(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::iou_yolo_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> arr1, KaiArray<KFloat> arr2, KInt nth) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::argmax(KaiArray<KFloat> arr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::max(KaiArray<KFloat> arr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::add_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt size = arr1.total_size();

	for (KInt n = 0; n < size; n++) {
		p1[n] = p1[n] + p2[n];
	}
}

void KaiHostMath::sub_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt size = arr1.total_size();

	for (KInt n = 0; n < size; n++) {
		p1[n] = p1[n] - p2[n];
	}
}

void KaiHostMath::mul_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt size = arr1.total_size();

	for (KInt n = 0; n < size; n++) {
		p1[n] = p1[n] * p2[n];
	}
}

void KaiHostMath::div_on(KaiArray<KFloat> arr1, KaiArray<KFloat> arr2) {
	if (arr1.shape() != arr2.shape()) throw KaiException(KERR_UNIMPEMENTED_YET);

	KFloat* p1 = arr1.data_ptr();
	KFloat* p2 = arr2.data_ptr();

	KInt size = arr1.total_size();

	for (KInt n = 0; n < size; n++) {
		p1[n] = p1[n] / p2[n];
	}
}

KaiArray<KFloat> KaiHostMath::add(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::add, arr, term);
}

KaiArray<KFloat> KaiHostMath::sub(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::sub, arr, term);
}

KaiArray<KFloat> KaiHostMath::mul(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::mult, arr, term);
}

KaiArray<KFloat> KaiHostMath::div(KaiArray<KFloat> arr, KFloat term) {
	return eval_binary_op(exp_op::div, arr, term);
}

KaiArray<KFloat> KaiHostMath::eval_binary_op(exp_op op_code, KaiArray<KFloat> arr, KFloat term) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* pd = dst.data_ptr();
	KFloat* p1 = arr.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		pd[n] = pd[n] = m_eval_binary_op(op_code, p1[n], term);
	}

	return dst;
}

void KaiHostMath::mul_on(KaiArray<KFloat> arr, KFloat term) {
	KFloat* pa = arr.data_ptr();

	KInt size = arr.total_size();

	for (KInt n = 0; n < size; n++) {
		pa[n] = pa[n] * term;
	}
}

KaiArray<KFloat> KaiHostMath::sign(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = (ap[n] > 0) ? 1.0f : ((ap[n] < 0) ? -1.0f : 0);
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::square(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = ap[n] * ap[n];
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::sqrt(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = ::sqrt(ap[n]);
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::log(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = ::log(ap[n]);
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::exp(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(arr.shape());

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	for (KInt n = 0; n < size; n++) {
		dp[n] = ::expf(ap[n]);
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::sum(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(KaiShape{ 1 });

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	*dp = 0;
	for (KInt n = 0; n < size; n++) {
		*dp += ap[n];
	}

	return dst;
}

KaiArray<KFloat> KaiHostMath::sum_grad(KaiArray<KFloat> grad_y, KaiArray<KFloat> farr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::mean(KaiArray<KFloat> arr) {
	KaiArray<KFloat> dst(KaiShape{ 1 });

	KFloat* ap = arr.data_ptr();
	KFloat* dp = dst.data_ptr();

	KInt size = dst.total_size();

	*dp = 0;
	for (KInt n = 0; n < size; n++) {
		*dp += ap[n];
	}
	*dp /= size;

	return dst;
}

KInt KaiHostMath::fetch(KInt* arr, KInt nIndex) {
	return arr[nIndex];
}

KFloat KaiHostMath::fetch(KFloat* arr, KInt nIndex) {
	return arr[nIndex];
}

KFloat KaiHostMath::fetch(KaiArray<KFloat> arr, KInt nIndex) {
	KInt size = arr.total_size();

	if (nIndex < 0 || nIndex >= size) throw KaiException(KERR_BAD_INDEX_ON_FARRAY_FETCH);

	KFloat* ap = arr.data_ptr();

	return ap[nIndex];
}

KInt KaiHostMath::fetch(KaiArray<KInt> arr, KInt nIndex) {
	KInt size = arr.total_size();

	if (nIndex < 0 || nIndex >= size) throw KaiException(KERR_BAD_INDEX_ON_FARRAY_FETCH);

	KInt* ap = arr.data_ptr();

	return ap[nIndex];
}

KaiArray<KFloat> KaiHostMath::softmax(KaiArray<KFloat> arr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::softmax_derv(KaiArray<KFloat> gyarr, KaiArray<KFloat> yarr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::eval_adam_delta(KaiArray<KFloat> grad, KaiArray<KFloat> pm_s, KaiArray<KFloat> pm_t, KFloat pm_n, KFloat ro1, KFloat ro2, KFloat epsilon) {
	mul_on(pm_s, ro1);
	mul_on(pm_t, ro2);

	add_on(pm_s, mul(grad, 1 - ro1));
	add_on(pm_t, mul(mul(grad, grad), 1 - ro2));

	KaiArray<KFloat> s = div(pm_s, 1.0f - ::powf(ro1, pm_n));
	KaiArray<KFloat> t = div(pm_t, 1.0f - ::powf(ro2, pm_n));

	return add(grad, div(s, add(sqrt(t), epsilon)));
}

KaiArray<KFloat> KaiHostMath::apply_decay(KaiArray<KFloat> pm, KaiArray<KFloat> grad, KFloat l2_decay, KFloat l1_decay) {
	if (l2_decay > 0) {
		grad = add(grad, mul(pm, l2_decay));
	}

	if (l1_decay > 0) {
		grad = add(grad, mul(sign(pm), l1_decay));
	}

	return grad;
}

KaiList KaiHostMath::to_host(KaiList list) {
	return list;
}

KaiDict KaiHostMath::to_host(KaiDict dict) {
	return dict;
}

KFloat KaiHostMath::mean(KaiList list) {
	KFloat count = (KFloat)list.size();
	return sum(list) / count;
}

KFloat KaiHostMath::sum(KaiList list) {
	KFloat count = (KFloat)list.size();
	KFloat sum = 0;

	for (auto& it : list) {
		sum += (KFloat)it;
	}

	return sum;
}

KaiArray<KFloat> KaiHostMath::convolution(KaiArray<KFloat> xarr, KaiArray<KFloat> kernel) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::convolution_derv_x(KaiArray<KFloat> gyarr, KaiArray<KFloat> kernel) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::convolution_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> xarr, KaiShape kshape) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::subrange(KaiArray<KFloat> xarr, KInt nth_ax, KInt nFrom, KInt nCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::subrange_derv(KaiArray<KFloat> gyarr, KInt nth_ax, KInt nFrom, KInt nCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::stride(KaiArray<KFloat> xarr, KaiShape stride) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::stride_derv(KaiArray<KFloat> gyarr, KaiShape stride, KaiShape xshape) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::max_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pMaxMap, KaiShape kernel) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::max_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> maxMap, KaiShape kernel) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::avg_pool(KaiArray<KFloat> xarr, KaiArray<KInt>* pAvgCnt, KaiShape kernel) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::avg_pool_derv(KaiArray<KFloat> gyarr, KaiArray<KInt> avgCnt, KaiShape kernel) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::globalavg(KaiArray<KFloat> xarr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::globalavg_derv(KaiArray<KFloat> gyarr, KaiShape xshape) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::BNCollectNorm(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KaiArray<KFloat>& var, KFloat momentum, KFloat epsilon) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::BnNormalize(KaiArray<KFloat> xarr, KaiArray<KFloat> mavg, KaiArray<KFloat> mvar, KFloat epsilon) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::BnScale(KaiArray<KFloat> xarr, KaiArray<KFloat> scale, KaiArray<KFloat> shift) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::BnNormDerv(KaiArray<KFloat> garr, KaiArray<KFloat> var, KFloat epsilon) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::rescale_derv_pm(KaiArray<KFloat> garr, KaiArray<KFloat> x, KaiArray<KFloat>* p_grad_scale, KaiArray<KFloat>* p_grad_shift) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::rescale_derv_x(KaiArray<KFloat> garr, KaiArray<KFloat> scale) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::CopyIntoSlice(KaiArray<KFloat> yarr, KaiArray<KFloat> barr, KInt& nChnPos) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::CopyFromSlice(KaiArray<KFloat> garr, KInt& nChnPos, KInt nChnCnt) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::random_bernoulli(KaiShape shape, KFloat one_ratio) {
	KaiArray<KFloat> arr(shape);

	std::uniform_real_distribution<float> coin(0.0f, 1.0f);

	KFloat* pData = arr.data_ptr();
	KInt size = arr.total_size();

	for (KInt n = 0; n < size; n++) {
#ifdef NORANDOM
		* pData++ = (norandom_uniform(0.0f, 1.0f) < one_ratio) ? 1.0f : 0.0f;
#else
		*pData++ = (coin(m_randGen) < one_ratio) ? 1.0f : 0.0f;
#endif	
		
	}

	return arr;
}

KaiArray<KFloat> KaiHostMath::dropout(KaiArray<KFloat> xarr, KaiArray<KFloat> mask, KFloat keep_raio) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::dropout_derv(KaiArray<KFloat> xarr, KaiArray<KFloat> mask, KFloat keep_raio) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::residual_add(KaiArray<KFloat> yarr, KaiArray<KFloat> xarr) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::residual_add_derv(KaiArray<KFloat> gyarr, KInt bchn) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::CombineExtendedInput(KaiArray<KFloat> recurrent, KBool isSeq, KaiArray<KFloat> xarr, KInt nth) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::SplitExtendedInputGrad(KaiArray<KFloat> g_exp_input, KBool isSeq, KaiArray<KFloat> g_x, KInt nth) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::CopyIntoTimeSlice(KaiArray<KFloat> yarr, KaiArray<KFloat> recurrent, KInt nth) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::add_time_slice_on_dest(KaiArray<KFloat> dest, KaiArray<KFloat> whole, KInt nth) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::lstm_gates(KaiArray<KFloat> affine) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::lstm_proc(KaiArray<KFloat> gates, KaiArray<KFloat>& state, KBool use_state) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::lstm_gates_derv(KaiArray<KFloat> g_gates, KaiArray<KFloat> gates) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::lstm_proc_derv(KaiArray<KFloat>& g_state, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_state, KaiArray<KFloat> post_recur, KBool use_state) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::gru_combine_extra(KaiArray<KFloat> exp_input, KaiArray<KFloat> gates) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::gru_combine_extra_derv(KaiArray<KFloat> g_exp_input, KaiArray<KFloat> g_gates, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> exp_input) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::gru_proc(KaiArray<KFloat> gates, KaiArray<KFloat> recurrent, KaiArray<KFloat> extra_affine) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::gru_proc_derv(KaiArray<KFloat>& g_gates, KaiArray<KFloat>& g_new_rec, KaiArray<KFloat> g_recurrent, KaiArray<KFloat> gates, KaiArray<KFloat> pre_recur, KaiArray<KFloat> extra_affine) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::add_embed_dict(KaiArray<KFloat> yarr, KaiArray<KInt> tokens, KaiArray<KFloat> word_vecs, KInt axis) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiList KaiHostMath::split_array(KaiArray<KFloat> arr, KInt piece_cnt) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::merge_array(KaiList arrs) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::multi_head_matmul_qk(KaiArray<KFloat> query, KaiArray<KFloat> key, KInt head_cnt) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::multi_head_matmul_qk_derv_q(KaiArray<KFloat> gyarr, KaiArray<KFloat> key, KInt head_cnt) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::multi_head_matmul_qk_derv_k(KaiArray<KFloat> gyarr, KaiArray<KFloat> query, KInt head_cnt) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::multi_head_matmul_pv(KaiArray<KFloat> probs, KaiArray<KFloat> value) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::multi_head_matmul_pv_derv_p(KaiArray<KFloat> gyarr, KaiArray<KFloat> value, KInt head_cnt) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::multi_head_matmul_pv_derv_v(KaiArray<KFloat> gyarr, KaiArray<KFloat> probs) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}


KaiArray<KFloat> KaiHostMath::extract(KaiArray<KFloat> xarr, KInt axis, KInt index, KInt count, KBool reduce_seq) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::extract_derv(KaiArray<KFloat> gyarr, KaiShape xshape, KInt axis, KInt index, KInt count, KBool reduce_seq) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::select(KaiArray<KFloat> xarr, KaiArray<KInt> selector, KaiShape vector_shape) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::select_derv(KaiArray<KFloat> garr, KaiArray<KInt> selector_arr, KaiShape xshape, KaiShape vshape) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::update_dic_weight_sgd(KaiArray<KFloat> weight, KaiArray<KFloat> grad, KaiArray<KInt> tokens, KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::update_dic_weight_adam(KaiArray<KFloat> weight, KaiArray<KFloat> s, KaiArray<KFloat> t, KaiArray<KFloat> n, KaiArray<KFloat> grad, KaiArray<KInt> tokens, KInt nth, KFloat learning_rate, KFloat l2_decay, KFloat l1_decay, KFloat ro1, KFloat ro2, KFloat epsilon) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::expand(KaiArray<KFloat> xarr, KaiShape ratio) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::expand_derv(KaiArray<KFloat> gyarr, KaiShape ratio) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KInt KaiHostMath::stack_on(KaiArray<KFloat> dest, KaiArray<KFloat> src, KInt tail_size, KInt nFrom, KInt nTo) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::stack_on_grad(KaiArray<KFloat> gyarr, KaiShape shape, KInt tail_size, KInt& nFrom, KInt nTo) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

KaiArray<KFloat> KaiHostMath::get_subvector(KaiArray<KFloat> arr, KInt nStart, KInt nCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::get_subvector_derv_acc(KaiArray<KFloat> grad, KaiArray<KFloat> grad_subvec, KInt nStart, KInt nCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiHostMath::fft(KFloat* pWave, KFloat* pFTT, KInt mb_size, KInt fetch_width, KInt step_width, KInt step_cnt, KInt fft_width, KInt freq_cnt) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}
