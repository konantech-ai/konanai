/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "karray.h"
#include "karr_math.h"
#include "../utils/kutil.h"

kmutex_wrap ms_mu_karr_data;
kmutex_wrap ms_mu_karr_core;

template<> int KaiArrayCore<KInt>::ms_checkCode = 18224093;
template<> int KaiArrayCore<KFloat>::ms_checkCode = 97301981;

#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif

template<> KaiArrayCore<KInt>::KaiArrayCore() : KaiObject(Ken_object_type::narray), m_shape(), m_mdata(NULL) {
	m_checkCode = ms_checkCode;
}

template<> KaiArrayCore<KFloat>::KaiArrayCore() : KaiObject(Ken_object_type::farray), m_shape(), m_mdata(NULL) {
	m_checkCode = ms_checkCode;
}

template<class T> KaiArrayCore<T>* KaiArrayCore<T>::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Array");

	KaiArrayCore<T>* pArr = (KaiArrayCore<T>*)hObject;

	if (pArr->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Array");

	return pArr;
}

template KaiArrayCore<KInt>* KaiArrayCore<KInt>::HandleToPointer(KHObject hObject, KaiSession* pSession);
template KaiArrayCore<KFloat>* KaiArrayCore<KFloat>::HandleToPointer(KHObject hObject, KaiSession* pSession);

template<class T> KaiArrayCore<T>* KaiArrayCore<T>::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Array");

	KaiArrayCore<T>* pArr = (KaiArrayCore<T>*)hObject;

	if (pArr->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Array");

	return pArr;
}

template KaiArrayCore<KInt>* KaiArrayCore<KInt>::HandleToPointer(KHObject hObject);
template KaiArrayCore<KFloat>* KaiArrayCore<KFloat>::HandleToPointer(KHObject hObject);

template<> Ken_object_type KaiArrayCore<KInt>::get_type() { return Ken_object_type::narray; }
template<> Ken_object_type KaiArrayCore<KFloat>::get_type() { return Ken_object_type::farray; }

template<class T> KaiArray<T> KaiArray<T>::zeros(KaiShape shape) {
	KInt size = shape.total_size();

	KaiArray<T> arr;
	arr.m_core->m_shape = shape;
	arr.m_core->m_mdata = new KaiArrayData<T>(size);
	memset(arr.m_core->data(), 0, arr.mem_size());

	return arr;
}

template KaiArray<KInt> KaiArray<KInt>::zeros(KaiShape shape);
template KaiArray<KFloat> KaiArray<KFloat>::zeros(KaiShape shape);

template<class T> KaiArray<T> KaiArray<T>::ones(KaiShape shape, T coef) {
	KInt size = shape.total_size();

	KaiArray<T> arr;
	arr.m_core->m_shape = shape;
	arr.m_core->m_mdata = new KaiArrayData<T>(size);

	T* pdata = arr.m_core->data();

	for (KInt n = 0; n < size; n++) pdata[n] = coef;

	return arr;
}

template KaiArray<KInt> KaiArray<KInt>::ones(KaiShape shape, KInt coef);
template KaiArray<KFloat> KaiArray<KFloat>::ones(KaiShape shape, KFloat coef);

template<class T> KaiArray<T> KaiArray<T>::to_host() {
	if (is_cuda()) {
		KaiArray<T> arr(shape());
		T* p_dst = arr.data_ptr();
		T* p_src = data_ptr();
		KInt size = arr.total_size();
		cudaMemcpy(p_dst, p_src, size * sizeof(T), cudaMemcpyDeviceToHost);
		return arr;
	}
	return *this;
}

template KaiArray<KInt> KaiArray<KInt>::to_host();
template KaiArray<KFloat> KaiArray<KFloat>::to_host();

template<class T>
T KaiArray<T>::get_at(KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8) const {
	KaiShape shape = m_core->m_shape;
	KInt nIndex = karr_math.eval_arr_index(shape, nth1, nth2, nth3, nth4, nth5, nth6, nth7, nth8);
	T* pdata = m_core->data();
	return pdata[nIndex];
}

template KInt KaiArray<KInt>::get_at(KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8) const;
template KFloat KaiArray<KFloat>::get_at(KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8) const;

template<class T>
T& KaiArray<T>::set_at(KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8) {
	KaiShape shape = m_core->m_shape;
	KInt nIndex = karr_math.eval_arr_index(shape, nth1, nth2, nth3, nth4, nth5, nth6, nth7, nth8);
	T* pdata = m_core->data();
	return pdata[nIndex];
}

template KInt& KaiArray<KInt>::set_at(KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8);
template KFloat& KaiArray<KFloat>::set_at(KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8);

template<class T>
void KaiArrayCore<T>::get_data(KInt nStart, KInt nCount, T* pBuffer) {
	T* pData = data();
	memcpy(pBuffer, pData + nStart, sizeof(T) * nCount);
}

template void KaiArrayCore<KInt>::get_data(KInt nStart, KInt nCount, KInt* pBuffer);
template void KaiArrayCore<KFloat>::get_data(KInt nStart, KInt nCount, KFloat* pBuffer);

template<> KString KaiArray<KInt>::element_type_name() { return "KInt"; }
template<> KString KaiArray<KFloat>::element_type_name() { return "KFloat"; }

template<class T>
KString KaiArrayCore<T>::desc() {
	char buf[128];
	sprintf_s(buf, 128, "<KaiArray<%s> %s 0x%llx>", KaiArray<T>::element_type_name().c_str(), m_shape.desc().c_str(), (KInt)(KHObject)this);
	return buf;
}

template KString KaiArrayCore<KInt>::desc();
template KString KaiArrayCore<KFloat>::desc();

template<> const char* KaiArray<KInt>::elem_format() { return " %lld"; }
template<> const char* KaiArray<KFloat>::elem_format() { return " %9.6f,"; }

template<> const char* KaiArray<KInt>::py_typename() { return "int"; }
template<> const char* KaiArray<KFloat>::py_typename() { return "float"; }

template<class T>
void KaiArray<T>::dump(KString sTitle, KBool bFull) {
	KString sTypename = KaiArray<T>::element_type_name();
	KString sCuda = is_cuda() ? "cuda" : "host";
	logger.Print("# KaiArray<%s> %s %s %s>", sTypename.c_str(), sTitle.c_str(), shape().desc().c_str(), sCuda.c_str());
	logger.PrintWait("_%s = [", sTitle.c_str());
	KaiArray<T> host_arr = to_host();
	T* pData = host_arr.data_ptr();

	if (total_size() <= 100) {
		for (KInt n = 0; n < total_size(); n++) {
			if (n % 10 == 0) logger.PrintWait("\n      ");
			logger.PrintWait(elem_format(), pData[n]);
		}
	}
	else {
		for (KInt n = 0; n < 50; n++) {
			if (n % 10 == 0) logger.PrintWait("\n      ");
			logger.PrintWait(elem_format(), pData[n]);
		}
		logger.PrintWait("\n      ...");
		for (KInt n = (total_size()-50)/10*10; n < total_size(); n++) {
			if (n % 10 == 0) logger.PrintWait("\n      ");
			logger.PrintWait(elem_format(), pData[n]);
		}
	}
	logger.Print("]");
	//logger.Print("%s = np.asarray(_%s, %s).reshape(%s)", sTitle.c_str(), sTitle.c_str(), py_typename(), shape().desc().c_str());
}

template void KaiArray<KInt>::dump(KString sTitle, KBool bFull);
template void KaiArray<KFloat>::dump(KString sTitle, KBool bFull);
