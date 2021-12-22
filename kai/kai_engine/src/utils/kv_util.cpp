/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kutil.h"

#ifdef XXX
#include <fstream>

#include <string.h>
#include <stdio.h>
#include <cstdarg>

//hs.cho
#ifdef KAI2021_WINDOWS
#include <direct.h>
#else
#define strcpy_s(a,b,c) !strncpy(a,c,b)
#define strtok_s strtok_r
#endif

KUtil kutil; 

/*
KString KUtil::to_decs_str(KaiValue value) {
	Ken_value_type val_type = value.type();
	switch (val_type) {
	case Ken_value_type::none:
		return "None";
	case Ken_value_type::kint:
		return to_string(value.m_core->m_value.m_int);
	case Ken_value_type::kfloat:
		return to_string(value.m_core->m_value.m_float);
	case Ken_value_type::string:
		return "'" + m_encode_esc(value.m_core->m_string) + "'";
	case Ken_value_type::list:
		return m_listDescription(*(KaiList*)value.m_core->m_value.m_pData);
	case Ken_value_type::dict:
		return m_dictDescription(*(KaiDict*)value.m_core->m_value.m_pData);
	case Ken_value_type::shape:
		return m_shapeDescription(*(KaiShape*)value.m_core->m_value.m_pData);
	case Ken_value_type::object:
	{
		KHObject hObject = value;
		Ken_object_type obj_type = hObject->get_type();
		if (obj_type == Ken_object_type::narray) {
			KaiArrayCore<KInt>* core = (KaiArrayCore<KInt>*)hObject;
			KaiArray<KInt> arr(core);
			return arrDescription(arr);
		}
		else if (obj_type == Ken_object_type::farray) {
			KaiArrayCore<KFloat>* core = (KaiArrayCore<KFloat>*)hObject;
			KaiArray<KFloat> arr(core);
			return arrDescription(arr);
		}
		else {
			throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "unknown kvalue type");
		}
	}
	}
	throw KaiException(KERR_INTERNAL_LOGIC_ERROR, "unknown kvalue type");
}
*/

KString KUtil::m_encode_esc(KString str) {
	if (str.find('\'') == string::npos && str.find('\\') == string::npos) return str;

	size_t pos = str.find('\\');

	while (pos != string::npos) {
		str = str.substr(0, pos - 1) + "\\" + str.substr(pos);
		pos = str.find('\\', pos + 2);
	}

	pos = str.find('\'');

	while (pos != string::npos) {
		str = str.substr(0, pos - 1) + "\\" + str.substr(pos);
		pos = str.find('\'', pos + 2);
	}

	return str;
}

KString KUtil::m_listDescription(KaiList list) {
	KString desc;
	KString delimeter = "[";

	for (auto it = list.begin(); it != list.end(); it++) {
		desc += delimeter;
		delimeter = ",";
		desc += to_decs_str(*it);
	}

	desc += "]";

	return desc;
}

KString KUtil::m_shapeDescription(KaiShape shape) {
	KString desc;
	KString delimeter = "<";

	for (KInt n = 0; n < shape.size(); n++) {
		desc += delimeter;
		delimeter = ",";
		desc += to_decs_str(shape[n]);
	}

	desc += ">";

	return desc;
}

KString KUtil::m_dictDescription(KaiDict dict) {
	KString desc;
	KString delimeter = "{";

	for (auto it = dict.begin(); it != dict.end(); it++) {
		desc += delimeter;
		delimeter = ",";

		string key = it->first;
		string value = to_decs_str(it->second);

		desc += "'" + key + "':" + value;
	}

	desc += "}";

	return desc;
}

template <class T> KString KUtil::arrDescription(KaiArray<T> arr) {
	KString desc = type_name(T(0)) + " ";

	desc += (arr.is_empty() ? "empty" : (arr.is_cuda() ? "cuda" : "host"));
	desc += " array";
	desc += arr.shape().desc();

	if (arr.is_empty()) return desc;
	if (arr.is_cuda()) arr = arr.to_host();

	KInt nSize = arr.total_size();

	if (nSize > 10) {
		T* pData = arr.data_ptr();
		desc += " [" + to_string(pData[0]);
		for (KInt n = 1; n < 4; n++) {
			desc += ", " + to_string(pData[n]);
		}
		desc += ", ...";
		for (KInt n = nSize-4; n < nSize; n++) {
			desc += ", " + to_string(pData[n]);
		}
		desc += "]";
	}
	else if (nSize > 0) {
		T* pData = arr.data_ptr();
		desc += " [" + to_string(pData[0]);
		for (KInt n = 1; n < nSize; n++) {
			desc += ", " + to_string(pData[n]);
		}
		desc += "]";
	}

	return desc;
}

template KString KUtil::arrDescription(KaiArray<KInt> arr);
template KString KUtil::arrDescription(KaiArray<KFloat> arr);

template <class T> void KUtil::save_array(FILE* fid, KaiArray<T> arr) {
	save_shape(fid, arr.shape());
	arr = arr.to_host();
	KInt size = arr.total_size();
	T* p = arr.data_ptr();
	if (fwrite(p, sizeof(T), size, fid) != size) throw KaiException(KERR_SAVE_ARRAY_FAILURE);
}

template void KUtil::save_array(FILE* fid, KaiArray<KInt> arr);
template void KUtil::save_array(FILE* fid, KaiArray<KFloat> arr);

KaiArray<KInt> KUtil::read_narray(FILE* fid) {
	KaiShape shape;
	read_shape(fid, shape);
	KaiArray<KInt> arr(shape);
	KInt size = arr.total_size();
	KInt* p = arr.data_ptr();
	if (fread(p, sizeof(KInt), size, fid) != size) throw KaiException(KERR_READ_ARRAY_FAILURE);
	return arr;
}

KaiArray<KFloat> KUtil::read_farray(FILE* fid) {
	KaiShape shape;
	read_shape(fid, shape);
	KaiArray<KFloat> arr(shape);
	KInt size = arr.total_size();
	KFloat* p = arr.data_ptr();
	if (fread(p, sizeof(KFloat), size, fid) != size) throw KaiException(KERR_READ_ARRAY_FAILURE);
	return arr;
}

template <class T> void KUtil::get_array_data(KHArray hArray, KInt nStart, KInt nCount, T* pBuffer) {
	KaiArray<T> arr = *(KaiArray<T>*)hArray;
	T* pData = arr.data_ptr();
	KInt size = arr.total_size();
	if (nStart < 0 || nStart >= size) throw KaiException(KERR_GET_ARRAY_DATA_OUT_OF_BOUND);
	if (nCount <= 0 || nStart+nCount > size) throw KaiException(KERR_GET_ARRAY_DATA_OUT_OF_BOUND);
	memcpy(pBuffer, pData + nStart, nCount * sizeof(nCount));
}

template void KUtil::get_array_data(KHArray hArray, KInt nStart, KInt nCount, KInt* pBuffer);
template void KUtil::get_array_data(KHArray hArray, KInt nStart, KInt nCount, KFloat* pBuffer);

/*
KaiDict& KUtil::dict_ref(KaiValue value) {
	return (KaiDict&)*(KaiDict*)value.m_core->m_value.m_pData;
}
*/

KFloat KUtil::mean(KaiList list) {
	KFloat count = (KFloat)list.size();
	KFloat sum = 0;

	for (auto& it : list) {
	for (auto& it : list) {
		sum += (KFloat)it;
	}

	return sum / count;
}

KFloat KUtil::sum(KaiList list) {
	KFloat sum = 0;

	for (auto& it : list) {
		sum += (KFloat)it;
	}

	return sum;
}

KaiDict KUtil::to_host(KaiDict dict, KaiMath* pMath) {
	KaiDict host_dict;

	for (auto& it : dict) {
		host_dict[it.first] = m_to_host(it.second, pMath);
	}

	return host_dict;
}

KaiList KUtil::to_host(KaiList list, KaiMath* pMath) {
	KaiList host_list;

	for (auto& it : list) {
		host_list.push_back(m_to_host(it, pMath));
	}

	return host_list;
}

KaiValue KUtil::m_to_host(KaiValue value, KaiMath* pMath) {
	KaiArray<KFloat> farr;
	KaiArray<KInt> narr;

	switch (value.type()) {
	case Ken_value_type::object:
		switch (((KHObject)value)->get_type()) {
		case Ken_object_type::farray:
			farr = FARRAY(value);
			farr = pMath->to_host(farr);
			return farr.get_core();
		case Ken_object_type::narray:
			narr = NARRAY(value);
			narr = pMath->to_host(narr);
			return narr.get_core();
		default:
			return value;
		}
	case Ken_value_type::dict:
		return to_host((KaiDict)value, pMath);
	case Ken_value_type::list:
		return to_host((KaiList)value, pMath);
	default:
		return value;
	}
}

KaiList KUtil::to_list(KStrList sList) {
	KaiList kList;
	for (auto& it : sList) kList.push_back(it);
	return kList;
}
#endif