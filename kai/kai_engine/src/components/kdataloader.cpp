/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
/*
#include "kdataloader.h"
#include "kdataset.h"
#include "../session/session.h"
#include "../math/kmath.h"
#include "../utils/kutil.h"

int KaiDataloader::ms_checkCode = 80192648;

KStrList KaiDataloader::ms_builtin = { "plain" };
//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif
KaiDataloader::KaiDataloader(KaiSession* pSession, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::dataloader, Ken_object_type::dataloader, kwArgs) {
	m_checkCode = ms_checkCode;

	m_pBoundDataset = NULL;

	m_init_data_count_info();
}

KaiDataloader::KaiDataloader(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiComponent(pSession, Ken_component_type::dataloader, Ken_object_type::dataloader, pLib, pComponentInfo) {
	m_checkCode = ms_checkCode;

	m_pBoundDataset = NULL;

	m_init_data_count_info();
}

KaiDataloader::~KaiDataloader() {
	m_checkCode = 0;
}

void KaiDataloader::m_init_data_count_info() {
	m_set_default("tr_ratio", 0.7f);
	m_set_default("va_ratio", 0.2f);
	m_set_default("te_ratio", 0.1f);

	m_set_default("tr_batch_size", 10);
	m_set_default("va_batch_size", 10);
	m_set_default("te_batch_size", 10);

	set_property("total_count", 0);

	set_property("tr_count", 0);
	set_property("te_count", 0);
	set_property("va_count", 0);
}

KaiDataloader* KaiDataloader::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Dataloader");

	KaiDataloader* pDataloader = (KaiDataloader*)hObject;

	if (pDataloader->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Dataloader");
	if (pDataloader->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Dataloader");

	return pDataloader;
}

KaiDataloader* KaiDataloader::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Dataloader");

	KaiDataloader* pDataloader = (KaiDataloader*)hObject;

	if (pDataloader->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Dataloader");

	return pDataloader;
}

void KaiDataloader::GetBuiltinNames(KStrList* pslNames) {
	for (auto it = ms_builtin.begin(); it != ms_builtin.end(); it++) {
		pslNames->push_back(*it);
	}
}

KaiDataloader* KaiDataloader::CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs) {
	KaiDataloader* pInstance = NULL;

	//KaiList aaa = kutil.list_dir(".");
	//KString saa = aaa.desc();

	if (sBuiltin == "plain") pInstance = new KaiPlainDataloader(pSession, kwArgs);
	else if (sBuiltin == "") pInstance = new KaiDataloader(pSession, kwArgs);

	if (!pInstance) throw KaiException(KERR_UNKNOWN_DATALOADER_SUBCLASS_NAME);

	pInstance->m_propDict["builtin"] = sBuiltin;
	pInstance->m_propDict["desc"] = pInstance->desc();

	return pInstance;
}

KString KaiDataloader::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf, 128, "<KaiComponent Dataloader %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

void KaiDataloader::bind(KaiDataset* pDataset) {
	KaiComponent::bind(pDataset, "dataset", true, true);

	m_pBoundDataset = pDataset;
	m_set_data_count_info();
}

void KaiDataloader::unbind(KaiDataset* pDataset) {
	KaiComponent::unbind(pDataset, true);
	m_pBoundDataset = NULL;
	m_reset_data_count_info();
}

void KaiDataloader::m_set_data_count_info() {
	KInt total_count = m_pBoundDataset->get_property("total_count");

	KFloat tr_ratio = m_propDict["tr_ratio"];
	KFloat te_ratio = m_propDict["te_ratio"];
	KFloat va_ratio = m_propDict["va_ratio"];

	KInt tr_count = KInt(total_count * tr_ratio);
	KInt te_count = KInt(total_count * te_ratio);
	KInt va_count = KInt(total_count * va_ratio);

	if (tr_count + te_count > total_count) throw KaiException(KERR_TVT_DATA_CNT_EXCEEDS_TOTAL_DATA_CNT);
	if (tr_count + te_count + va_count < total_count) va_count = total_count - (tr_count + te_count);

	set_property("total_count", total_count);

	set_property("tr_count", tr_count);
	set_property("te_count", te_count);
	set_property("va_count", va_count);

	KaiMath* pMath = KaiMath::GetHostMath();
	KaiArray<KInt> total_index = pMath->arange(total_count);

	KString sDataSplit = m_propDict["data_split"];
	if (sDataSplit != "sequential") {
		pMath->shuffle(total_index);
		if (sDataSplit != "random") {
			logger.Print("Need to support Dataloder::data_split::%s option", sDataSplit.c_str());
		}
	}

	set_property("total_index", total_index.get_core());
}

void KaiDataloader::m_reset_data_count_info() {
	set_property("total_count", 0);

	set_property("tr_count", 0);
	set_property("te_count", 0);
	set_property("va_count", 0);

	m_propDict.erase("total_index");
}

KaiPlainDataloader::KaiPlainDataloader(KaiSession* pSession, KaiDict kwArgs) : KaiDataloader(pSession, kwArgs) {
}

KaiPlainDataloader::~KaiPlainDataloader() {
}
*/