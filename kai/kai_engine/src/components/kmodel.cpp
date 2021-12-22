/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kmodel.h"
#include "kdataset.h"
#include "knetwork.h"
#include "koptimizer.h"
#include "kmodel_instance.h"
#include "../session/session.h"
#include "../exec/exec_context.h"
#include "../math/karray.h"
#include "../utils/kutil.h"

//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif


int KaiModel::ms_checkCode = 18266602;

KStrList KaiModel::ms_builtin = { "basic" };

KaiModel::KaiModel(KaiSession* pSession, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::model, Ken_object_type::model, kwArgs) {
	m_componentType = Ken_component_type::model;
	m_checkCode = ms_checkCode;

	m_pModelInstance = NULL;
}

KaiModel::KaiModel(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiComponent(pSession, Ken_component_type::model, Ken_object_type::model, pLib, pComponentInfo) {
	m_componentType = Ken_component_type::model;
	m_checkCode = ms_checkCode;

	m_pModelInstance = NULL;
}

KaiModel::~KaiModel() {
	m_pModelInstance->destroy();
	m_checkCode = 0;
}

KaiModel* KaiModel::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Model");

	KaiModel* pModel = (KaiModel*)hObject;

	if (pModel->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Model");
	if (pModel->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Model");

	return pModel;
}

KaiModel* KaiModel::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Model");

	KaiModel* pModel = (KaiModel*)hObject;

	if (pModel->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Model");

	return pModel;
}

void KaiModel::GetBuiltinNames(KStrList* pslNames) {
	for (auto it = ms_builtin.begin(); it != ms_builtin.end(); it++) {
		pslNames->push_back(*it);
	}
}

KaiModel* KaiModel::CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs) {
	KaiModel* pInstance = NULL;

	if (sBuiltin == "basic") pInstance = new KaiBasicModel(pSession, kwArgs);
	else if (sBuiltin == "") pInstance = new KaiModel(pSession, kwArgs);

	if (!pInstance) throw KaiException(KERR_UNKNOWN_DATASET_SUBCLASS_NAME);

	pInstance->m_propDict["builtin"] = sBuiltin;
	pInstance->m_propDict["desc"] = pInstance->desc();

	return pInstance;
}

void KaiModel::train(KaiDict kwArgs, KBool bAsync) {
	m_pModelInstance = get_instance(kwArgs);
	m_pModelInstance->train(kwArgs, bAsync);
}

void KaiModel::test(KaiDict kwArgs, KBool bAsync) {
	m_pModelInstance = get_instance(kwArgs);
	m_pModelInstance->test(kwArgs, bAsync);
}

void KaiModel::visualize(KaiDict kwArgs, KBool bAsync) {
	m_pModelInstance = get_instance(kwArgs);
	m_pModelInstance->visualize(kwArgs, bAsync);
}

KaiList KaiModel::predict(KaiDict kwArgs) {
	m_pModelInstance = get_instance(kwArgs);
	return m_pModelInstance->predict(kwArgs);
}

KInt KaiModel::get_trained_epoch_count() {
	return m_pModelInstance ? m_pModelInstance->get_trained_epoch_count() : 0;
}

KaiModelInstance* KaiModel::get_instance(KaiDict kwArgs) {
	if (m_pModelInstance && m_isCleanModel()) return m_pModelInstance;
	if (m_pModelInstance) m_pModelInstance->destroy();

	m_pModelInstance = new KaiModelInstance(m_pSession, this, kwArgs);

	return m_pModelInstance;
}

KBool KaiModel::m_isCleanModel() {
	if (m_bDirty) return false;
	return m_pSession->areCleanBoundComponents(this) ;
}

KString KaiModel::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf, 128, "<KaiComponent Model %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

KaiBasicModel::KaiBasicModel(KaiSession* pSession, KaiDict kwArgs) : KaiModel(pSession, kwArgs) {
	static KString relations[] = { "dataset", "network", "loss_exp", "accuracy_exp", "visualize_exp", "predict_exp", "optimizer" };

	int size = sizeof(relations) / sizeof(relations[0]);

	for (int n = 0; n < size; n++) {
		KString relation = relations[n];
		if (m_propDict.find(relation) != m_propDict.end()) {
			KaiComponent* pComponent = KaiComponent::HandleToPointer(m_propDict[relation]);
			KaiComponent::bind(pComponent, relation, true, true);
		}
	}
}

KaiBasicModel::~KaiBasicModel() {
}
