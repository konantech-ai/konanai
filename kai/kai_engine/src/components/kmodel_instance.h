/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../components/component.h"
#include "../session/kcommon.h"
#include "../session/session.h"
#include "../include/kai_api.h"

class KaiSession;

class KaiModel;
class KaiDataset;
class KaiNetwork;
class KaiOptimizer;
class KaiExpression;

class KaiParameters;
class KaiMath;
class KaiCallbackAgent;

enum class exec_mode { em_train, em_validate, em_test, em_visualize };

class KaiModelInstance : public KaiComponent {
public:
	KaiModelInstance(KaiSession* pSession, KaiModel* pModel, KaiDict kwArgs);
	KaiModelInstance(KaiSession* pSession, KaiDict kwArgs);
	KaiModelInstance(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	virtual ~KaiModelInstance();

	Ken_object_type get_type() { return Ken_object_type::model_instance; }

	static KaiModelInstance* HandleToPointer(KHObject hObject);
	static KaiModelInstance* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiModelInstance* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	virtual void train(KaiDict kwArgs, KBool bAsync);
	virtual void test(KaiDict kwArgs, KBool bAsync);
	virtual void visualize(KaiDict kwArgs, KBool bAsync);

	virtual KaiList predict(KaiDict kwArgs);

	virtual KInt get_trained_epoch_count();

	KaiModelInstance* incRefCount();

	KaiSession* get_session() { return m_pSession; }

	void destroy();

	//KaiMath* getMath() { return m_pMath; }

	KInt get_int_property(KString sKey);
	KInt get_int_property(KString sKey, KInt nDefault);

	KaiValue get_property(KString sKey, KaiValue vDefault=KaiValue());
	KaiValue get_property(KStrList slKeys);

	void updateParams(KaiList layerParams, KaiList train_historyToAdd);

	KString desc();

protected:
	KaiCallbackAgent* m_pCbAgent;
	/*
	void m_extractModelInfo(KaiModel* pModel);
	void m_extractDatasetInfo(KaiDataset* pDataset);
	void m_extractNetworkInfo(KaiNetwork* pNetwork);
	void m_extractOptimizerInfo(KaiOptimizer* pOptimizer);
	void m_extractLossFuncInfo(KaiExpression* pLossFunc);
	void m_extractAccuracyFuncInfo(KaiExpression* pAccFunc);
	void m_extractEstVisualFuncInfo(KaiExpression* pEstVisualFunc);
	*/

	void m_setDataInfo();

	void m_createParameters(KaiNetwork* pNetwork, KaiDataset* pDataset, KaiOptimizer* pOptimizer);

protected:
	static void ms_trainThreadMain(void* aux);

protected:
	static int ms_checkCode;
	static KStrList ms_builtin;
};
