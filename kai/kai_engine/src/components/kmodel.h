/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "component.h"
#include "../exec/exec_context.h"

class KaiModelInstance;

class KaiModel : public KaiComponent {
public:
	KaiModel(KaiSession* pSession, KaiDict kwArgs);
	KaiModel(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	virtual ~KaiModel();

	Ken_object_type get_type() { return Ken_object_type::model; }

	static KaiModel* HandleToPointer(KHObject hObject);
	static KaiModel* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiModel* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	virtual void train(KaiDict kwArgs, KBool bAsync);
	virtual void test(KaiDict kwArgs, KBool bAsync);
	virtual void visualize(KaiDict kwArgs, KBool bAsync);
	
	virtual KaiList predict(KaiDict kwArgs);

	virtual KInt get_trained_epoch_count();

	virtual KaiModelInstance* get_instance(KaiDict kwArgs);

	KString desc();

protected:
	KBool m_isCleanModel();

protected:
	static int ms_checkCode;
	static KStrList ms_builtin;

	KaiModelInstance* m_pModelInstance;
};

class KaiBasicModel : public KaiModel {
public:
	KaiBasicModel(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiBasicModel();

protected:
};
