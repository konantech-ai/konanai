/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

depreciated;
/*
#pragma once

#include "component.h"
#include "../exec/exec_context.h"

class KaiDataset;
class KaiDataFeeder;

class KaiDataloader : public KaiComponent {
public:
	KaiDataloader(KaiSession* pSession, KaiDict kwArgs);
	KaiDataloader(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	virtual ~KaiDataloader();

	Ken_object_type get_type() { return Ken_object_type::dataloader;  }

	static KaiDataloader* HandleToPointer(KHObject hObject);
	static KaiDataloader* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiDataloader* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	void bind(KaiDataset* pDataset);
	void unbind(KaiDataset* pDataset);

	KString desc();

	//void getDataCountInfo(KaiExecContext* pContext, KaiDataset* pDataset, exec_mode mode);

protected:
	void m_init_data_count_info();
	void m_set_data_count_info();
	void m_reset_data_count_info();

protected:
	static int ms_checkCode;
	static KStrList ms_builtin;

	KaiDataset* m_pBoundDataset;
};

class KaiPlainDataloader : public KaiDataloader {
public:
	KaiPlainDataloader(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiPlainDataloader();
};
*/