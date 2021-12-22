/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../include/kai_api.h"

class KaiSession;
class KaiExecContext;

/*
struct _CB_INFO {
	void* pInst;
	void* pFunc;
	void* pAux;
};
*/

class KaiCallbackManager {
public:
	KaiCallbackManager(KaiSession* pSession);
	virtual ~KaiCallbackManager();

	void set_cb_info(KaiDict& props, Ken_cb_family cb_family, int cb_kind, void* pCbInst, void* pCbFunc, void* pCbAux);

	/*
	KBool familyExist(KaiExecContext* pContext, Ken_cb_family cb_family);

	KBool get_cb_info(KaiDict props, Ken_cb_family cb_family, int cb_kind, _CB_INFO& cb);
	KBool get_cb_info(KaiExecContext* pContext, Ken_cb_family cb_family, int cb_kind, _CB_INFO& cb);
	*/

protected:
	KaiSession* m_pSession;
};

//extern KaiCallbackManager cbman;
