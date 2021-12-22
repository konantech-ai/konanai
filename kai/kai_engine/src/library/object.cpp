/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
/*
#include "object.h"
#include "library.h"
#include "../utils/kutil.h"

int KaiObject::ms_checkCode = 34701918;

KaiObject::KaiObject() {
	m_checkCode = ms_checkCode;
}

KaiObject::~KaiObject() {
}

KaiObject* KaiObject::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Object");

	KaiObject* pObject = (KaiObject*)hObject;

	if (pObject->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Object");

	return pObject;
}


void KaiObject::dump(KString sTitle, KaiValue vObject) {
	KString desc = kutil.to_decs_str(vObject);
	logger.Print("%s %s", sTitle.c_str(), desc.c_str());
}
*/