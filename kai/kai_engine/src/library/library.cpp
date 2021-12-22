/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "library.h"
#include "local_library.h"
#include "public_library.h"

int KaiLibrary::ms_checkCodeLocal = 72301411;
int KaiLibrary::ms_checkCodePublic = 90175813;

KaiLibrary::KaiLibrary(KaiSession* pSession, KString sLibName) {
	m_pOwnSession = pSession;
	m_sLibName = sLibName;
}

KaiLibrary::~KaiLibrary() {
}

KaiLibrary* KaiLibrary::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Library");

	KaiLibrary* pLib = (KaiLibrary*)hObject;

	if (pLib->m_checkCode != ms_checkCodeLocal && pLib->m_checkCode != ms_checkCodePublic) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Library");
	if (pLib->m_pOwnSession != pSession) throw KaiException(KERR_BAD_SESSION_USED, "Library");

	return pLib;
}
