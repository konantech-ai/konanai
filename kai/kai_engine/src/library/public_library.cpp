/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "public_library.h"

KaiPublicLibrary::KaiPublicLibrary(KaiSession* pSession, KString sLibName) : KaiLibrary(pSession, sLibName) {
	m_checkCode = ms_checkCodePublic;
	throw KaiException(KERR_WILL_BE_SUPPORTED_IN_NEXT_VERSION);
}

KaiPublicLibrary::~KaiPublicLibrary() {
	m_checkCode = 0;
}

KaiPublicLibrary* KaiPublicLibrary::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "PublicLibrary");
	KaiPublicLibrary* pPublicLib = (KaiPublicLibrary*)hObject;
	if (pPublicLib->m_checkCode != ms_checkCodePublic) throw KaiException(KERR_INVALIDL_HANDLE_USED, "PublicLibrary");
	if (pPublicLib->m_pOwnSession != pSession) throw KaiException(KERR_BAD_SESSION_USED, "PublicLibrary");
	return pPublicLib;
}

void KaiPublicLibrary::login(KString sUserName, KString sPassword) {
	throw KaiException(KERR_WILL_BE_SUPPORTED_IN_NEXT_VERSION);
}

void KaiPublicLibrary::logout() {
	throw KaiException(KERR_WILL_BE_SUPPORTED_IN_NEXT_VERSION);
}

void KaiPublicLibrary::close() {
	throw KaiException(KERR_WILL_BE_SUPPORTED_IN_NEXT_VERSION);
}

void KaiPublicLibrary::changePassword(KString sOldPassword, KString sNewPassword) {
	throw KaiException(KERR_WILL_BE_SUPPORTED_IN_NEXT_VERSION);
}
