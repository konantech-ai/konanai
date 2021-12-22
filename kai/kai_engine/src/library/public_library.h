/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "library.h"

class KaiPublicLibrary : public KaiLibrary {
public:
	KaiPublicLibrary(KaiSession* pSession, KString sLibName);
	virtual ~KaiPublicLibrary();

	static KaiPublicLibrary* HandleToPointer(KHObject hObject, KaiSession* pSession);

	void login(KString sUserName, KString sPassword);
	void logout();
	void close();
	void changePassword(KString sOldPassword, KString sNewPassword);

protected:
};
