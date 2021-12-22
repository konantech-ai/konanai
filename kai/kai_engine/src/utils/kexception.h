/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../include/kai_errors.h"
#include "../include/kai_types.h"

class KaiException {
public:
	KaiException();
	KaiException(int nErrCode);
	KaiException(int nErrCode, KString sParam);
	KaiException(int nErrCode, KString sParam1, KString sParam2);

	virtual ~KaiException();

	KRetCode GetErrorCode();
	KString GetErrorMessage();

	static KString GetErrorMessage(KRetCode nErrorCode);

protected:
	int m_nErrCode;
	KString m_sParam1;
	KString m_sParam2;
};