/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kexception.h"

KaiException::KaiException() {
	m_nErrCode = 0;
}

KaiException::KaiException(int nErrCode) {
	m_nErrCode = nErrCode;
}

KaiException::KaiException(int nErrCode, KString sParam) {
	m_nErrCode = nErrCode;
	m_sParam1 = sParam;
}

KaiException::KaiException(int nErrCode, KString sParam1, KString sParam2) {
	m_nErrCode = nErrCode;
	m_sParam1 = sParam1;
	m_sParam2 = sParam2;
}

KaiException::~KaiException() {
}

KRetCode KaiException::GetErrorCode() {
	return m_nErrCode;
}

KString KaiException::GetErrorMessage() {
	return "Sorry, error message is not implemented yet...";
}

KString KaiException::GetErrorMessage(KRetCode nErrorCode) {
	return "Sorry, error message is not implemented yet...";
}
