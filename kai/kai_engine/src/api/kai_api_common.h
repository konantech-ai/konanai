/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "../include/kai_api_shell.h"

#include "../session/session.h"

#include "../utils/kexception.h"
#include "../utils/kutil.h"

#define SESSION_OPEN()   KaiSession* pSession = NULL; try { pSession = KaiSession::HandleToPointer(hSession); } catch (...) { return KERR_INVALID_SESSION_HANDLE; } try { try {
#define SESSION_CLOSE()  return KRetOK; } \
		catch (KValueException ex) { throw KaiException(ex.m_nErrCode); } } \
		catch (KaiException ex) { pSession->SetLastError(ex); return ex.GetErrorCode(); } \
		catch (...) { return KERR_UNKNOWN_ERROR; }

#define NO_SESSION_OPEN()   try { try {
#define NO_SESSION_CLOSE()  return KRetOK; } \
		catch (KValueException ex) { throw KaiException(ex.m_nErrCode); } } \
		catch (KaiException ex) { return ex.GetErrorCode(); } \
		catch (...) { return KERR_UNKNOWN_ERROR; }

#define HANDLE_OPEN(cls, hObject, ptr) cls* ptr = cls::HandleToPointer(hObject, pSession)
#define NO_SESSION_HANDLE_OPEN(cls, hObject, ptr) cls* ptr = cls::HandleToPointer(hObject)
#define HANDLE_OPEN_OR_NULL(cls, hObject, ptr) if (hObject == NULL) return KRetOK; cls* ptr = cls::HandleToPointer(hObject, pSession)

#define POINTER_CHECK(ptr) if (ptr == NULL) return KERR_NULL_POINTER_USED_FOR_RETURN_VALUE;

