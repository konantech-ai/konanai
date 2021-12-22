/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#ifdef KAI2021_WINDOWS
#ifdef KAI_EXPORTS
#ifdef KAI_API
#undef KAI_API
#endif
#define KAI_API __declspec(dllexport)
#else
#define KAI_API __declspec(dllimport)
#endif
#else
#ifdef KAI_API
#undef KAI_API
#endif
#define KAI_API __attribute__((__visibility__("default")))
#endif

#include "kai_errors.h"
#include "kai_types.h"

extern "C" KAI_API KRetCode KAI_OpenCudaOldVersion(KHSession hSession); // 2020 예제들을 실행시키기 위해 올드 버전의 쿠다 연동 기능 오픈

extern "C" KAI_API KRetCode KAI_GetMissionNames(KHSession hSession, KStrList * pslMissionNames);

extern "C" KAI_API KRetCode KAI_GetPluginTypeNames(KHSession hSession, KStrList * pslPluginTypeNames);
extern "C" KAI_API KRetCode KAI_GetPluginNomNames(KHSession hSession, KString sPluginTypeName, KStrList * pslPluginNomNames);

extern "C" KAI_API KRetCode KAI_SetCudaOption(KHSession hSession, KStrList slTokens);
extern "C" KAI_API KRetCode KAI_GetCudaOption(KHSession hSession, int* pnDeviceCnt, int* pnDeviceNum, int* pnBlockSize, KInt* pnAvailMem, KInt * pnUsingMem);
extern "C" KAI_API KRetCode KAI_SetImageOption(KHSession hSession, KStrList slTokens);
extern "C" KAI_API KRetCode KAI_GetImageOption(KHSession hSession, KBool* pbImgScreen, KString* psImgFolder);
extern "C" KAI_API KRetCode KAI_SetSelectOption(KHSession hSession, KStrList slTokens);
extern "C" KAI_API KRetCode KAI_GetSelectDesc(KHSession hSession, KString sTypeName, KString* psTypeDesc);

extern "C" KAI_API KRetCode KAI_ExecMission(KHSession hSession, KString sMissionName, KString sExecMode);
