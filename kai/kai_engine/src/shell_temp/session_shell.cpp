/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "../session/session.h"

#include "../../src2020/core/samples.h"
#include "../../src2020/core/dataset.h"
#include "../../src2020/core/log.h"
#include "../../src2020/core/util.h"
#include "../../src2020/cuda/cuda_conn.cuh"
#include "../../src2020/int_plugin/internal_plugin.h"

void KaiSession::OpenCudaOldVersion() {
	CudaConn::OpenCuda();
}

void KaiSession::GetMissionNames(KStrList* pslMissionNames) {
	Samples samples;
	int mission_count;
	const char** mission_names = samples.get_all_missions(&mission_count);

	pslMissionNames->clear();

	for (int n = 0; n < mission_count; n++) {
		pslMissionNames->push_back(mission_names[n]);
	}
}

void KaiSession::GetPluginTypeNames(KStrList* pslPluginTypeNames) {
	int_plugin_man.getTypeNames(pslPluginTypeNames);
}

void KaiSession::GetPluginNomNames(KString sPluginTypeName, KStrList* pslPluginNomNames) {
	int_plugin_man.getNomNames(sPluginTypeName, pslPluginNomNames);
}

void KaiSession::SetCudaOption(KStrList slTokens) {
	if (slTokens.size() == 2) {
		if (slTokens[1] == "on") CudaConn::OpenCuda();
		else if (slTokens[1] == "off") CudaConn::CloseCuda();
		else throw KaiException(KERR_SET_CUDA_OPTION_USAGE_ERROR);
	}
	else if (slTokens.size() == 3) {
		if (slTokens[1] == "device") {
			if (!CudaConn::SetDevice(std::stoi(slTokens[2]))) throw KaiException(KERR_SET_CUDA_BAD_DEVICE_NUMBER);
		}
		else if (slTokens[1] == "blocksize") CudaConn::SetBlockSize(std::stoi(slTokens[2]));
	}
	else throw KaiException(KERR_SET_CUDA_OPTION_USAGE_ERROR);
}

void KaiSession::GetCudaOption(int* pnDeviceCnt, int* pnDeviceNum, int* pnBlockSize, KInt* pnAvailMem, KInt* pnUsingMem) {
	if (pnDeviceCnt) *pnDeviceCnt = CudaConn::GetDeviceCount();
	if (pnDeviceNum) *pnDeviceNum = CudaConn::GetCurrDevice();
	if (pnBlockSize) *pnBlockSize = CudaConn::GetBlockSize();
	if (pnAvailMem) *pnAvailMem = CudaConn::getAvailMemSize();
	if (pnUsingMem) *pnUsingMem = CudaConn::getUsingMemSize();
}

void KaiSession::SetImageOption(KStrList slTokens) {
	if (slTokens.size() == 2) {
		if (slTokens[1] == "off") {
			Dataset::set_img_display_mode(false);
			Dataset::set_img_save_folder("");
		}
		else if (slTokens[1] == "screen") {
			Dataset::set_img_display_mode(true);
		}
		else throw KaiException(KERR_SET_IMAGE_OPTION_USAGE_ERROR);
	}
	else if (slTokens.size() == 3) {
		if (slTokens[1] == "screen") {
			if (slTokens[2] == "on") Dataset::set_img_display_mode(true);
			else if (slTokens[2] == "off") Dataset::set_img_display_mode(false);
			else throw KaiException(KERR_SET_IMAGE_OPTION_USAGE_ERROR);
		}
		else if (slTokens[1] == "save") {
			if (slTokens[2] == "off") Dataset::set_img_save_folder("");
			else Dataset::set_img_save_folder(slTokens[2]);
		}
		else throw KaiException(KERR_SET_IMAGE_OPTION_USAGE_ERROR);
	}
	else throw KaiException(KERR_SET_IMAGE_OPTION_USAGE_ERROR);
}

void KaiSession::GetImageOption(KBool* pbImgScreen, KString* psImgFolder) {
	if (pbImgScreen) *pbImgScreen = Dataset::get_img_display_mode();
	if (psImgFolder) *psImgFolder = Dataset::get_img_save_folder();
}

void KaiSession::SetSelectOption(KStrList slTokens) {
	if (slTokens.size() == 3) {
		string type_name = slTokens[1];
		string component_name = slTokens[2];

		int_plugin_man.set_plugin_component(type_name, component_name);
	}
	else throw KaiException(KERR_SET_SELECT_OPTION_USAGE_ERROR);
}

KString KaiSession::GetSelectDesc(KString sTypeName) {
	return int_plugin_man.introduce(sTypeName);
}

void KaiSession::ExecMission(KString sMissionName, KString sExecMode) {
	logger.Print("");
	logger.Print("[%s]", sMissionName.c_str());

	Samples samples;
	samples.execute(sMissionName.c_str(), sExecMode);
}
