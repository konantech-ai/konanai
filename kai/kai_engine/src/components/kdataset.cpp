/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "kdataset.h"
#include "../include/kai_api.h"
#include "../session/session.h"
#include "../utils/kutil.h"
#include "../math/karray.h"
#include "../math/khostmath.h"
#include "../exec/exec_context.h"
#include "../exec/callback.h"
#include "../nightly/nightly_utils.h"

int KaiDataset::ms_checkCode = 90289572;

KStrList KaiDataset::ms_builtin = { "csv_reader", "folder_classes", "feeding"};
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif
KaiDataset::KaiDataset(KaiSession* pSession, KaiDict kwArgs) : KaiComponent(pSession, Ken_component_type::dataset, Ken_object_type::dataset, kwArgs) {
	m_checkCode = ms_checkCode;
}

KaiDataset::KaiDataset(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo) : KaiComponent(pSession, Ken_component_type::dataset, Ken_object_type::dataset, pLib, pComponentInfo) {
	m_checkCode = ms_checkCode;
}

KaiDataset::~KaiDataset() {
	m_checkCode = 0;
}

KaiDataset* KaiDataset::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Dataset");

	KaiDataset* pDataset = (KaiDataset*)hObject;

	if (pDataset->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Dataset");
	if (pDataset->m_pSession != pSession) throw KaiException(KERR_SESSION_MISMTACH_FOR_COMPONENT_HANDLE, "Dataset");

	return pDataset;
}

KaiDataset* KaiDataset::HandleToPointer(KHObject hObject) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "Dataset");

	KaiDataset* pDataset = (KaiDataset*)hObject;

	if (pDataset->m_checkCode != ms_checkCode) throw KaiException(KERR_INVALIDL_HANDLE_USED, "Dataset");

	return pDataset;
}

void KaiDataset::GetBuiltinNames(KStrList* pslNames) {
	for (auto it = ms_builtin.begin(); it != ms_builtin.end(); it++) {
		pslNames->push_back(*it);
	}
}

KaiDataset* KaiDataset::CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs) {
	KaiDataset* pInstance = NULL;

	if (sBuiltin == "csv_reader") pInstance = new KaiCsvReaderDataset(pSession, kwArgs);
	else if (sBuiltin == "folder_classes") pInstance = new KaiFolderClassDataset(pSession, kwArgs);
	else if (sBuiltin == "folder_classes_recursive") pInstance = new KaiFolderClassRecursiveDataset(pSession, kwArgs);
	else if (sBuiltin == "feeding") pInstance = new KaiFeedingDataset(pSession, kwArgs);
	else if (sBuiltin == "") pInstance = new KaiDataset(pSession, kwArgs);

	if (!pInstance) throw KaiException(KERR_UNKNOWN_DATASET_SUBCLASS_NAME);

	pInstance->m_propDict["builtin"] = sBuiltin;
	pInstance->m_propDict["desc"] = pInstance->desc();

	return pInstance;
}

KString KaiDataset::desc() {
	char buf[128];
	KString sBuiltin = m_propDict["builtin"];

	sprintf_s(buf, 128, "<KaiComponent Dataset %s 0x%llx>", sBuiltin.c_str(), (KInt)(KHObject)this);

	return buf;
}

void KaiDataset::read_file(KString sDataFilePath) {
	m_bDirty = true;
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiDataset::setDataFeedingInfo(KaiModelInstance* pModelInst, KaiCallbackAgent* pCbAgent) {
	throw KaiException(KERR_INVALID_JOB_FOR_FEEDING_DATASET, "setContextInfo");
}

void KaiDataset::fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext) {
	KaiDict dsetInfo = pContext->get_component_property("dataset");
	KString sBuiltin = dsetInfo["builtin"];

	if (sBuiltin == "csv_reader") KaiCsvReaderDataset::fetchData(batch_index, xs, ys, pContext);
	else if (sBuiltin == "folder_classes") KaiFolderClassDataset::fetchData(batch_index, xs, ys, pContext);
	else if (sBuiltin == "folder_classes_recursive") KaiFolderClassRecursiveDataset::fetchData(batch_index, xs, ys, pContext);
	else if (sBuiltin == "feeding") KaiFeedingDataset::fetchData(batch_index, xs, ys, pContext);
	else throw KaiException(KERR_NEED_CODE_MODIFICATION, "unknown dataset builtin");
}

KaiDict KaiDataset::fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount) {
	KaiDict dsetInfo = pContext->get_component_property("dataset");
	KString sBuiltin = dsetInfo["builtin"];

	if (sBuiltin == "csv_reader") return KaiCsvReaderDataset::fetch_predict_data(pContext, pnDataCount);
	else if (sBuiltin == "folder_classes") return KaiFolderClassDataset::fetch_predict_data(pContext, pnDataCount);
	else if (sBuiltin == "folder_classes_recursive") return KaiFolderClassRecursiveDataset::fetch_predict_data(pContext, pnDataCount);
	else if (sBuiltin == "feeding") return KaiFeedingDataset::fetch_predict_data(pContext, pnDataCount);
	else throw KaiException(KERR_NEED_CODE_MODIFICATION, "unknown dataset builtin");
}

KaiCsvReaderDataset::KaiCsvReaderDataset(KaiSession* pSession, KaiDict kwArgs) : KaiDataset(pSession, kwArgs) {
}

KaiCsvReaderDataset::~KaiCsvReaderDataset() {
}

void KaiCsvReaderDataset::read_file(KString sDataFilePath) {
	// If this variable is false, then thid method is failed.
	KBool bHeaderExist = m_propDict["header_exist"];

	KaiList rawHeader;
	KaiList dataHeader;
	KaiList* pHeader = bHeaderExist ? &rawHeader : NULL;

	// CSV 파일을 읽고, pHeader(==&rawHeader)에 attributes 명을, rows에 데이터 값을 로드
	vector<vector<string>> rows = kutil.load_csv(KArgs::data_root + sDataFilePath, pHeader);

	KInt nRows = rows.size();
	KInt nCols = rows[0].size();
	KInt nDataCols = nCols;

	// Check the validation
	if (nRows < 1) {
		printf("error: %s(%u): data not found.\n", __FUNCTION__, __LINE__);
		throw KaiException(KERR_NO_DATASET_FOR_MODEL_EXEC);
	}

	KaiList to_onehot_list;
	
	// m_propDict["to_onehot"]이 존재하는 경우, KList 타입으로 [0]=0, [1]=3이 저장되어 있음
	// ([0]은 대상 column의 인덱스, [1]은 one-hot 벡터로 만들 때 벡터의 크기를 저장)
	// one-hot vector로 표현하는 경우, 첫 번째 'Sex' 컬럼을 'I(Infant)', 'M(Male)', 'F(Female)'로 분리해야 하므로 col 수를 +2
	// 결과적으로 nDataCols 값이 9->11로 변경됨
	if (m_propDict.find("to_onehot") != m_propDict.end()) {
		to_onehot_list = (KaiList) m_propDict["to_onehot"];

		for (auto it = to_onehot_list.begin(); it != to_onehot_list.end(); it++) {
			KaiList to_onehot = *it;
			nDataCols += (KInt) to_onehot[1] - 1;
		}
	}

	//KaiArray<KFloat> means = KaiArray<float>::zeros(KaiShape{ nDataCols });
	//KaiArray<KFloat> stds = KaiArray<float>::ones(KaiShape{ nDataCols });

	KBool bInputNorm = get_property("input_normalize", false);
	KInt temp_input_columns = get_property("temp_input_columns", 0);

	if ((KBool)get_property("input_normalize", false)) {
		//printf("input normalizetion here!!!");
		// 속성 보존: 정규화 적용 여부 플래그, 정규화에 사용된 평귭값 표준편차값
		// 부울 변수를 true로 설정한다. 
		// 이 값에 따라 (!found) 조건 처리 루프 앞-안-뒤에 정보 수집 후 재루프 통해 정규화 반영
		// 정규화에 이용된 필드별 평균, 표준편차는 시각화 때 벡터 원복, predict 때 정규화된 벡터 생성 위해 보존
		// 앞으로 dataset - dataloader 기능을 통합하고 datafeeder 역할만 하게 할 예정이므로 datafeeder가 책임질 부분임
		// 따라서 입력 정규화 및 복원은 datafeeder가 책임질 부분이므로 평균, 표준편차는 저장하지 않기로 한다. 
	}

	
	// Create a blank array
	KaiArray<float>csv_data = KaiArray<float>::zeros(KaiShape{ nRows, nDataCols });
	
	// one-hot vector 적용 여부에 따라 rawHeader로부터 dataHeader를 만들고 (적용일 경우 같거나 더 커지고, 비적용일 경우는 동일)
	// rows의 string 데이터들을 float으로 변환하여 csv_data에 저장
	for (KInt nc = 0, nd = 0; nc < nCols; nc++, nd++) {
		bool found = false;

		// one-hot 벡터로 변경하는 옵션이 활성화 되어있는 경우
		for (auto& it : to_onehot_list) {
			// [0]=대상 column 인덱스, [1]=변환할 one-hot 벡터 크기가 저장된 리스트
			KaiList to_onehot = it;

			// 현재 column 인덱스가 0일때 ('Sex' column)
			if (to_onehot[0] == nc) {
				// 변환할 one-hot 벡터 크기 (3)
				KInt nvec = to_onehot[1];
				// 빈 KaiDict (map<KString, KValue>) 생성
				KaiDict value_idx;

				// 전체 rows 수만큼 loop하여 0번째 'column'에 어떤 값들이 들었는지 확인
				for (KInt nr = 0; nr < nRows; nr++) {
					// nr번째 행의 데이터를 get
					vector<string> row = rows[nr];
					// nc번째 열의 데이터를 string으로 get
					KString sData = (KString) row[nc];

					// 해당 데이터(문자열)이 KaiDict에 없으면,
					if (value_idx.find(sData) == value_idx.end()) {
						// KaiDict 크기를 인덱스로 하여, 인덱스가 nvec(3)보다 크거나 같으면 Error
						KInt idx = value_idx.size();
						if (idx >= nvec)
							throw KaiException(KERR_ONE_HOT_DATA_EXCEED_SIZE);

						// sData 값(I,M,F 중 하나)에 idx를 저장
						value_idx[sData] = idx;

						// nc번째 열의 attribute 명을 string으로 get
						KString sValue = rawHeader[nc];

						// dataHeader에 'Sex/I', 'Sex/M', 또는 'Sex/F' 값을 Push
						if (pHeader)
							dataHeader.push_back(sValue+"/"+sData);
					}

					// 현재 문자(I,M,F 중)가 처음으로 나왔을 때의 column idx(0~2)를 get
					// ***** 데이터의 순서에 따라 dataHeader 및 csv_data의 입력 순서가 바뀜 *****
					KInt idx = value_idx[sData];

					// nr행 nd+idx열(0~2열) 값을 1로 지정
					csv_data.set_at(nr, nd + idx) = 1.0f;
				}

				// one-hot vector 크기만큼, nd의 값을 0->2로 변경 (for문에서 +1 하므로)
				nd += nvec - 1;
				found = true;
				break;	// one-hot vector는 한 번만 만들고 나면 break
			}
		}

		// one-hot vector 변경이 해당되지 않거나, 해당 column이 아닌 경우
		// (nd 값 변경되지 않았음)
		if (!found) {
			KFloat sum = 0, sqsum = 0;
			// 전체 행(row)에 대해 loop를 돌려서
			for (KInt nr = 0; nr < nRows; nr++) {
				vector<string> row = rows[nr];
				KFloat element = std::stof(row[nc]);
				csv_data.set_at(nr, nd) = element;
				if (bInputNorm) {
					sum += element;
					sqsum += element * element;
				}
			}
			if (bInputNorm && nd < temp_input_columns) {
				KFloat mean = sum / nRows;
				KFloat std = ::sqrt(sqsum / nRows - mean * mean);
				//means.set_at(nd) = mean;
				//stds.set_at(nd) = std;
				for (KInt nr = 0; nr < nRows; nr++) {
					KFloat element = csv_data.get_at(nr, nd);
					csv_data.set_at(nr, nd) = (element - mean) / std;
				}
			}

			// 'Sex' column이 아닌 경우 attribute의 string을 그대로 dataHeader에 input
			if (pHeader)
				dataHeader.push_back(rawHeader[nc]);
		}
	}

	/*
	m_propDict["inorm"] = bInputNorm;

	if (bInputNorm) {
		m_propDict["inorm_means"] = means.get_core();
		m_propDict["inorm_stds"] = stds.get_core();
	}
	*/

	
	// 생성한 원본 헤더(rawHeader)와 one-hot vector 적용 여부에 따라 변환된 헤더(dataHeader)를
	// m_propDict에 저장
	if (bHeaderExist) {
		// Original 9 attributes(string) :
		// {Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight, Rings}
		m_propDict["raw_header"] = rawHeader;

		// Renewed 11 attributes(string) :
		// {'Sex/I(Infant)', 'Sex/M(Male)', 'Sex/F(Female)', Length, Diameter, ... , Shell weight, Rings}
		m_propDict["header"] = dataHeader;
	}
	
	//m_data = m_data.to_cuda();
	KaiDict data = KaiDict{ {"#default", csv_data.get_core()} };

	// 생성된 csv_data의 core(핸들)과 rows/cols 정보를 저장
	m_propDict["total_count"] = nRows;
	m_propDict["vec_size"] = nDataCols;
	m_propDict["data"] = data;

	// 데이터 오염(변경)이 발생했음을 기록
	m_bDirty = true;
}

void KaiCsvReaderDataset::fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext) {
	KaiDict data_dict = pContext->get_property("data");

	KaiArray<KFloat> data = FARRAY(data_dict["#default"]);
	KInt nVecSize = data.axis_size(1);
	
	KaiList input_columns = pContext->get_property("input_columns");
	KaiList output_columns = pContext->get_property("output_columns");

	KInt nXCols = 0;
	KInt nYCols = 0;

	//if (input_columns.size() % 2 != 0) throw KaiException(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
	//if (output_columns.size() % 2 != 0) throw KaiException(KERR_BAD_OUTPUT_COLUMN_SPECIFICATION);
	
	for (auto& it: input_columns) {
		KaiList range = it;
		if (range.size() != 2) throw KaiException(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
		KInt nStart = range[0];
		KInt nCount = range[1];
		if (nStart < 0 || nCount <= 0 || nStart + nCount > nVecSize) throw KaiException(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
		nXCols += nCount;
	}
	
	for (auto& it : output_columns) {
		KaiList range = it;
		if (range.size() != 2) throw KaiException(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
		KInt nStart = range[0];
		KInt nCount = range[1];
		if (nStart < 0 || nCount <= 0 || nStart + nCount > nVecSize) throw KaiException(KERR_BAD_OUTPUT_COLUMN_SPECIFICATION);
		nYCols += nCount;
	}
	
	KInt mb_size = batch_index.total_size();

	KaiArray<KFloat> x_dat(KaiShape{ mb_size, nXCols });
	KaiArray<KFloat> y_dat(KaiShape{ mb_size, nYCols });

	KFloat* pSrc = data.data_ptr();
	KFloat* pxDst = x_dat.data_ptr();
	KFloat* pyDst = y_dat.data_ptr();
	
	for (KInt m = 0; m < mb_size; m++) {
		KInt dat_idx = batch_index.get_at(m);
		KInt nxcol = 0;
		KInt nycol = 0;

		// input_columns = {I, M} : I번부터 M개
		for (auto& it : input_columns) {
			KaiList range = it;
			KInt nStart = range[0];
			KInt nCount = range[1];
			memcpy(pxDst + m * nXCols + nxcol, pSrc + dat_idx * nVecSize + nStart, sizeof(KFloat) * nCount);
			nxcol += nCount;
		}

		// output_columns = {M, N} : M번부터 N개
		for (auto& it : output_columns) {
			KaiList range = it;
			KInt nStart = range[0];
			KInt nCount = range[1];
			memcpy(pyDst + m * nYCols + nycol, pSrc + dat_idx * nVecSize + nStart, sizeof(KFloat) * nCount);
			nycol += nCount;
		}
	}
	
	KaiMath* pMath = pContext->get_math();

	xs["#default"] = pMath->to_cuda(x_dat).get_core();
	ys["#default"] = pMath->to_cuda(y_dat).get_core();
}

KaiDict KaiCsvReaderDataset::fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount) {
	KString sFormat = pContext->get_property("input_format");
	KBool bMultiple = pContext->get_property("input_multiple");

	KaiList userData;

	if (bMultiple) userData = pContext->get_property("userdata");
	else userData.push_back(pContext->get_property("userdata"));

	KInt mb_size = (KInt)userData.size();
	*pnDataCount = mb_size;

	KaiList input_columns = pContext->get_property("input_columns");

	KInt nXCols = 0;

	//if (input_columns.size() % 2 != 0) throw KaiException(KERR_BAD_INPUT_COLUMN_SPECIFICATION);

	for (auto& it : input_columns) {
		KaiList range = it;
		if (range.size() != 2) throw KaiException(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
		KInt nStart = range[0];
		KInt nCount = range[1];
		nXCols += nCount;
	}

	KaiArray<KFloat> x_dat(KaiShape{ mb_size, nXCols });

	KFloat* pxDst = x_dat.data_ptr();

	for (KInt m = 0; m < mb_size; m++) {
		if (sFormat == "csv_vector") {
			KaiList vector = userData[m];
			if (vector.size() != nXCols) throw KaiException(KERR_BAD_SIZE_VECTOR_AS_USER_DATA);
			for (KInt n = 0; n < nXCols; n++) {
				*pxDst++ = (KFloat)vector[n];
			}
		}
		else if (sFormat == "csv_fields") {
			KaiList header = pContext->get_property("header");
			KaiDict fields = userData[m];
			for (KInt n = 0; n < nXCols; n++) {
				KString sField = header[n];
				if (fields.find(sField) == fields.end()) throw KaiException(KERR_BAD_FIELD_NOT_FOUND_IN_USER_DATA, sField);
				*pxDst++ = (KFloat)fields[sField];
			}
		}
		else if (sFormat == "raw_csv_fields") {
			KaiList header = pContext->get_property("header");
			KaiDict fields = userData[m];
			for (KInt n = 0; n < nXCols; n++) {
				KString sField = header[n];
				if (fields.find(sField) != fields.end()) {
					*pxDst++ = (KFloat)fields[sField];
				}
				else {
					std::size_t pos = sField.find('/');
					if (pos == std::string::npos) throw KaiException(KERR_BAD_FIELD_NOT_FOUND_IN_USER_DATA, sField);
					KString onehot_field = sField.substr(0, pos);
					KString onehot_value = sField.substr(pos + 1);
					if (fields.find(onehot_field) == fields.end()) throw KaiException(KERR_BAD_FIELD_NOT_FOUND_IN_USER_DATA, onehot_field);
					*pxDst++ = (fields[onehot_field] == onehot_value) ? 1.0f : 0.0f;
				}
			}
		}
	}

	KaiDict xs;
	xs["#default"] = pContext->get_math()->to_cuda(x_dat).get_core();
	return xs;
}

KaiFolderClassDataset::KaiFolderClassDataset(KaiSession* pSession, KaiDict kwArgs) : KaiDataset(pSession, kwArgs) {
}

KaiFolderClassDataset::~KaiFolderClassDataset() {
}

void KaiFolderClassDataset::read_file(KString sDataFilePath) {
	KString data_path = KArgs::data_root + sDataFilePath;
	KString sFileFormat = m_propDict["file_format"];

	if (sFileFormat != "image") throw KaiException(KERR_UNIMPEMENTED_YET);

	KaiShape dataShape = m_propDict["data_shape"];
	KaiShape imageShape = m_propDict["image_shape"];

	KaiList target_names = kutil.list_dir(data_path);
	KaiList filenames;
	KaiList cat_idxs;

	//logger.Print("[WARNING] reading data is temporally limited");

	for (KInt idx = 0; idx < (KInt) target_names.size(); idx++) {
		KString subpath = data_path + "/" + (KString)target_names[idx];
		KaiList fnames = kutil.list_dir(subpath);
		//KInt tmp_count = 0;
		for (auto& it: fnames) {
			KString fname = it;
			if (fname.length() < 4 || fname.substr(fname.length() - 4) != ".jpg") continue;
			filenames.push_back(it);
			cat_idxs.push_back(idx);
			//if (tmp_count++ >= 50) break;
		}
	}

	KInt data_count = cat_idxs.size();
	KInt data_size = dataShape.total_size();

	KaiArray<float> pixels = KaiArray<float>::zeros(KaiShape{ data_count, data_size });

	KFloat* pXs = pixels.data_ptr();

	for (KInt n = 0; n < (KInt) cat_idxs.size(); n++) {
		string filepath = data_path + '/' + (KString)target_names[(KInt)cat_idxs[n]] + '/' + (KString)filenames[n];
		kutil.load_jpeg_image_pixels(pXs, filepath, imageShape);
		pXs += data_size;
	}

	pixels = hostmath.div(hostmath.sub(pixels, 127.5f), 127.5f);

	KaiDict data;
	
	data["#default"] = pixels.get_core();
	data["idxs"] = cat_idxs;
	data["target_names"] = target_names;

	m_propDict["total_count"] = data_count;
	m_propDict["data"] = data;

	m_bDirty = true;
}

void KaiFolderClassDataset::fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext) {
	KaiDict data_dict = pContext->get_property("data");

	KaiArray<KFloat> data = FARRAY(data_dict["#default"]);
	KaiList idxs = data_dict["idxs"];
	KaiList target_names = data_dict["target_names"];

	KInt mb_size = batch_index.total_size();

	KaiShape dshape = pContext->get_property("data_shape");
	KaiShape xshape = dshape.insert_head(mb_size);
	KaiShape yshape = KaiShape{ mb_size, (KInt)target_names.size() };

	KaiArray<KFloat> x_dat(xshape);
	KaiArray<KFloat> y_dat = KaiArray<KFloat>::zeros(yshape);

	KFloat* pData = data.data_ptr();
	KFloat* pxDst = x_dat.data_ptr();
	KFloat* pyDst = y_dat.data_ptr();

	KInt xsize = dshape.total_size();
	KInt ysize = (KInt)target_names.size();

	for (KInt m = 0; m < mb_size; m++) {
		KInt dat_idx = batch_index.get_at(m);
		
		KFloat* pSrc = pData + dat_idx * xsize;
		KInt nIdx = idxs[dat_idx];

		memcpy(pxDst, pSrc, sizeof(KFloat) * xsize);
		pyDst[nIdx] = 1.0f;

		pxDst += xsize;
		pyDst += ysize;
	}

	KaiMath* pMath = pContext->get_math();

	xs["#default"] = pMath->to_cuda(x_dat).get_core();
	ys["#default"] = pMath->to_cuda(y_dat).get_core();
}

KaiDict KaiFolderClassDataset::fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount) {
	KString sFormat = pContext->get_property("input_format");
	//KBool bMultiple = pContext->get_property("input_multiple");

	KaiList userData;

	userData = pContext->get_property("userdata");
	
	KaiShape ishape = pContext->get_property("image_shape");
	KaiShape dshape = pContext->get_property("data_shape");

	KInt mb_size = (KInt)userData.size();
	KInt data_size = dshape.total_size();

	*pnDataCount = mb_size;

	KaiShape xshape = dshape.insert_head(mb_size);
	KaiArray<KFloat> pixels(xshape);
	KFloat* pxDst = pixels.data_ptr();

	for (KInt m = 0; m < mb_size; m++) {
		if (sFormat == "image_file_path") {
			KBool visualize = pContext->get_property("visualize", false);
			KString filepath = userData[m];
			kutil.load_jpeg_image_pixels(pxDst, filepath, ishape);
			pxDst += data_size;
		}
	}

	pixels = hostmath.div(hostmath.sub(pixels, 127.5f), 127.5f);

	KaiDict xs;
	xs["#default"] = pContext->get_math()->to_cuda(pixels).get_core();
	return xs;
}

KaiFolderClassRecursiveDataset::KaiFolderClassRecursiveDataset(KaiSession* pSession, KaiDict kwArgs) : KaiDataset(pSession, kwArgs) {
}

KaiFolderClassRecursiveDataset::~KaiFolderClassRecursiveDataset() {
}

void KaiFolderClassRecursiveDataset::read_file(KString sDataFilePath) {
	// Generate the data path
	KString data_path = KArgs::data_root + sDataFilePath;

	// Check the validation
	KString sFileFormat = m_propDict["file_format"];
	if (sFileFormat != "image")
		throw KaiException(KERR_UNIMPEMENTED_YET);

	// Set shapes(px) of the data and image to resize : {rows, cols, channels}
	KaiShape dataShape = m_propDict["data_shape"];
	KaiShape imageShape = m_propDict["image_shape"];

	// Declare a variable to store the target names
	KaiList target_names;

	// Get a list of domains : 3 domains (amazon, dslr, webcam)
	target_names.push_back( kutil.list_dir(data_path) );

	// Get a list of products : 31 products (back_pack, bike, bike_helmet, ..., tape_dispenser, trash_can)
	target_names.push_back( kutil.list_dir(data_path + "/" + (KString)((KaiList)target_names[0])[0] + "/images") );

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_FOLDER_CLASS_RECURSIVE_DATASET_READFILE)
	{
		printf("[TRACE]  %s(%u) {\n", __FUNCTION__, __LINE__);
		print_klist(target_names, "", 4, "target_names");
		printf("}\n\n");
	}
#endif

	// Get the sizes
	KInt domain_count  = ((KaiList)target_names[0]).size();
	KInt product_count = ((KaiList)target_names[1]).size();
	KInt cat_count     = domain_count + product_count;

	KaiList filenames;
	KaiList cat_idxs, domain_idxs, product_idxs;

	logger.Print("Now loading dataset...");

	//logger.Print("[WARNING] reading data is temporally limited");

	// Get domain indexes, product indexes, and filenames (Total 4110 files)
	for (KInt idx1=0; idx1<domain_count; ++idx1) {
		for (KInt idx2 = 0; idx2 < product_count; ++idx2) {
			KString subpath = data_path + "/" + (KString)((KaiList)target_names[0])[idx1] + "/images/" + (KString)((KaiList)target_names[1])[idx2];
			KaiList fnames = kutil.list_dir(subpath);
			
			for (auto& it : fnames) {
				KString fname = it;
				if (fname.length() < 4 || fname.substr(fname.length() - 4) != ".jpg")
					continue;
				filenames.push_back(it);

				// Save a domain index of the current file
				domain_idxs.push_back(idx1);

				// Save a product index of the current file
				product_idxs.push_back(idx2);
			}
		}
	}

	// Insert all indexes into cat_idxs
	cat_idxs.push_back(domain_idxs);
	cat_idxs.push_back(product_idxs);

	// Calculate the number of files and each data size
	KInt data_count = ((KaiList)cat_idxs[0]).size();	// The number of files : 4110
	KInt data_size = dataShape.total_size();			// Pixel count each image : rows * cols * channels

	// Create image data buffer
	KaiArray<float> pixels = KaiArray<float>::zeros(KaiShape{ data_count, data_size });

	// Load image data
	KFloat* pXs = pixels.data_ptr();
	for (KInt n=0; n<data_count; n++) {
		string filepath = data_path + '/' + (KString)((KaiList)target_names[0])[domain_idxs[n]] + "/images/" + (KString)((KaiList)target_names[1])[product_idxs[n]] + "/" + (KString)filenames[n];
		kutil.load_jpeg_image_pixels(pXs, filepath, imageShape);
		pXs += data_size;
	}

	// Calulcate : pixels = (each pixels - 127.5f) / 127.5f
	// In other words, adjust the ranges of each pixel value from [0,255] to [-1,1]
	pixels = hostmath.div(hostmath.sub(pixels, 127.5f), 127.5f);
	
	// Set properties of the dataset
	KaiDict data;

	data["#default"] = pixels.get_core();
	data["idxs"] = cat_idxs;
	data["target_names"] = target_names;

	m_propDict["total_count"] = data_count;
	m_propDict["data"] = data;

	m_bDirty = true;

	logger.Print("The dataset has been loaded.");
}

void KaiFolderClassRecursiveDataset::fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext) {
	KaiDict data_dict = pContext->get_property("data");

	KaiArray<KFloat> data = FARRAY(data_dict["#default"]);
	
	KaiList idxs = data_dict["idxs"];
	KaiList domain_idxs = idxs[0];
	KaiList product_idxs = idxs[1];
	
	KaiList target_names = data_dict["target_names"];
	KaiList domain_names = target_names[0];
	KaiList product_names = target_names[1];

	KInt domain_count = domain_names.size();
	KInt product_count = product_names.size();
	KInt target_count = domain_count + product_count;

	KInt mb_size = batch_index.total_size();

	KaiShape dshape = pContext->get_property("data_shape");
	KaiShape xshape = dshape.insert_head(mb_size);
	KaiShape yshape = KaiShape{ mb_size, target_count };

	KaiArray<KFloat> x_dat(xshape);
	KaiArray<KFloat> y_dat = KaiArray<KFloat>::zeros(yshape);

	KFloat* pData = data.data_ptr();
	KFloat* pxDst = x_dat.data_ptr();
	KFloat* pyDst = y_dat.data_ptr();

	KInt xsize = dshape.total_size();
	KInt ysize = target_count;

	// TRACE
#if (ACTIVATE_TRACE && TRACE_KAI_FOLDER_CLASS_RECURSIVE_DATASET_FETCHDATA)
	{
		printf("[TRACE]  %s(%u) {\n", __FUNCTION__, __LINE__);
		printf("    idxs.size()          = %lld\n", idxs.size());
		printf("    domain_idxs.size()   = %lld\n", domain_idxs.size());
		printf("    product_idxs.size()  = %lld\n", product_idxs.size());
		printf("\n");

		printf("    target_names.size()  = %lld\n", target_names.size());
		printf("    domain_names.size()  = %lld\n", domain_names.size());
		printf("    product_names.size() = %lld\n", product_names.size());
		printf("\n");

		printf("    target_count  = %lld\n", target_count);
		printf("    domain_count  = %lld\n", domain_count);
		printf("    product_count = %lld\n", product_count);
		printf("\n");

		printf("    mb_size = %lld\n", mb_size);
		printf("\n");

		printf("    dshape.desc().c_str() = %s\n", dshape.desc().c_str());
		printf("    xshape.desc().c_str() = %s\n", xshape.desc().c_str());
		printf("    yshape.desc().c_str() = %s\n", yshape.desc().c_str());
		printf("\n");

		printf("    x_dat.dim()           = %lld\n", x_dat.dim());
		for (KInt idxDim=0; idxDim<x_dat.dim(); ++idxDim)
			printf("    x_dat.axis_size(%lld)    = %lld\n", idxDim, x_dat.axis_size(idxDim));
		printf("    x_dat.mem_size()      = %lld\n", x_dat.mem_size());
		printf("    x_dat.total_size()    = %lld\n", x_dat.total_size());
		printf("\n");

		printf("    y_dat.dim()           = %lld\n", y_dat.dim());
		for (KInt idxDim = 0; idxDim < y_dat.dim(); ++idxDim)
			printf("    y_dat.axis_size(%lld)    = %lld\n", idxDim, y_dat.axis_size(idxDim));
		printf("    y_dat.mem_size()      = %lld\n", y_dat.mem_size());
		printf("    y_dat.total_size()    = %lld\n", y_dat.total_size());
		printf("\n");

		printf("    xsize = %lld\n", xsize);
		printf("    ysize = %lld\n", ysize);
		printf("}\n\n");
	}
#endif

	for (KInt m = 0; m < mb_size; m++) {
		KInt dat_idx = batch_index.get_at(m);

		KFloat* pSrc = pData + dat_idx * xsize;
		//KInt nIdx = idxs[dat_idx];
		KInt nIdx1 = domain_idxs[dat_idx];
		KInt nIdx2 = product_idxs[dat_idx];

		memcpy(pxDst, pSrc, sizeof(KFloat) * xsize);
		//pyDst[nIdx] = 1.0f;
		pyDst[nIdx1] = 1.0f;
		pyDst[domain_count + nIdx2] = 1.0f;

		pxDst += xsize;
		pyDst += ysize;
	}

	KaiMath* pMath = pContext->get_math();

	xs["#default"] = pMath->to_cuda(x_dat).get_core();
	ys["#default"] = pMath->to_cuda(y_dat).get_core();
}

KaiDict KaiFolderClassRecursiveDataset::fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount) {
	KString sFormat = pContext->get_property("input_format");
	//KBool bMultiple = pContext->get_property("input_multiple");

	KaiList userData;

	userData = pContext->get_property("userdata");

	KaiShape ishape = pContext->get_property("image_shape");
	KaiShape dshape = pContext->get_property("data_shape");

	KInt mb_size = (KInt)userData.size();
	KInt data_size = dshape.total_size();

	*pnDataCount = mb_size;

	KaiShape xshape = dshape.insert_head(mb_size);
	KaiArray<KFloat> pixels(xshape);
	KFloat* pxDst = pixels.data_ptr();

	for (KInt m = 0; m < mb_size; m++) {
		if (sFormat == "image_file_path") {
			KBool visualize = pContext->get_property("visualize", false);
			KString filepath = userData[m];
			kutil.load_jpeg_image_pixels(pxDst, filepath, ishape);
			pxDst += data_size;
		}
	}

	pixels = hostmath.div(hostmath.sub(pixels, 127.5f), 127.5f);

	KaiDict xs;
	xs["#default"] = pContext->get_math()->to_cuda(pixels).get_core();
	return xs;
}

KaiFeedingDataset::KaiFeedingDataset(KaiSession* pSession, KaiDict kwArgs) : KaiDataset(pSession, kwArgs) {
	//m_bInfoFetched = false;
	//m_bInFloat = true;
	//m_bOutFloat = true;
}

KaiFeedingDataset::~KaiFeedingDataset() {
}

/*
void KaiFeedingDataset::m_fetchFeedingInfo() {
	m_nDataCount = 0;

	throw KaiException(KERR_UNIMPEMENTED_YET);
	_CB_INFO cb;

	if (cbman.get_cb_info(m_propDict, Ken_cb_family::datafeed, (int)Ken_dataset_cbfunc::get_data_cnt, cb)) {
		KCbDatasetGetDataCount* pFunc = (KCbDatasetGetDataCount*)cb.pFunc;
		pFunc(cb.pInst, cb.pAux, &m_nDataCount);
	}
	else {
		throw KaiException(KERR_FEEDING_DATASET_WITHOUT_CALLBACK, "GetDataCount");
	}

	if (cbman.get_cb_info(m_propDict, Ken_cb_family::datafeed, (int)Ken_dataset_cbfunc::get_suffle_method, cb)) {
		KCbDatasetGetDataSuffleMethod* pFunc = (KCbDatasetGetDataSuffleMethod*)cb.pFunc;
		pFunc(cb.pInst, cb.pAux, &m_enumSuffleMethod);
	}
	else {
		m_enumSuffleMethod = Ken_data_suffle_method::random;
	}

	if (cbman.get_cb_info(m_propDict, Ken_cb_family::datafeed, (int)Ken_dataset_cbfunc::get_sec_data_cnt, cb)) {
		KCbDatasetGetSectionDataCount* pFunc = (KCbDatasetGetSectionDataCount*)cb.pFunc;
		pFunc(cb.pInst, cb.pAux, Ken_data_section::train, m_nDataCount, &m_nTrCount);
		pFunc(cb.pInst, cb.pAux, Ken_data_section::validate, m_nDataCount, &m_nVaCount);
		pFunc(cb.pInst, cb.pAux, Ken_data_section::test, m_nDataCount, &m_nTeCount);

		if (m_nTrCount + m_nTeCount > m_nDataCount) throw KaiException(KERR_TVT_DATA_CNT_EXCEEDS_TOTAL_DATA_CNT);
		if (m_nTrCount + m_nTeCount + m_nVaCount < m_nDataCount) m_nVaCount = m_nDataCount - (m_nTrCount + m_nTeCount);
	}
	else {
		m_nTrCount = (KInt)(m_nDataCount * 0.8);
		m_nTeCount = (KInt)(m_nDataCount * 0.1);
		m_nVaCount = m_nDataCount - m_nTrCount - m_nTeCount;
	}

	if (cbman.get_cb_info(m_propDict, Ken_cb_family::datafeed, (int)Ken_dataset_cbfunc::get_batch_size, cb)) {
		KCbDatasetGetBatchSize* pFunc = (KCbDatasetGetBatchSize*)cb.pFunc;
		pFunc(cb.pInst, cb.pAux, Ken_data_section::train, &m_nTrBatch);
		pFunc(cb.pInst, cb.pAux, Ken_data_section::validate, &m_nVaBatch);
		pFunc(cb.pInst, cb.pAux, Ken_data_section::test, &m_nTeBatch);
	}
	else {
		m_nTrBatch = 10;
		m_nVaBatch = 10;
		m_nTeBatch = 10;
	}

	if (cbman.get_cb_info(m_propDict, Ken_cb_family::datafeed, (int)Ken_dataset_cbfunc::get_field_spec, cb)) {
		KCbDatasetGetFieldSpec* pFunc = (KCbDatasetGetFieldSpec*)cb.pFunc;
		pFunc(cb.pInst, cb.pAux, true, "", &m_bInFloat, &m_ishape, &m_bInSeq, &m_nInTimesteps);
		pFunc(cb.pInst, cb.pAux, false, "", &m_bOutFloat, &m_oshape, &m_bOutSeq, &m_nOutTimesteps);
		if (!m_bInFloat || !m_bOutFloat) throw KaiException(KERR_UNIMPEMENTED_YET, "정수형 입출력 데이터는 언어처리 가면 사용");
	}

	m_bInfoFetched = true;

	//if (m_cbFuncMap.find(Ken_dataset_cbfunc::feed_int_data) == m_cbFuncMap.end()) {
	//	throw KaiException(KERR_INPUT_FEEDING_CALLBACK_UNDEFINED);
	//}

	//if (m_cbFuncMap.find(Ken_dataset_cbfunc::feed_float_data) == m_cbFuncMap.end()) {
	//	throw KaiException(KERR_OUTPUT_FEEDING_CALLBACK_UNDEFINED);
	//}
}
	*/

void KaiFeedingDataset::setDataFeedingInfo(KaiModelInstance* pModelInst, KaiCallbackAgent* pCbAgent) {
	//if (!m_bInfoFetched) m_fetchInfo();
	//m_fetchFeedingInfo();

	KInt nDataCount;
	Ken_data_suffle_method enumSuffleMethod = Ken_data_suffle_method::random;

	// Added by Hyung-jae, Son (2021-09-01)
	KString shuffle_method = pModelInst->get_property("data_split", "random");

	pCbAgent->get_data_count(&nDataCount, true);
	pCbAgent->get_data_suffle_method(&enumSuffleMethod, false);
	
	KaiMath* pMath = KaiMath::GetHostMath();
	KaiArray<KInt> total_index = pMath->arange(nDataCount);
	
	if (enumSuffleMethod != Ken_data_suffle_method::sequential) {
		pMath->shuffle(total_index);
		if (enumSuffleMethod != Ken_data_suffle_method::random) {
			logger.Print("Need to support Dataloder::data_split::%d option", (int)enumSuffleMethod);
		}
	}
	
	KInt nTrCount = 0; // (KInt)(nDataCount * 0.8);
	KInt nTeCount = 0; // (KInt)(nDataCount * 0.1);
	KInt nVaCount = 0; // nDataCount - nTrCount - nTeCount;
	
	pCbAgent->get_section_data_count(nDataCount, &nTrCount, &nTeCount, &nVaCount, false);
	
	if (nTrCount + nTeCount > nDataCount) throw KaiException(KERR_TVT_DATA_CNT_EXCEEDS_TOTAL_DATA_CNT);
	if (nTrCount + nTeCount + nVaCount < nDataCount) nVaCount = nDataCount - (nTrCount + nTeCount);

	KInt nTrBatch = 0; // 10;
	KInt nVaBatch = 0; // 10;
	KInt nTeBatch = 0; // 10;

	pCbAgent->get_section_batch_size(&nTrBatch, &nTeBatch, &nVaBatch, false);
	
	KaiDict tr_data, te_data, va_data;

	tr_data["data_start"] = (KInt)0;
	tr_data["data_count"] = nTrCount;
	tr_data["batch_size"] = nTrBatch;

	te_data["data_start"] = nTrCount;
	te_data["data_count"] = nTeCount;
	te_data["batch_size"] = nTeBatch;

	va_data["data_start"] = nTrCount + nTeCount;
	va_data["data_count"] = nVaCount;
	va_data["batch_size"] = nVaBatch;
	
	pModelInst->set_property("data_count", nDataCount);
	pModelInst->set_property("data_index", total_index.get_core());

	// TRACE
	//KaiArray<KInt> index_list = total_index;
	//printf("[TRACE] %s(%u): index_list[%lld] : { ", __FUNCTION__, __LINE__, index_list.total_size());
	//for (int i=0; i<10 && i<index_list.total_size(); ++i)
	//	printf("%lld ", index_list.get_at(i));
	//if (index_list.total_size() > 11)
	//	printf("... ");
	//if (index_list.total_size() > 10)
	//	printf("%lld ", index_list.get_at( index_list.total_size() - 1 ));
	//printf("}\n");

	pModelInst->set_property("tr_data", tr_data);
	pModelInst->set_property("te_data", te_data);
	pModelInst->set_property("va_data", va_data);

	KStrList inFieldNames;
	KStrList outFieldNames;
	
	pCbAgent->get_extra_fields(inFieldNames, outFieldNames, false);

	KaiDict input_fields;
	KaiDict output_fields;

	for (auto& it : inFieldNames) {
		KaiDict field_info;
		pCbAgent->get_field_spec(true, it, field_info, true);
		input_fields[it] = field_info;
	}

	for (auto& it : outFieldNames) {
		KaiDict field_info;
		pCbAgent->get_field_spec(false, it, field_info, true);
		output_fields[it] = field_info;
	}

	pModelInst->set_property("input_fields", input_fields);
	pModelInst->set_property("output_fields", output_fields);

	/*
	_CbInfo cb_info = m_cbFuncMap[Ken_dataset_cbfunc::feed_float_data];

	pModelInst->set_property("cb_feed_aux", (KInt)cb_info.m_pCbAux);
	pModelInst->set_property("cb_feed_func", (KInt)cb_info.m_pCbFunc);
	*/
}

KaiDict KaiFeedingDataset::fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiFeedingDataset::read_file(KString sDataFilePath) {
	throw KaiException(KERR_NO_DIRECT_READING_IN_FEEDING_DATASET);
}
