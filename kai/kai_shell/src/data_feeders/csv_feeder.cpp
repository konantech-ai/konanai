/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "csv_feeder.h"
#include "../../../kai_engine/src/include/kai_errors.h"
#include "../../../kai_engine/src/nightly/nightly_utils.h"

//hs.cho
#ifdef KAI2021_WINDOWS
#else
#define strcpy_s(a,b,c) !strncpy(a,c,b)
#define strtok_s strtok_r
#endif


using std::ifstream;
using std::string;
using std::vector;

int CsvFeeder::ms_checkCode = 25186118;

typedef struct CsvCache {
	int     m_checkCode;	
	int     m_version;
	int     m_cntData;
	int     m_vecSize;
	int     m_headerSize;
	KaiList m_header;	// A list of KString
	KFloat* m_parData;

	CsvCache() : m_checkCode(0), m_version(0), m_cntData(0), m_vecSize(0), m_headerSize(0), m_parData(NULL) { }

} CsvCache;

CsvFeeder::CsvFeeder() : DataFeeder() {
	m_version       = 1;
	m_cntData       = 0;
	m_vecSize       = 0;
	m_headerSize    = 0;
	m_parData       = NULL;
	m_propDict.clear();
}

CsvFeeder::~CsvFeeder() {
	// Reset properties
	if (m_parData) {
		delete[] m_parData;
		m_parData = NULL;
	}

	m_cntData       = 0;
	m_vecSize       = 0;
	m_headerSize    = 0;

	m_propDict.clear();
}

KaiValue CsvFeeder::m_getProperty(KString sKey, KaiValue def) {
	if (m_propDict.find(sKey) != m_propDict.end())
		return m_propDict[sKey];

	return def;
}

void CsvFeeder::m_setProperty(KString sKey, KaiValue val) {
	m_propDict[sKey] = val;
}

void CsvFeeder::loadData(KString sCsvFilename, KString sCachePath, KaiDict kwArgs, KHSession hSession, KHDataset hDataset) {
	// Set user-defined properties
	m_propDict.clear();
	m_propDict = kwArgs;

	KBool bLoadCache = m_getProperty("load_cache", false);
	KBool bSaveCache = m_getProperty("save_cache", true);

	// Make a directory for using cache
	if (bSaveCache)
		Utils::mkdir(sCachePath);

	// Get a base filename of the CSV file
	KString sBaseFilename = sCsvFilename.substr(sCsvFilename.find_last_of("/\\") + 1);

	// Remove an extension
	KString::size_type posDot = sBaseFilename.find_last_of('.');
	sBaseFilename = sBaseFilename.substr(0, posDot);

	// Generate a cache filename
	KString sCacheFilename = sCachePath + "/" + sBaseFilename + ".dat";

	if (!bLoadCache || !m_loadCache(sCacheFilename)) {
		m_createDataset(sCsvFilename);

		if (bSaveCache)
			m_saveCache(sCacheFilename);
	}

	// Update the properties to dataset
	if (hSession && hDataset)
		KAI_Component_set_property(hSession, hDataset, m_propDict);
	else
		printf("warning: %s(%u): KAI_Component_set_property() failed.\n", __FUNCTION__, __LINE__);
}

KaiList CsvFeeder::getOuputFieldNames() {
	KaiList targetNames;

	KaiList dataHeader = m_propDict["header"];
	KaiList srcColumns = m_propDict["output_columns"];

	for (auto& it : srcColumns) {
		KaiList range = it;

		KInt idxStart = range[0];
		KInt nCount = range[1];

		for (KInt n = 0; n < nCount; n++) targetNames.push_back(dataHeader[idxStart+n]);
	}

	return targetNames;
}

void CsvFeeder::m_createDataset(KString sCsvFilename) {
	printf("Loading csv file...\n");

	// If this variable is false, then thid method is failed.
	KBool bHeaderExist = m_propDict["header_exist"];

	KaiList rawHeader;
	KaiList dataHeader;
	KaiList* pHeader = bHeaderExist ? &rawHeader : NULL;

	// CSV 파일을 읽고, pHeader(==&rawHeader)에 attributes 명을, rows에 데이터 값을 로드
	//vector<vector<string>> rows = kutil.load_csv(KArgs::data_root + filepath, pHeader);
	KaiList rows = m_readCsvFile(sCsvFilename, pHeader);

	KInt nRows = rows.size();
	KInt nCols = ((KaiList)rows[0]).size();
	KInt nDataCols = nCols;

	// Check the validation
	if (nRows < 1) {
		printf("error: %s(%u): data not found.\n", __FUNCTION__, __LINE__);
		THROW(KERR_NO_DATASET_FOR_MODEL_EXEC);
	}

	KaiList to_onehot_list;

	// m_propDict["to_onehot"]이 존재하는 경우, KList 타입으로 [0]=0, [1]=3이 저장되어 있음
	// ([0]은 대상 column의 인덱스, [1]은 one-hot 벡터로 만들 때 벡터의 크기를 저장)
	// one-hot vector로 표현하는 경우, 첫 번째 'Sex' 컬럼을 'I(Infant)', 'M(Male)', 'F(Female)'로 분리해야 하므로 col 수를 +2
	// 결과적으로 nDataCols 값이 9->11로 변경됨
	if (m_propDict.find("to_onehot") != m_propDict.end()) {
		to_onehot_list = (KaiList)m_propDict["to_onehot"];

		for (auto it = to_onehot_list.begin(); it != to_onehot_list.end(); it++) {
			KaiList to_onehot = *it;
			nDataCols += (KInt)to_onehot[1] - 1;
		}
	}

	KBool bInputNorm = m_getProperty("input_normalize", false);
	KInt temp_input_columns = m_getProperty("temp_input_columns", 0);

	if ((KBool)m_getProperty("input_normalize", false)) {
		//printf("input normalizetion here!!!");
		// 속성 보존: 정규화 적용 여부 플래그, 정규화에 사용된 평귭값 표준편차값
		// 부울 변수를 true로 설정한다. 
		// 이 값에 따라 (!found) 조건 처리 루프 앞-안-뒤에 정보 수집 후 재루프 통해 정규화 반영
		// 정규화에 이용된 필드별 평균, 표준편차는 시각화 때 벡터 원복, predict 때 정규화된 벡터 생성 위해 보존
		// 앞으로 dataset - dataloader 기능을 통합하고 datafeeder 역할만 하게 할 예정이므로 datafeeder가 책임질 부분임
		// 따라서 입력 정규화 및 복원은 datafeeder가 책임질 부분이므로 평균, 표준편차는 저장하지 않기로 한다. 
	}

	// Create a blank array
	m_parData = new KFloat[nRows * nDataCols];
	memset(m_parData, 0, sizeof(KFloat) * nRows * nDataCols);

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
					KaiList row = (KaiList)(rows[nr]);
					// nc번째 열의 데이터를 string으로 get
					KString sData = (KString)row[nc];

					// 해당 데이터(문자열)이 KaiDict에 없으면,
					if (value_idx.find(sData) == value_idx.end()) {
						// KaiDict 크기를 인덱스로 하여, 인덱스가 nvec(3)보다 크거나 같으면 Error
						KInt idx = value_idx.size();
						if (idx >= nvec)
							THROW(KERR_ONE_HOT_DATA_EXCEED_SIZE);

						// sData 값(I,M,F 중 하나)에 idx를 저장
						value_idx[sData] = idx;

						// nc번째 열의 attribute 명을 string으로 get
						KString sValue = rawHeader[nc];

						// dataHeader에 'Sex/I', 'Sex/M', 또는 'Sex/F' 값을 Push
						if (pHeader)
							dataHeader.push_back(sValue + "/" + sData);
					}

					// 현재 문자(I,M,F 중)가 처음으로 나왔을 때의 column idx(0~2)를 get
					// ***** 데이터의 순서에 따라 dataHeader 및 csv_data의 입력 순서가 바뀜 *****
					KInt idx = value_idx[sData];

					// nr행 nd+idx열(0~2열) 값을 1로 지정
					m_parData[nr*nDataCols + nd + idx] = 1.0f;
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
				KaiList row = KaiList(rows[nr]);
				KFloat element = std::stof(row[nc]);
				m_parData[nr*nDataCols + nd] = element;
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
					KFloat element = m_parData[nr*nDataCols + nd];
					m_parData[nr*nDataCols + nd] = (element - mean) / std;
				}
			}

			// 'Sex' column이 아닌 경우 attribute의 string을 그대로 dataHeader에 input
			if (pHeader)
				dataHeader.push_back(rawHeader[nc]);
		}
	}

	// 생성한 원본 헤더(rawHeader)와 one-hot vector 적용 여부에 따라 변환된 헤더(dataHeader)를
	// m_propDict에 저장
	if (bHeaderExist) {
		// Original 9 attributes(string) :
		// {Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight, Rings}
		m_propDict["raw_header"] = rawHeader;

		// Renewed 11 attributes(string) :
		// {'Sex/I(Infant)', 'Sex/M(Male)', 'Sex/F(Female)', Length, Diameter, ... , Shell weight, Rings}
		m_propDict["header"] = dataHeader;

		m_headerSize = (int)dataHeader.size();
		m_propDict["header_size"] = m_headerSize;
	}
	else
		m_headerSize = 0;

	//m_data = m_data.to_cuda();
	//KaiDict data = KaiDict{ {"default", csv_data.get_core()} };

	// 생성된 csv_data의 core(핸들)과 rows/cols 정보를 저장
	m_propDict["total_count"] = m_cntData = nRows;
	m_propDict["vec_size"] = m_vecSize = nDataCols;
	//m_propDict["data"] = data;
}

KaiList CsvFeeder::m_readCsvFile(KString sCsvFilename, KaiList* pHead) {
#ifdef KAI2021_WINDOWS
	std::replace(sCsvFilename.begin(), sCsvFilename.end(), '/', '\\');
#endif

	ifstream infile(sCsvFilename);

	if (infile.fail()) {
		//hs.cho
		//THROW(KERR_FILE_OPEN_FAILURE, sCsvFilename);
		THROW(KERR_FILE_OPEN_FAILURE);
	}

	vector<vector<string>> rows;

	string line;
	char buffer[1024];
	char* context = NULL;

	getline(infile, line);
	if (pHead) {
		if (strcpy_s(buffer, 1024, line.c_str()))
			THROW(KERR_ASSERT);
		char* token = strtok_s(buffer, ",", &context);
		while (token) {
			(*pHead).push_back(token);
			token = strtok_s(NULL, ",", &context);
		}
	}

	while (std::getline(infile, line)) {
		if (line[0] == ',') {
			line = "0" + line;
		}
		if (line[line.length() - 1] == ',') {
			line = line + "0";;
		}

		std::size_t pos = line.find(",,");
		while (pos != std::string::npos) {
			line = line.substr(0, pos + 1) + "0" + line.substr(pos + 1);
			pos = line.find(",,");
		}

		if (strcpy_s(buffer, 1024, line.c_str()))
			THROW(KERR_ASSERT);
		char* token = strtok_s(buffer, ",", &context);
		vector<string> row;
		while (token) {
			row.push_back(token);
			token = strtok_s(NULL, ",", &context);
		}
		rows.push_back(row);
	}

	infile.close();

	// Conversion
	KaiList csv_data;
	for (vector<vector<string>>::iterator it=rows.begin(); it!=rows.end(); ++it) {
		KaiList row_data;
		for (vector<string>::iterator it_row=it->begin(); it_row!=it->end(); ++it_row) {
			row_data.push_back((KString)*it_row);
		}
		csv_data.push_back(row_data);
	}

	return csv_data;
}

bool CsvFeeder::m_loadCache(KString sCacheFilename) {
	FILE* fid = Utils::fopen(sCacheFilename, "rb", false);

	if (fid == NULL)
		return false;

	printf("Loading cache data...\n");

	CsvCache cache;

	if (fread(&cache.m_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_version, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_cntData, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_vecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_headerSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);

	if (cache.m_checkCode != ms_checkCode) { fclose(fid); return false; }
	if (cache.m_version != m_version) { fclose(fid); return false; }

	for (int i=0; i<cache.m_headerSize; ++i) {
		size_t strlen = 0;
		if (fread(&strlen, sizeof(size_t), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);

		char szFieldName[256] = {0, };

		if (strlen > sizeof(szFieldName) / sizeof(char))
			THROW(KERR_FAILURE_ON_FILE_READ);

		if (fread(szFieldName, sizeof(char), strlen, fid) != strlen) THROW(KERR_FAILURE_ON_FILE_READ);

		cache.m_header.push_back( KString(szFieldName) );
	}

	int payload_size = cache.m_cntData * cache.m_vecSize;

	cache.m_parData = new KFloat[payload_size];
	std::memset(cache.m_parData, 0, sizeof(KFloat) * payload_size);

	if (fread(cache.m_parData, sizeof(KFloat), payload_size, fid) != payload_size) THROW(KERR_FAILURE_ON_FILE_READ);

	// Copy the cache data to this class
	m_propDict["total_count"] = m_cntData = cache.m_cntData;
	m_propDict["vec_size"] = m_vecSize = cache.m_vecSize;
	m_propDict["header_size"] = m_headerSize = cache.m_headerSize;
	if (m_headerSize > 0)
		m_propDict["header"] = cache.m_header;

	if (m_parData) {
		delete [] m_parData;
		m_parData = NULL;
	}

	m_parData = cache.m_parData;

	return true;
}

void CsvFeeder::m_saveCache(KString sCacheFilename) {
	printf("Saving cache data...\n");

	FILE* fid = Utils::fopen(sCacheFilename, "wb");

	if (fid == NULL)
		THROW(KERR_FAILURE_ON_FILE_SAVE);

	// Write headers
	if (fwrite(&ms_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_version, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_cntData, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_vecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_headerSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);

	KaiList header = m_propDict["header"];
	for (int i=0; i<m_headerSize; ++i) {
		KString sFieldName = (KString)header[i];
		size_t  strlen     = sFieldName.length();

		if (fwrite(&strlen, sizeof(size_t), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
		if (fwrite(sFieldName.c_str(), sizeof(char), strlen, fid) != strlen) THROW(KERR_FAILURE_ON_FILE_SAVE);
	}

	int payload_size = m_cntData * m_vecSize;
	if (fwrite(m_parData, sizeof(KFloat), payload_size, fid) != payload_size) THROW(KERR_FAILURE_ON_FILE_SAVE);

	fclose(fid);
}

KBool CsvFeeder::m_getDataCount(void* pAux, KInt* pnDataCount) {
	if (!pnDataCount)
		return false;

	*pnDataCount = m_cntData;
	return true;
}

KBool CsvFeeder::m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) {
	////////////////////////////////////////////////////////////////

	// User-defined code (To set shapes of feeding data)
	
	// feed_shape = is_seq ? { data_step, data_shape } : { data_shape }

	KaiList srcColumns      = m_propDict[bInput ? "input_columns" : "output_columns"];
	KInt    nSrcVecSize     = m_propDict["vec_size"];
	KInt    nTargetDataSize = 0;

	for (auto& it: srcColumns) {
		KaiList range = it;
		if (range.size() != 2)
			THROW(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
		KInt idxStart = range[0];
		KInt nCount   = range[1];
		if (idxStart < 0 || nCount <= 0 || idxStart + nCount > nSrcVecSize)
			THROW(KERR_BAD_INPUT_COLUMN_SPECIFICATION);

		nTargetDataSize += nCount;
	}

	if (nTargetDataSize > nSrcVecSize)
		THROW(KERR_BAD_INPUT_COLUMN_SPECIFICATION);

	KBool    is_float_type = true;
	KBool    is_seq        = false;
	KInt     data_step     = 1;
	KaiShape data_shape    = KaiShape{ nTargetDataSize };

	////////////////////////////////////////////////////////////////

	// For interworking with callbacks (do not modify)
	pFieldInfo->m_bIsFloat   = is_float_type;
	pFieldInfo->m_shape      = data_shape;
	pFieldInfo->m_bIsSeq     = is_seq;
	pFieldInfo->m_nTimesteps = data_step;

	////////////////////////////////////////////////////////////////

	return true;
}

KBool CsvFeeder::m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	THROW(KERR_INTERNAL_LOGIC_ERROR);
	return false;
}

KBool CsvFeeder::m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
	/*
	* Description :
	*  - shape :
	*      the array size of one data
	*      (e.g. If it is Abalone data, then the shape value is <10> or <1>)
	*  - nDatIndexs :
	*      A list of batch indexes
	*      (e.g. If nDatIndexs is {0,1,2}, then this method returns <3,10>or<3,1>-sized array)
	*/

	// Get the sizes of vectors
	KInt nSrcVecSize = m_propDict["vec_size"];
	KInt nDstVecSize = shape.total_size();
	
	// Get the range of the source data
	KaiList srcColumns = m_propDict[bInput ? "input_columns" : "output_columns"];

	for (KInt idxDstData=0; idxDstData<(KInt)nDatIndexs.size(); ++idxDstData) {
		KInt idxSrcData = nDatIndexs[idxDstData];

		KInt nCntCopy = 0;

		for (auto& it: srcColumns) {
			KaiList range = it;
			if (range.size() != 2)
				THROW(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
			KInt idxStart = range[0];
			KInt nCount   = range[1];
			if (idxStart < 0 || nCount <= 0 || idxStart + nCount > nSrcVecSize)
				THROW(KERR_BAD_INPUT_COLUMN_SPECIFICATION);

			memcpy(
				pfBuffer + idxDstData*nDstVecSize + nCntCopy,
				m_parData + idxSrcData*nSrcVecSize + idxStart,
				sizeof(KFloat)*nCount);

			nCntCopy += nCount;
		}

		// Check the validation
		if (nCntCopy != nDstVecSize)
			THROW(KERR_BAD_INPUT_COLUMN_SPECIFICATION);
	}

	return true;
}
