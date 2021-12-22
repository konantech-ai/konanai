/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
// Reference : (English notation about the hierarchy of folder)
// https://oshiete.goo.ne.jp/qa/164562.html

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

// Check the C++ compiler, STL version, and OS type.
#if (defined(_MSC_VER) && (_MSVC_LANG >= 201703L || _HAS_CXX17)) || (defined(__GNUC__) && (__cplusplus >= 201703L))
	// ISO C++17 Standard (/std:c++17 or -std=c++17)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
//hs.cho
	#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
	#include <experimental/filesystem> // C++14
	namespace fs = std::experimental::filesystem;
#endif

//#elif _WIN32 || _WIN64
//	// Microsoft Windows x86/64
//	#include <Windows.h>
//#elif __unix__ || __linux__
//	// UNIX or Linux system
//	// Not supported yet
//	#include "../../../kai_engine/src/nightly/findfirst.h"
//#endif

#include "folder_class_feeder.h"
#include "../../../kai_engine/src/include/kai_errors.h"
#include "../../../kai_engine/src/nightly/nightly_utils.h"

using std::ifstream;
using std::string;
using std::vector;

int FolderClassFeeder::ms_checkCode = 25186118;

typedef struct FolderClassCache {
	int     m_checkCode;	
	int     m_version;
	int     m_cntData;
	int     m_vecSize;
	int     m_inputVecSize;
	int     m_outputVecSize;
	int     m_headerSize;
	KaiList m_header;	// A list of KString
	KFloat* m_parData;

	FolderClassCache() : m_checkCode(0), m_version(0), m_cntData(0), m_vecSize(0), m_inputVecSize(0), m_outputVecSize(0), m_headerSize(0), m_parData(NULL) { }

} FolderClassCache;

FolderClassFeeder::FolderClassFeeder() : DataFeeder() {
	m_version       = 1;
	m_cntData       = 0;
	m_vecSize       = 0;
	m_inputVecSize  = 0;
	m_outputVecSize = 0;
	m_headerSize    = 0;
	m_parData       = NULL;
	m_propDict.clear();
}

FolderClassFeeder::~FolderClassFeeder() {
	// Reset properties
	if (m_parData) {
		delete[] m_parData;
		m_parData = NULL;
	}

	m_cntData       = 0;
	m_vecSize       = 0;
	m_inputVecSize  = 0;
	m_outputVecSize = 0;
	m_headerSize    = 0;

	m_propDict.clear();
}

KaiValue FolderClassFeeder::getProperty(KString sKey, KaiValue def) {
	if (m_propDict.find(sKey) != m_propDict.end())
		return m_propDict[sKey];

	return def;
}

void FolderClassFeeder::setProperty(KString sKey, KaiValue val) {
	m_propDict[sKey] = val;
}

KaiList FolderClassFeeder::getTargetNames() {
	KaiList targetNames = m_propDict["header"];
	return targetNames;
}

void FolderClassFeeder::loadData(KString sSrcFolderPath, KString sCachePath, KaiDict kwArgs, KHSession hSession, KHDataset hDataset) {
	// Set user-defined properties
	m_propDict.clear();
	m_propDict = kwArgs;

	KBool bLoadCache = getProperty("load_cache", false);
	KBool bSaveCache = getProperty("save_cache", true);

	// Make a directory for using cache
	if (bSaveCache)
		Utils::mkdir(sCachePath);

	// Get a base filename of the FOLDER_CLASS file
	KString sBaseFilename = sSrcFolderPath.substr(sSrcFolderPath.find_last_of("/\\") + 1);

	// Remove an extension
	KString::size_type posDot = sBaseFilename.find_last_of('.');
	sBaseFilename = sBaseFilename.substr(0, posDot);

	// Generate a cache filename
	KString sCacheFilename = sCachePath + "/" + sBaseFilename + ".dat";

	if (!bLoadCache || !m_loadCache(sCacheFilename)) {
		m_createDataset(sSrcFolderPath);

		if (bSaveCache)
			m_saveCache(sCacheFilename);
	}

	// Update the properties to dataset
	if (hSession && hDataset)
		KAI_Component_set_property(hSession, hDataset, m_propDict);
	else
		printf("warning: %s(%u): KAI_Component_set_property() failed.\n", __FUNCTION__, __LINE__);
}

void FolderClassFeeder::m_createDataset(KString sFolderPath) {
	////////////////////////////////////////////////////////////////
	//// Get user-defined properties ///////////////////////////////
	////////////////////////////////////////////////////////////////

	// Check the file format
	KString sFileFormat = (KString)m_propDict["file_format"];
	KString sFilterString;
	if (sFileFormat == "image")
		sFilterString = ".bmp;.jpg;.jpeg;.png";
	else {
		printf("error: %s(%u): Unsupported format\n", __FUNCTION__, __LINE__);
		THROW(KERR_UNIMPEMENTED_YET);
	}

	// Set shapes(px) of the resized image : {rows, cols, channels}
	KaiShape resizedImageShape = m_propDict["image_shape"];

	// Set properties of the resized image
	KInt resizedImageSize = resizedImageShape.total_size();

	////////////////////////////////////////////////////////////////
	//// Get a list of classes(folder names) ///////////////////////
	////////////////////////////////////////////////////////////////

	printf("Create a dataset using %s...\n", "FolderClassFeeder");

	// Get a list of all files recursively
	std::vector<KString> vsFilenames;
	if (!m_getFileList(sFolderPath, vsFilenames, true)) {
		printf("error: %s(%u): m_getFileList() failed.\n", __FUNCTION__, __LINE__);
		THROW(KERR_FILE_OPEN_FAILURE);
	}

	// Filename filtering
	vsFilenames = m_filtering(vsFilenames, sFilterString);

	// Split each filename into sub-path names
	std::vector<KStrList> vsSplitFilenames = m_splitPath(vsFilenames);

	// Generate a list of classes based on each sub-path name
	std::vector< std::map<KString,KInt> > vmClassList = m_generateClassList(vsSplitFilenames);

	// Set the total file count
	m_propDict["total_count"] = m_cntData = vsFilenames.size();

	////////////////////////////////////////////////////////////////
	//// Calculate intput/output_columns and vector size ///////////
	////////////////////////////////////////////////////////////////

	// Set the vector size to create
	// Input vector  : [ image pixels ]
	// Output vector : [ Level-1 classes ] [ Level-2 Classes ] ... (execludes file that are not contained in any folder.)
	// Each class    : [ folder name 1 ] [ folder name 2 ] ... (Union-type integration)
	KInt inputVecSize  = resizedImageSize;
	KInt outputVecSize = 0;
	for (std::vector< std::map<KString,KInt> >::size_type i=0; i<vmClassList.size(); ++i) {
		outputVecSize += vmClassList[i].size();
	}

	if (outputVecSize <= 0) {
		printf("error: %s(%u): The class(folder) doesn't exist in that location.\n", __FUNCTION__, __LINE__);
		THROW(KERR_BAD_SIZE_VECTOR_AS_USER_DATA);
	}

	// Set the vector sizes
	m_inputVecSize  = inputVecSize;
	m_outputVecSize = outputVecSize;
	m_propDict["vec_size"] = m_vecSize = (m_inputVecSize + m_outputVecSize);

	// Set values of "input/output_columns"
	KaiList input_columns;
	input_columns.push_back( KaiList{(KInt)0, (KInt)m_inputVecSize} );
	KaiList output_columns;
	output_columns.push_back( KaiList{(KInt)m_inputVecSize, (KInt)m_outputVecSize} );

	m_propDict["input_columns"]  = input_columns;
	m_propDict["output_columns"] = output_columns;

	////////////////////////////////////////////////////////////////
	//// Allocate the buffer memory ////////////////////////////////
	////////////////////////////////////////////////////////////////

	// Set the buffer size to create (= total data size)
	KInt buffSize = (KInt)(m_cntData * m_vecSize);

	// Create image data buffer
	if (m_parData) {
		delete [] m_parData;
		m_parData = NULL;
	}
	m_parData = new KFloat[buffSize];
	std::memset(m_parData, 0, sizeof(KFloat)*buffSize);

	////////////////////////////////////////////////////////////////
	//// Generate input data ///////////////////////////////////////
	////////////////////////////////////////////////////////////////

	KFloat* pBuff = m_parData;
	for (KStrList::size_type i=0; i<vsFilenames.size(); ++i) {
		// Load and resize images
		cv_load_and_resize_image(pBuff, sFolderPath + "/" + vsFilenames[i], resizedImageShape[0], resizedImageShape[1], resizedImageShape[2]);
		
		// Normalize : (each pixels - 127.5f) / 127.5f
		// In other words, adjust the ranges of each pixel value from [0,255] to [-1,1]
		KFloat* pPixel = pBuff;
		for (KInt j=0; j<resizedImageSize; ++j) {
			*pPixel = ( (*pPixel) - 127.5f ) / 127.5f;
			++pPixel;
		}

		pBuff += m_vecSize;
	}

	////////////////////////////////////////////////////////////////
	//// Generate output data //////////////////////////////////////
	////////////////////////////////////////////////////////////////
	
	// Convert the class list to vector type
	std::vector<KStrList> vsClassList;
	for (std::vector< std::map<KString,KInt> >::const_iterator it_depth=vmClassList.cbegin(); it_depth!=vmClassList.cend(); ++it_depth) {
		KStrList vsClasses;
		for (std::map<KString,KInt>::const_iterator it_class=it_depth->cbegin(); it_class!=it_depth->cend(); ++it_class) {
			vsClasses.push_back(it_class->first);
		}
		vsClassList.push_back(vsClasses);
	}

	for (int idxFile=0; idxFile<(int)vsSplitFilenames.size(); ++idxFile) {
		const KStrList& vsSplitFilename = vsSplitFilenames[idxFile];

		int classOffset = 0;

		// Exclude filename
		int cntClass = (int)vsSplitFilename.size() - 1;

		for (int idxDepth=0; idxDepth<cntClass; ++idxDepth) {
			const KString& sSplitWord = vsSplitFilename[idxDepth];
			const KStrList& vsClasses = vsClassList[idxDepth];

			// Search for the word in the list of pre-created class names
			int idxFind = -1;
			for (int idxClass=0; idxClass<(int)vsClasses.size(); ++idxClass) {
				if (vsClasses[idxClass] == sSplitWord) {
					idxFind = idxClass;
					break;
				}
			}

			// Check the validation
			if (idxFind < 0) {
				printf("error: %s(%u): Failed to search for string \"%s\".\n", __FUNCTION__, __LINE__, sSplitWord.c_str());
				THROW(KERR_UNKNOWN_DATASET_SUBCLASS_NAME);
			}

			// Calculate an index of the class(folder name)
			int answerOffset = idxFind;

			// Record in output vector
			m_parData[ idxFile*m_vecSize + (int)inputVecSize + classOffset + answerOffset ] = 1.0f;

			// Update the offset
			classOffset += vsClasses.size();
		}
	}

	////////////////////////////////////////////////////////////////
	//// Generate a list of headers for output data ////////////////
	////////////////////////////////////////////////////////////////

	KBool bHeaderExist = m_propDict["header_exist"];

	if (bHeaderExist) {
		KaiList classHeader;

		for (int idxDepth=0; idxDepth<(int)vsClassList.size(); ++idxDepth) {
			const KStrList& vsClasses = vsClassList[idxDepth];

			// No class
			//char szClassName[128] = {0, };
			//snprintf(szClassName, sizeof(szClassName)/sizeof(char), "[File/depth-%d]", idxDepth);
			//classHeader.push_back(KString(szClassName));

			for (KStrList::const_iterator it=vsClasses.cbegin(); it!=vsClasses.cend(); ++it) {
				classHeader.push_back(*it);
			}
		}

		//m_propDict["raw_header"] = classHeader;
		m_propDict["header"] = classHeader;

		m_headerSize = (int)classHeader.size();
		m_propDict["header_size"] = m_headerSize;
	}
	else
		m_headerSize = 0;

	////////////////////////////////////////////////////////////////

	printf("The dataset has been created.\n");
}

KBool FolderClassFeeder::m_getFileList(KString sFolderPath, KStrList &vsFilenames, KBool bRelativePath)
{
    char cLastCh = *sFolderPath.rbegin();

    if (cLastCh != '/' && cLastCh != '\\')
        sFolderPath += "/";

    return m_getFileList(sFolderPath, KString(""), vsFilenames, bRelativePath);
}


KBool FolderClassFeeder::m_getFileList(KString sBaseFolderPath, KString sSubFolderPath, KStrList &vsFilenames, KBool bRelativePath)
{
// Check the C++ compiler, STL version, and OS type.
////#if (defined(_MSC_VER) && (_MSVC_LANG >= 201703L || _HAS_CXX17)) || (defined(__GNUC__) && (__cplusplus >= 201703L))
//#if 1
	// ISO C++17 Standard (/std:c++17 or -std=c++17)
    KString sSearchName= sBaseFolderPath + sSubFolderPath;

    fs::path search_name(sSearchName);
    if (!fs::exists(search_name)) {
        printf("error: %s(%u): invalid path (%s)\n", __FUNCTION__, __LINE__, sSearchName.c_str());
        return false;
    }

    // Get all filenames in the target folder
    for (auto &it : fs::directory_iterator(search_name)) {
        KString sFilename = it.path().filename().string();
		
		if (fs::is_directory(it)) {
            // Is directory
            m_getFileList(sBaseFolderPath, sSubFolderPath + sFilename + "/", vsFilenames, bRelativePath);
        }
        else {
            // Is file
            if (!bRelativePath)
                vsFilenames.push_back(sBaseFolderPath + sSubFolderPath + sFilename);
            else
                vsFilenames.push_back(sSubFolderPath + sFilename);
        }
    }

    return true;

//#elif _WIN32 || _WIN64
//	// Microsoft Windows x86/64
//    HANDLE hFind;
//    WIN32_FIND_DATAA win32fd;
//
//    KString sSearchName= sBaseFolderPath + sSubFolderPath + "*";
//
//    hFind = FindFirstFileA(sSearchName.c_str(), &win32fd);
//
//    if (hFind == INVALID_HANDLE_VALUE) {
//        printf("error: %s(%u): invalid path (%s)\n", __FUNCTION__, __LINE__, sSearchName.c_str());
//        return false;
//    }
//
//    KString sDot(".");
//    KString sDoubleDot("..");
//
//    // Get all filenames in the target folder
//    do {
//        KString sFilename = KString(win32fd.cFileName);
//
//        if (win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
//            // Is directory
//            if (sFilename==sDot || sFilename==sDoubleDot)
//                continue;
//
//            m_getFileList(sBaseFolderPath, sSubFolderPath + sFilename + "/", vsFilenames, bRelativePath);
//        }
//        else {
//            // Is file
//            if (!bRelativePath)
//                vsFilenames.push_back(sBaseFolderPath + sSubFolderPath + sFilename);
//            else
//                vsFilenames.push_back(sSubFolderPath + sFilename);
//        }
//    } while (FindNextFileA(hFind, &win32fd));
//
//    FindClose(hFind);
//
//    return true;
//
//#elif __unix__ || __linux__
//	// UNIX or Linux system
//	// Not supported yet
//    printf("error: %s(%u): Not supported yet.\n", __FUNCTION__, __LINE__);
//    return false;
//
//#endif
}

KStrList FolderClassFeeder::m_filtering(const KStrList &vsFilenames, const KString &sFilterString)
{
	// Get a list of filters
	KStrList vsFilterList = m_splitString(sFilterString, ";");

	KStrList rs;

	for (KStrList::const_iterator it=vsFilenames.cbegin(); it!=vsFilenames.cend(); ++it) {
		if (!m_isValidate(*it, vsFilterList))
			continue;

		rs.push_back(*it);
	}

	return rs;
}

KBool FolderClassFeeder::m_isValidate(const KString &sFilename, const KStrList &vsFilterList)
{
	KString::size_type filenameLength = sFilename.length();

	KBool bSucc = false;

	for (auto &sFilter : vsFilterList) {

		KString::size_type filterLength = sFilter.length();

		if (filenameLength < filterLength)
			continue;

		KString sSubString = sFilename.substr(filenameLength - filterLength, KString::npos);

		if (sSubString == sFilter) {
			bSucc = true;
			break;
		}
	}

	return bSucc;
}

std::vector< KStrList > FolderClassFeeder::m_splitPath(const KStrList &vsFilenames)
{
	std::vector< KStrList > vsSplitFilenames;

	for (KStrList::const_iterator it=vsFilenames.cbegin(); it!=vsFilenames.cend(); ++it) {
		vsSplitFilenames.push_back( m_splitString(*it, "\\/") );
	}

	return vsSplitFilenames;
}

KStrList FolderClassFeeder::m_splitString(const KString &str, KString sDelimiters)
{
	KStrList ret;

    KString::size_type i = 0;
    KString::size_type j = 0;

	while (i != KString::npos) {
		// Ignore prefix delimiters
        i = str.find_first_not_of(sDelimiters, i);

		// Find the end position of the next segment
		j = str.find_first_of(sDelimiters, i);
        
		// Copy characters in the range of [i,j)
		if (i != KString::npos)
			ret.push_back( str.substr(i, j-i) );

		i = j;
	}

	return ret;
}

std::vector< std::map<KString,KInt> > FolderClassFeeder::m_generateClassList(const std::vector<KStrList> &vsSplitFilenames)
{
	// Check the maximum level(depth of folder tree)
	KStrList::size_type maxLevel = 0;
	for (std::vector<KStrList>::const_iterator it=vsSplitFilenames.cbegin(); it!=vsSplitFilenames.cend(); ++it) {
		// Get the class(folder) depth (excluding filename)
		KStrList::size_type cntClass = it->size() - 1;

		if (maxLevel < cntClass)
			maxLevel = cntClass;
	}

	// Create an empty list each level(folder name)
	std::vector< std::map<KString,KInt> > vmClassList;
	for (KStrList::size_type i=0; i<maxLevel; ++i)
		vmClassList.push_back( std::map<KString,KInt>() );

	// Generate the class list
	for (std::vector<KStrList>::const_iterator it=vsSplitFilenames.cbegin(); it!=vsSplitFilenames.cend(); ++it) {
		for (KStrList::size_type i=0; i<it->size() - 1; ++i)
			vmClassList[i][ (*it)[i] ]++;
	}

	return vmClassList;
}

bool FolderClassFeeder::m_loadCache(KString sCacheFilename) {
	FILE* fid = Utils::fopen(sCacheFilename, "rb", false);

	if (fid == NULL)
		return false;

	printf("Loading cache data...\n");

	FolderClassCache cache;

	if (fread(&cache.m_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_version, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_cntData, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_vecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_inputVecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
	if (fread(&cache.m_outputVecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_READ);
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

	// Get the vector sizes
	m_inputVecSize  = cache.m_inputVecSize;
	m_outputVecSize = cache.m_outputVecSize;
	m_propDict["vec_size"] = m_vecSize = cache.m_vecSize;

	// Set values of "input/output_columns"
	KaiList input_columns;
	input_columns.push_back( KaiList{(KInt)0, (KInt)m_inputVecSize} );
	KaiList output_columns;
	output_columns.push_back( KaiList{(KInt)m_inputVecSize, (KInt)m_outputVecSize} );

	m_propDict["input_columns"]  = input_columns;
	m_propDict["output_columns"] = output_columns;

	// Get a list of headers
	m_propDict["header_size"] = m_headerSize = cache.m_headerSize;
	if (m_headerSize > 0)
		m_propDict["header"] = cache.m_header;

	// Get payload data
	if (m_parData) {
		delete [] m_parData;
		m_parData = NULL;
	}

	m_parData = cache.m_parData;

	return true;
}

void FolderClassFeeder::m_saveCache(KString sCacheFilename) {
	printf("Saving cache data...\n");

	FILE* fid = Utils::fopen(sCacheFilename, "wb");

	if (fid == NULL)
		THROW(KERR_FAILURE_ON_FILE_SAVE);

	// Write headers
	if (fwrite(&ms_checkCode, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_version, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_cntData, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_vecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_inputVecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
	if (fwrite(&m_outputVecSize, sizeof(int), 1, fid) != 1) THROW(KERR_FAILURE_ON_FILE_SAVE);
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

KBool FolderClassFeeder::m_getDataCount(void* pAux, KInt* pnDataCount) {
	if (!pnDataCount)
		return false;

	*pnDataCount = m_cntData;
	return true;
}

KBool FolderClassFeeder::m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) {
	////////////////////////////////////////////////////////////////

	// User-defined code (To set shapes of feeding data)
	
	// feed_shape = is_seq ? { data_step, data_shape } : { data_shape }

	/*
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
	*/

	KBool    is_float_type = true;
	KBool    is_seq        = false;
	KInt     data_step     = 1;
	KaiShape data_shape	   = m_propDict[bInput ? "input_shape" : "output_shape"]; // KaiShape{ nTargetDataSize };

	////////////////////////////////////////////////////////////////

	// For interworking with callbacks (do not modify)
	pFieldInfo->m_bIsFloat   = is_float_type;
	pFieldInfo->m_shape      = data_shape;
	pFieldInfo->m_bIsSeq     = is_seq;
	pFieldInfo->m_nTimesteps = data_step;

	////////////////////////////////////////////////////////////////

	return true;
}

KBool FolderClassFeeder::m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	THROW(KERR_INTERNAL_LOGIC_ERROR);
	return false;
}

KBool FolderClassFeeder::m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
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
