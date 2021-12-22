/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
/*
 * Caution!
 * 
 * This program requires a C++17 compiler or MS Windows OS.
 * Also, the program only supports multi-byte character set.
 */

#pragma once

#include "../data_feeder.h"
#include "../utils/utils.h"

typedef class FolderClassFeeder : public DataFeeder {
public:
	FolderClassFeeder();
	virtual ~FolderClassFeeder();

	// Load a source folder recursively
	virtual void  loadData(KString sSrcFolderPath, KString sCachePath, KaiDict kwArgs, KHSession hSession, KHDataset hDataset);
	virtual KaiList getTargetNames();

	// Getter & setter
	KaiValue      getProperty(KString sKey, KaiValue def);
	void          setProperty(KString sKey, KaiValue val);

protected:
	// Essential methods
	virtual KBool m_getDataCount(void* pAux, KInt* pnDataCount);
	virtual KBool m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);
	virtual KBool m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	virtual KBool m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);

	// Dataset generator
	void          m_createDataset(KString sFolderPath);

	// File list collector
	KBool         m_getFileList(KString sFolderPath, KStrList &vsFilenames, KBool bRelativePath = false);
	KBool         m_getFileList(KString sBaseFolderPath, KString sSubFolderPath, KStrList &vsFilenames, KBool bRelativePath);

	// Filename filtering
	KStrList      m_filtering(const KStrList &vsFilenames, const KString &sFilterString);
	KBool         m_isValidate(const KString &sFilename, const KStrList &vsFilterList);

	// Splitter (Performance on Windows : O(N*M))
	std::vector<KStrList> m_splitPath(const KStrList &vsFilenames);
	KStrList              m_splitString(const KString &str, KString sDelimiters);

	// Class list generator
	std::vector< std::map<KString,KInt> > m_generateClassList(const std::vector<KStrList> &vsSplitFilenames);

	// Cache manager
	bool          m_loadCache(KString sCacheFilename);
	void          m_saveCache(KString sCacheFilename);

	// Member variables
	static int    ms_checkCode;
	int           m_version;
	int           m_cntData;
	int           m_vecSize;
	int           m_inputVecSize;
	int           m_outputVecSize;
	int           m_headerSize;
	KaiDict       m_propDict;
	KFloat*       m_parData;

} FolderClassFeeder;
