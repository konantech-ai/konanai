/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../data_feeder.h"
#include "../utils/utils.h"

typedef class CsvFeeder : public DataFeeder {
public:
	CsvFeeder();
	virtual ~CsvFeeder();

	// Load a CSV file
	virtual void  loadData(KString sCsvFilename, KString sCachePath, KaiDict kwArgs, KHSession hSession, KHDataset hDataset);
	virtual KaiList getOuputFieldNames();
protected:
	// Essential methods
	virtual KBool m_getDataCount(void* pAux, KInt* pnDataCount);
	virtual KBool m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);
	virtual KBool m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	virtual KBool m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);

	// Getter & setter
	KaiValue      m_getProperty(KString sKey, KaiValue def);
	void          m_setProperty(KString sKey, KaiValue val);

	// Dataset generator
	void          m_createDataset(KString sCsvFilename);

	// CSV file reader
	KaiList       m_readCsvFile(KString sCsvFilename, KaiList* pHead);	// Extracted from Util::load_csv()

	// Cache manager
	bool          m_loadCache(KString sCacheFilename);
	void          m_saveCache(KString sCacheFilename);

	// Member variables
	static int    ms_checkCode;
	int           m_version;
	int           m_cntData;
	int           m_vecSize;
	int           m_headerSize;
	KaiDict       m_propDict;
	KFloat*       m_parData;

} CsvFeeder;
