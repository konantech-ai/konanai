/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "component.h"
#include "../math/karray.h"

class KaiExecContext;
class KaiModelInstance;
class KaiCallbackAgent;

class KaiDataset : public KaiComponent {
public:
	KaiDataset(KaiSession* pSession, KaiDict kwArgs);
	KaiDataset(KaiSession* pSession, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);
	virtual ~KaiDataset();

	Ken_object_type get_type() { return Ken_object_type::dataset; }

	static KaiDataset* HandleToPointer(KHObject hObject);
	static KaiDataset* HandleToPointer(KHObject hObject, KaiSession* pSession);

	static KaiDataset* CreateInstance(KaiSession* pSession, KString sBuiltin, KaiDict kwArgs);

	static void GetBuiltinNames(KStrList* pslNames);

	virtual void read_file(KString sDataFilePath);
	virtual void setDataFeedingInfo(KaiModelInstance* pModelInst, KaiCallbackAgent* pCbAgent);

	KString desc();

	static void fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext);
	static KaiDict fetch_predict_data(KaiExecContext* pContext, KInt* pnDatCount);

protected:
	static int ms_checkCode;
	static KStrList ms_builtin;
};

class KaiCsvReaderDataset : public KaiDataset {
public:
	KaiCsvReaderDataset(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiCsvReaderDataset();

	virtual void read_file(KString sDataFilePath);

	static void fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext);
	static KaiDict fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount);

protected:
};

class KaiFolderClassDataset : public KaiDataset {
public:
	KaiFolderClassDataset(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiFolderClassDataset();

	virtual void read_file(KString sDataFilePath);

	static void fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext);
	static KaiDict fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount);

protected:
};

// Caution! More than 3 layers are not supported yet. (2021-08-19)
class KaiFolderClassRecursiveDataset : public KaiDataset {
public:
	KaiFolderClassRecursiveDataset(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiFolderClassRecursiveDataset();

	// Load target image files and create answer sets.
	virtual void read_file(KString sDataFilePath);

	// Integrate the target image data and answer sets into one data block.
	// Also, upload the integrated data to GPU memory.
	static void fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext);
	static KaiDict fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount);

protected:
	// TODO
};
class KaiFeedingDataset : public KaiDataset {
public:
	KaiFeedingDataset(KaiSession* pSession, KaiDict kwArgs);
	virtual ~KaiFeedingDataset();

	virtual void read_file(KString sDataFilePath);

	virtual void setDataFeedingInfo(KaiModelInstance* pModelInst, KaiCallbackAgent* pCbAgent);

	//static void fetchData(KaiArray<KInt> batch_index, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext);
	static KaiDict fetch_predict_data(KaiExecContext* pContext, KInt* pnDataCount);

protected:
	//struct _CbInfo { void* m_pCbInst; void* m_pCbFunc; void* m_pCbAux; };

	//std::map< Ken_dataset_cbfunc, _CbInfo> m_cbFuncMap;

	//bool m_bInfoFetched;

	/*
	void m_fetchFeedingInfo();

	KInt m_nDataCount;

	KInt m_nTrCount;
	KInt m_nVaCount;
	KInt m_nTeCount;

	KInt m_nTrBatch;
	KInt m_nVaBatch;
	KInt m_nTeBatch;

	KBool m_bInFloat;
	KBool m_bOutFloat;

	KBool m_bInSeq;
	KBool m_bOutSeq;

	KInt m_nInTimesteps;
	KInt m_nOutTimesteps;

	KaiShape m_ishape;
	KaiShape m_oshape;

	Ken_data_suffle_method m_enumSuffleMethod;
	*/
};

