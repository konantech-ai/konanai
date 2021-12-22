/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../include/kai_api.h"
#include "../math/karray.h"

class KaiModelInstance;
class KaiExecContext;
class KaiMath;

class KaiCallbackAgent {
public:
	KaiCallbackAgent(KaiModelInstance* pModelInst);
	virtual ~KaiCallbackAgent();

	KBool get_data_count(KInt* pnDataCount, KBool bThrow);
	KBool get_data_suffle_method(Ken_data_suffle_method* penumSuffleMethod, KBool bThrow);
	KBool get_section_data_count(KInt nDataCount, KInt* pnTrCount, KInt* pnTeCount, KInt* pnVaCount, KBool bThrow);
	KBool get_section_batch_size(KInt* pnTrBatch, KInt* pnTeBatch, KInt* pnVaBatch, KBool bThrow);
	KBool get_extra_fields(KStrList& inFieldNames, KStrList& outFieldNames, KBool bThrow);
	KBool get_field_spec(KBool bInput, KString sFieldName, KaiDict& field_info, KBool bThrow);

	KBool inform_data_indexes(KIntList data_indexes, KInt nRangeStart, KInt nRangeCount);
	KBool fetchData(KIntList data_indexes, KaiDict& xs, KaiDict& ys, KaiExecContext* pContext, KBool bThrow);

	KBool train_start(KString sName, KString sTimestamp, KInt epoch_count, KInt data_count, KInt batch_size, KInt batch_count);
	KBool train_end(KString sName, KString sTimestamp);
	KBool train_epoch_start(KInt epoch_count, KInt epoch_index, KaiDict tr_data);
	KBool train_epoch_end(KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy);
	KBool train_batch_start(KInt batch_count, KInt batch_index);
	KBool train_batch_end(KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy);
	KBool train_validate_start(KInt data_count, KInt batch_size);
	KBool train_validate_end(KaiDict accuracy);

	KBool test_start(KString sName, KString sTimestamp, KInt data_count, KaiDict te_data);
	KBool test_end(KString sName, KString sTimestamp, KaiDict accuracy);

	KBool visualize_start(KString sName, Ken_visualize_mode mode, KInt nDatCount, KaiDict data_index);
	KBool visualize_end(KString sName, KaiDict xs, KaiDict ys, KaiDict outs, KaiDict vis_dict);

protected:
	std::map<int, std::map<int, KaiList>> m_cbFamilies;

	KBool m_fetch_functions(int family, int event, KaiList& cbFuncs);
	KBool m_feed_data(KaiDict& dats, KString sFieldName, KaiDict field_info, KaiList cbFloatFuncs, KaiList cbIntFuncs, KInt mb_size, KBool bInput, KIntList indexs, KaiMath* pMath);
	KBool m_feed_free_shape_data(KaiDict& dats, KString sFieldName, KaiDict field_info, KaiList cbFloatFuncs, KaiList cbIntFuncs, KInt mb_size, KBool bInput, KIntList indexs, KaiMath* pMath);
};

class KaiCallbackTransaction {
public:
	KaiCallbackTransaction();
	virtual ~KaiCallbackTransaction();

	KaiDict create_token(KaiValue value);
	KaiDict conv_to_token_dict(KaiDict dict);

protected:
	struct _ArrToken {
		KaiCallbackTransaction* m_pTransaction;
		int m_index;
		int m_nCheckCode;
	};

	friend class KaiSession;

	std::vector<_ArrToken*> m_tokens;
	KaiList m_values;

	int m_nCheckCode;

	static int ms_tokenCode;
	static int ms_checkCode;

public:
	void download_float_data(_ArrToken* pToken, KInt nSize, KFloat* pBuffer);
	void download_int_data(_ArrToken* pToken, KInt nSize, KInt* pBuffer);

};
