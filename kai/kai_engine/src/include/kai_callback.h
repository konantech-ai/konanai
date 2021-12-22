/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "kai_types.h"
#include "kai_value.hpp"

enum class Ken_cb_family { session, datafeed, train, test, visualize, predict };

enum class Ken_session_cb_event{ print };
enum class Ken_datafeed_cb_event { get_data_cnt, get_suffle_method, get_sec_data_cnt, get_sec_batch_size, get_extra_fields, get_field_spec,
								   inform_data_indexes, feed_int_data, feed_int_free_data, feed_float_data, feed_float_free_data };
enum class Ken_train_cb_event { train_start, train_end, train_epoch_start, train_epoch_end, train_batch_start, train_batch_end, train_validate_start, train_validate_end };
enum class Ken_test_cb_event { test_start, test_end };
enum class Ken_visualize_cb_event { visualize_start, visualize_end };
enum class Ken_loss_cb_event { loss_start, get_loss_info, eval_loss, loss_end };

enum class Ken_data_section { train, validate, test };
enum class Ken_data_suffle_method { sequential, random };

typedef void KCbPrint(void* pInstance, KString sOutput, KBool bNewLine);

struct KCbFieldInfo {
	//KBool m_bFilled;	// pMatchInfo 구조체에 정보 기입이 필요한 경우에 한해 true로 변경
	KBool m_bIsFloat;
	KaiShape m_shape;
	KBool m_bIsSeq;
	KInt m_nTimesteps;
	KInt m_bFreeShape;
};

typedef KBool KCbGetDataCount(void* pInst, void* pAux, KInt* pnDataCount);																						// get_data_cnt
typedef KBool KCbGetDataSuffleMethod(void* pInst, void* pAux, Ken_data_suffle_method* pSuffleMethod);															// get_suffle_method
typedef KBool KCbGetSectionDataCount(void* pInst, void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount);										// get_sec_data_cnt
typedef KBool KCbGetSectionBatchSize(void* pInst, void* pAux, Ken_data_section section, KInt* pnBatchSize);														// get_sec_batch_size
typedef KBool KCbGetExtraFields(void* pInst, void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields);		// get_extra_field
// pMatchInfo 불필요해 보임. 필요성 나타나지 않으면 정리
typedef KBool KCbGetFieldSpec(void* pInst, void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);					// get_field_spec
typedef KBool KCbInformDataIndexes(void* pInst, void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount);											// inform_data_indexes
typedef KBool KCbFeedIntData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);					// feed_int_data
typedef KBool KCbFeedFreeShapeIntData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer);		// feed_int_free_data
typedef KBool KCbFeedFloatData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);				// feed_float_data
typedef KBool KCbFeedFreeShapeFloatData(void* pInst, void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer);	// feed_float_free_data

typedef KBool KCbTrainStart(void* pInst, void* pAux, KString sName, KString sTimestamp, KInt epoch_count, KInt data_count);										// train_start
typedef KBool KCbTrainEnd(void* pInst, void* pAux, KString sName, KString sTimestamp);																			// train_end
typedef KBool KCbEpochStart(void* pInst, void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt batch_size, KaiDict data_index_token);		// train_epoch_start
typedef KBool KCbEpochEnd(void* pInst, void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy);		// train_epoch_end
typedef KBool KCbBatchStart(void* pInst, void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index);													// train_batch_start
typedef KBool KCbBatchEnd(void* pInst, void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy);	// train_batch_end
typedef KBool KCbValidateStart(void* pInst, void* pAux, KString sTimestamp, KInt data_count, KInt batch_size);													// train_validate_start
typedef KBool KCbValidateEnd(void* pInst, void* pAux, KString sTimestamp, KaiDict accuracy);																	// train_validate_end

typedef KBool KCbInformLossStart(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbGetSrcFieldsForLoss(void* pInst, void* pAux, KaiList& fieldNames);
typedef KBool KCbEvaluateLoss(void* pInst, void* pAux, KaiDict firldPack, KaiDict& loss, KBool& finished);
typedef KBool KCbInformLossEnd(void* pInst, void* pAux, KaiValue info);

typedef KBool KCbLayerForwardStart(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbLayerForwardEnd(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbLayerBackpropStart(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbLayerBackpropEnd(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbNetworkStart(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbNetworkEnd(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbEvalAccStart(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbEvalAccEnd(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbParamUpdateStart(void* pInst, void* pAux, KaiValue info);
typedef KBool KCbParamUpdateEnd(void* pInst, void* pAux, KaiValue info);

typedef KBool KCbTestStart(void* pInst, void* pAux, KString sName, KString sTimestamp, KInt dat_count, KaiDict data_index_token);										// test_start
typedef KBool KCbTestEnd(void* pInst, void* pAux, KString sName, KString sTimestamp, KaiDict accuracy);																	// test_end

typedef KBool KCbVisualizeStart(void* pInst, void* pAux, KString sName, KString sTimestamp, Ken_visualize_mode mode, KInt dat_count, KaiDict data_index_token);			// visualize_start
typedef KBool KCbVisualizeEnd(void* pInst, void* pAux, KString sName, KString sTimestamp, KaiDict xs_tokens, KaiDict ys_tokens, KaiDict os_tokens, KaiDict vis_tokens);	// visualize_end

typedef KBool KCbPredictStart(void* pInst, void* pAux, KInt nCount);
typedef KBool KCbPredictData(void* pInst, void* pAux, KInt nth);
typedef KBool KCbPredictEnd(void* pInst, void* pAux);

const KInt KCb_mask_train_start				= 0x00000001LL;
const KInt KCb_mask_train_end				= 0x00000002LL;
const KInt KCb_mask_train_epoch_start		= 0x00000004LL;
const KInt KCb_mask_train_epoch_end			= 0x00000008LL;
const KInt KCb_mask_train_batch_start		= 0x00000010LL;
const KInt KCb_mask_train_batch_end			= 0x00000020LL;
const KInt KCb_mask_train_validate_start	= 0x00000040LL;
const KInt KCb_mask_train_validate_end		= 0x00000080LL;

const KInt KCb_mask_test_start				= 0x01000000LL;
const KInt KCb_mask_test_end				= 0x02000000LL;

const KInt KCb_mask_visualize_start			= 0x10000000LL;
const KInt KCb_mask_visualize_end			= 0x20000000LL;

const KInt KCb_mask_all						= 0xFFFFFFFFLL;
