/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../data_feeder.h"
#include "../utils/utils.h"

class Yolo3Feeder : public DataFeeder {
public:
	Yolo3Feeder(KString sub_model);
	virtual ~Yolo3Feeder();

	virtual void loadData(KString data_path, KString cache_path);

protected:
	virtual KBool m_getDataCount(void* pAux, KInt* pnDataCount);
	virtual KBool m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod);
	//virtual KBool m_getSecDataCount(void* pAux, Ken_data_section section, KInt nDatCount, KInt* pnDataCount);
	virtual KBool m_getSecBatchSize(void* pAux, Ken_data_section section, KInt* pnBatchSize);
	virtual KBool m_getExtraFields(void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields);
	virtual KBool m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo);
	virtual KBool m_informDataIndexes(void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount);
	virtual KBool m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer);
	virtual KBool m_feedFreeIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer);
	virtual KBool m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer);
	virtual KBool m_feedFreeFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer);

protected:
	int m_version;
	static int ms_checkCode;

protected:
	KString m_sub_model;

protected:
	void m_load_data(KString data_path);

	bool m_load_cache(KString cache_file);
	void m_save_cache(KString cache_path);

	void m_save_chche_for_sub_model(KString cache_path, KString model_name, int skip);

	void m_reset();

protected:
	struct _ImageAnnotaions {
		int m_category;
		KFloat m_bbox[4];
	};

	struct _ImageInfo {
		int m_image_id;
		int m_width;
		int m_height;
		KString m_filepath;
		std::vector<_ImageAnnotaions> m_annotations;
	};

	std::map<int, KString> m_categories;
	std::vector<_ImageInfo> m_image_info_list;

protected:
	// 미니배치 준비용 버퍼
	// 미니배치 데이터 생성시 50% 확률로 이미지를 믹스하자. 믹스비율은 beta(1.5, 1.5)
	const int mc_image_size = 416;

	const int mc_scale = 3;
	const int mc_anc_per_scale = 3;

	const int mc_class_num = 80;
	const int mc_true_vec_size = 92; // 4 for bounding-box, 1 for confidence, 80 for classes, 7 extra information

	const int mc_grid_cnt[3] = { 13, 26, 52 };
	const int mc_grid_size[3] = { 32, 16, 8 };

	const int mc_grid_cell_count = (13 * 13) + (26 * 26) + (52 * 52);

	const int mc_anchor[3][3][2] = {
		{ {116, 90}, { 156,198 }, { 373,326 }},
		{ {30,61}, {62,45}, {59,119} },
		{ {10,13}, {16,30}, {33,23} } };

	KString m_data_path;

	float* m_pImages;
	float* m_pTrueMap;
	float* m_pTrueBox;

	int m_nBoxCountPerImage;

	int m_select_mixup_image(int nIndex1, int nRangeStart, int nRangeCount, float& mix_ratio);
	void m_fill_image_pixels(float* pBuffer, int nImageIdx);
	void m_mixup_image_pixels(float* pBuffer, int nImageIdx, float mix_ratio);
	void m_fill_box_info(int nth_data, int nImageIdx, float mix_ratio, KaiList& boxes);
	float* m_get_map_pos(int x, int y, int nth_data, int nscale, int nanchor);
};
