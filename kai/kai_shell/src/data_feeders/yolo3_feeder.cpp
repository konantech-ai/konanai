/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "yolo3_feeder.h"
#include "../utils/json_parser.h"
#include <cmath>
int Yolo3Feeder::ms_checkCode = 61997224;

Yolo3Feeder::Yolo3Feeder(KString sub_model) : DataFeeder() {
	m_version = 3;
	m_sub_model = sub_model;

	m_pImages = 0;
	m_pTrueMap = 0;
	m_pTrueBox = 0;
}

Yolo3Feeder::~Yolo3Feeder() {
	delete[] m_pImages;
	delete[] m_pTrueMap;
	delete[] m_pTrueBox;
}

void Yolo3Feeder::loadData(KString data_path, KString cache_path) {
	Utils::mkdir(cache_path);

	m_data_path = data_path;

	KString cache_file = cache_path + "/yolo3." + m_sub_model + ".cache";;

	if (!m_load_cache(cache_file)) {
		m_load_data(data_path);
		m_save_cache(cache_path);
		m_reset();
		m_load_cache(cache_file);
	}
}

void Yolo3Feeder::m_load_data(KString data_path) {
	printf("loading yolo3 information file...\n");

	KString inst_path = data_path + "/annotations_trainval2014/annotations/instances_train2014.json";

	JsonParser parser;
	KaiDict inst_info = parser.parse_file(inst_path);  // LR-파싱으로 빠르게 처리 가능할 듯

	std::map<int, _ImageInfo> image_info_map;

	for (auto& it : (KaiList)inst_info["categories"]) {
		KaiDict cat_term = it;
		m_categories[cat_term["id"]] =(KString) cat_term["name"];
	}

	int lost_file = 0;

	for (auto& it : (KaiList)inst_info["images"]) {
		KaiDict image_term = it;

		std::string filepath = "E:/alzza/coco-datasets/train2014/" + (KString)image_term["file_name"];
		if (Utils::file_exist(filepath)) {
			int image_id = image_term["id"];
			_ImageInfo image_info;
			image_info.m_image_id = image_id;
			image_info.m_width = image_term["width"];
			image_info.m_height = image_term["height"];
			image_info.m_filepath = (KString)image_term["file_name"];
			image_info_map[image_id] = image_info;
		}
		else {
			printf("  LostFile-%d: %s\n", lost_file++, filepath.c_str());
		}
	}

	int info_for_missing_image = 0;

	for (auto& it : (KaiList)inst_info["annotations"]) {
		// 이미지 파일 없으면 처리 생략할 것, 정보 손상 건수 보고
		KaiDict ann_term = it;
		int image_id = ann_term["image_id"];

		if (image_info_map.find(image_id) == image_info_map.end()) {
			info_for_missing_image++;
			continue;
		}
		_ImageAnnotaions ann_info;

		KaiList bbox = ann_term["bbox"];
		ann_info.m_category = ann_term["category_id"];

		ann_info.m_bbox[0] = bbox[0];
		ann_info.m_bbox[1] = bbox[1];
		ann_info.m_bbox[2] = bbox[2];
		ann_info.m_bbox[3] = bbox[3];

		_ImageInfo& image_info = image_info_map[image_id];
		std::vector<_ImageAnnotaions>& annotaions = image_info.m_annotations;
		annotaions.push_back(ann_info);
	}

	if (info_for_missing_image > 0) {
		printf("info_for_missing_image: %d\n", info_for_missing_image);
	}

	for (auto& it : image_info_map) {
		m_image_info_list.push_back(it.second);
	}
}

void Yolo3Feeder::m_save_cache(KString cache_path) {
	printf("saving yolo3 cache data...\n");

	m_save_chche_for_sub_model(cache_path, "large", 1);
	m_save_chche_for_sub_model(cache_path, "medium", 10);
	m_save_chche_for_sub_model(cache_path, "small", 100);
}

void Yolo3Feeder::m_save_chche_for_sub_model(KString cache_path, KString model_name, int skip) {
	KString cache_file = cache_path + "/yolo3." + model_name + ".cache";;
	
	FILE* fid = Utils::fopen(cache_file.c_str(), "wb");

	fwrite(&m_version, sizeof(int), 1, fid);
	fwrite(&ms_checkCode, sizeof(int), 1, fid);

	int cat_count = (int)m_categories.size();

	fwrite(&cat_count, sizeof(int), 1, fid);

	for (auto &it: m_categories) {
		int cat_id = it.first;
		KString cat_name = it.second;

		int word_len = (int)strlen(cat_name.c_str());

		fwrite(&cat_id, sizeof(int), 1, fid);
		fwrite(&word_len, sizeof(int), 1, fid);
		fwrite(cat_name.c_str(), sizeof(char), word_len, fid);
	}

	fwrite(&ms_checkCode, sizeof(int), 1, fid);

	int image_count = ((int)m_image_info_list.size() + skip - 1) / skip;
	fwrite(&image_count, sizeof(int), 1, fid);

	int nth = 0;
	for (auto &it : m_image_info_list) {
		if (nth++ % skip != 0) continue;

		_ImageInfo& image_info = it;

		fwrite(&image_info.m_image_id, sizeof(int), 1, fid);
		fwrite(&image_info.m_width, sizeof(int), 1, fid);
		fwrite(&image_info.m_height, sizeof(int), 1, fid);

		int path_len = (int)strlen(image_info.m_filepath.c_str());
		fwrite(&path_len, sizeof(int), 1, fid);
		fwrite(image_info.m_filepath.c_str(), sizeof(char), path_len, fid);

		std::vector<_ImageAnnotaions>& annotations = image_info.m_annotations;

		int info_cnt = (int)annotations.size();
		fwrite(&info_cnt, sizeof(int), 1, fid);

		for (auto& it : annotations) {
			_ImageAnnotaions& ann_info = it;

			fwrite(&ann_info.m_category, sizeof(int), 1, fid);
			fwrite(ann_info.m_bbox, sizeof(float), 4, fid);
		}
	}

	fwrite(&ms_checkCode, sizeof(int), 1, fid);

	fclose(fid);
}

bool Yolo3Feeder::m_load_cache(KString cache_file) {
	FILE* fid = Utils::fopen(cache_file.c_str(), "rb", false);

	if (fid == NULL) return false;

	printf("loading yolo3 cache data...\n");

	char buffer[1024];

	int version;
	int checkCode;

	fread(&version, sizeof(int), 1, fid);
	fread(&checkCode, sizeof(int), 1, fid);

	if (version != m_version || checkCode != ms_checkCode) {
		fclose(fid);
		return false;
	}

	int cat_count;

	fread(&cat_count, sizeof(int), 1, fid);

	for (int n = 0; n < cat_count; n++) {
		int cat_id;
		fread(&cat_id, sizeof(int), 1, fid);

		int word_len;
		KString cat_name;
		fread(&word_len, sizeof(int), 1, fid);
		assert(word_len < 1024);
		fread(buffer, sizeof(char), word_len, fid);
		std::string word(buffer, word_len);

		m_categories[cat_id] = word;
	}

	fread(&checkCode, sizeof(int), 1, fid);
	if (checkCode != ms_checkCode) {
		fclose(fid);
		m_reset();
		return false;
	}

	int image_count;

	fread(&image_count, sizeof(int), 1, fid);

	for (int n = 0; n < image_count; n++) {
		_ImageInfo image_info;

		fread(&image_info.m_image_id, sizeof(int), 1, fid);
		fread(&image_info.m_width, sizeof(int), 1, fid);
		fread(&image_info.m_height, sizeof(int), 1, fid);

		int word_len;
		KString cat_name;
		fread(&word_len, sizeof(int), 1, fid);
		assert(word_len < 1024);
		fread(buffer, sizeof(char), word_len, fid);
		image_info.m_filepath = std::string(buffer, word_len);

		int info_cnt;
		fread(&info_cnt, sizeof(int), 1, fid);

		for (int m = 0; m < info_cnt; m++) {
			_ImageAnnotaions ann_info;

			fread(&ann_info.m_category, sizeof(int), 1, fid);
			fread(ann_info.m_bbox, sizeof(float), 4, fid);

			image_info.m_annotations.push_back(ann_info);
		}

		m_image_info_list.push_back(image_info);
	}

	fread(&checkCode, sizeof(int), 1, fid);
	if (checkCode != ms_checkCode) {
		fclose(fid);
		m_reset();
		return false;
	}

	fclose(fid);

	return true;
}

void Yolo3Feeder::m_reset() {
	m_categories.clear();
	m_image_info_list.clear();
}

KBool Yolo3Feeder::m_getDataCount(void* pAux, KInt* pnDataCount) {
	*pnDataCount = m_image_info_list.size();
	return true;
}

KBool Yolo3Feeder::m_getDataSuffleMethod(void* pAux, Ken_data_suffle_method* pSuffleMethod) {
	*pSuffleMethod = Ken_data_suffle_method::random;
	return true;
}

KBool Yolo3Feeder::m_getSecBatchSize(void* pAux, Ken_data_section section, KInt* pnBatchSize) {
	*pnBatchSize = 8;
	return true;
}

KBool Yolo3Feeder::m_getExtraFields(void* pAux, KBool* pbUseDefInput, KStrList* psInputFields, KBool* pbUseDefOutput, KStrList* psOutputFields) {
	*pbUseDefInput = true;
	*pbUseDefOutput = false; // #default output은 신경망 출력과 형상을 비교하기 때문에 형상이 다른 경우 추가 필드로 따로 지정해야 한다.

	psOutputFields->push_back("true_map"); // 신경망 출력은 [...,85] 크기 벡터이고 true_map은 [...,88]로 마지막 차원 형상이 다른다.
	psOutputFields->push_back("true_box"); // 신경망 출력은 [...,85] 크기 벡터이고 true_map은 [...,88]로 마지막 차원 형상이 다른다.

	return true;
}

KBool Yolo3Feeder::m_getFieldSpec(void* pAux, KBool bInput, KString sFieldName, KCbFieldInfo* pFieldInfo, KCbFieldInfo* pMatchInfo) {
	if (bInput) {	// [416,416] 해상도로 크기 조절된 입력 이미지, mix로 인해 데이터 당 두 개의 이미지에서 합성된 내용이 입력될 수도 있다.
		pFieldInfo->m_bIsFloat = true;
		pFieldInfo->m_shape = KaiShape{ mc_image_size, mc_image_size, 3 };
		pFieldInfo->m_bIsSeq = false;
		pFieldInfo->m_nTimesteps = 0;
	}
	else if (sFieldName == "true_map") {
		pFieldInfo->m_bIsFloat = true;
		pFieldInfo->m_shape = KaiShape{ mc_grid_cell_count, mc_anc_per_scale, mc_true_vec_size };
		pFieldInfo->m_bIsSeq = false;
		pFieldInfo->m_nTimesteps = 0;
	}
	else if (sFieldName == "true_box") {
		pFieldInfo->m_bIsFloat = true;
		//pFieldInfo->m_shape = KaiShape{ 4 };
		pFieldInfo->m_bIsSeq = false;
		pFieldInfo->m_nTimesteps = 0;
		pFieldInfo->m_bFreeShape = true;
	}

	return true;
}

int Yolo3Feeder::m_select_mixup_image(int nIndex1, int nRangeStart, int nRangeCount, float& mix_ratio) {
	if (rand() % 2) return -1;
	mix_ratio = (rand() % 3 + rand() % 3 + rand() % 3 + rand() % 3 + 1) / 10.0f; // 추후 beta(1.5, 1.5) 분포로 대체
	return nRangeStart + rand() % nRangeCount;  // 같은 그림이랑 겹치는 경우도 허용해본다.
}

void Yolo3Feeder::m_fill_image_pixels(float* pBuffer, int nImageIdx) {
	_ImageInfo image_info = m_image_info_list[nImageIdx];

	std::string filepath = "E:/alzza/coco-datasets/train2014/" + image_info.m_filepath;
	Utils::load_jpeg_image_pixels(pBuffer, filepath, KaiShape{ mc_image_size, mc_image_size }, true);
}

void Yolo3Feeder::m_mixup_image_pixels(float* pBuffer, int nImageIdx, float mix_ratio) {
	float* pImage = new float[mc_image_size * mc_image_size * 3];

	m_fill_image_pixels(pImage, nImageIdx);

	for (int n = 0; n < mc_image_size * mc_image_size * 3; n++) {
		pBuffer[n] = pBuffer[n] * mix_ratio + pImage[n] * (1.0f - mix_ratio);
	}

	delete[] pImage;
}

void Yolo3Feeder::m_fill_box_info(int nth_data, int nImageIdx, float mix_ratio, KaiList& boxes) {
	_ImageInfo image_info = m_image_info_list[nImageIdx];

	int image_width = image_info.m_width;
	int image_height = image_info.m_height;

	int image_size = MAX(image_width, image_height);

	int h_base = 0, w_base = 0;

	if (image_height > image_width) {
		float ratio = (float)image_width / image_height;
		w_base = (int)((mc_image_size - mc_image_size * ratio) / 2);
	}
	else if (image_height < image_width) {
		float ratio = (float)image_height / image_width;
		h_base = (int)((mc_image_size - mc_image_size * ratio) / 2);
	}

	for (auto& it : image_info.m_annotations) {
		_ImageAnnotaions annotation = it;

		float best_iou = 0;
		int best_ns = 0;
		int best_na = 0;

		float* bbox = annotation.m_bbox;

		float box_width = bbox[2] / image_size * mc_image_size;
		float box_height = bbox[3] / image_size * mc_image_size;

		for (int ns = 0; ns < 3; ns++) {
			for (int na = 0; na < 3; na++) {
				float anchor_width = (float)mc_anchor[ns][na][0];
				float anchor_height = (float)mc_anchor[ns][na][1];

				float max_width = MAX(box_width, anchor_width);
				float min_width = MIN(box_width, anchor_width);

				float max_height = MAX(box_height, anchor_height);
				float min_height = MIN(box_height, anchor_height);

				float intersect_area = min_height * min_width;
				float union_area = max_height * max_width;

				float iou = intersect_area / union_area;

				if (iou > best_iou) {
					best_iou = iou;
					best_ns = ns;
					best_na = na;
				}
			}
		}

		float center_x = (bbox[0] + bbox[2] / 2) / image_size * mc_image_size + w_base;
		float center_y = (bbox[1] + bbox[3] / 2) / image_size * mc_image_size + h_base;

		int grid_x = (int)(center_x / mc_grid_size[best_ns]);
		int grid_y = (int)(center_y / mc_grid_size[best_ns]);

		assert(grid_x >= 0 && grid_x < mc_grid_cnt[best_ns]);
		assert(grid_y >= 0 && grid_y < mc_grid_cnt[best_ns]);

		float* pMap = m_get_map_pos(grid_x, grid_y, nth_data, best_ns, best_na);
		
		float box_scale_loss = 2.0f - box_width * box_height / (mc_image_size * mc_image_size);

		pMap[0] = center_x / mc_grid_size[best_ns] - grid_x;
		pMap[1] = center_y / mc_grid_size[best_ns] - grid_y;
		pMap[2] = ::logf(box_width / mc_anchor[best_ns][best_na][0]);
		pMap[3] = ::logf(box_height / mc_anchor[best_ns][best_na][1]);
		pMap[4] = 1.0f;
		pMap[5 + annotation.m_category] = 1.0f;
		pMap[85] = mix_ratio;
		pMap[86] = box_scale_loss;
		pMap[87] = (float)grid_x;
		pMap[88] = (float)grid_y;
		pMap[89] = (float)mc_grid_size[best_ns];
		pMap[90] = (float)mc_anchor[best_ns][best_na][0];
		pMap[91] = (float)mc_anchor[best_ns][best_na][1];

		KaiList box;
		
		KFloat left = center_x - box_width / 2;		// left
		KFloat right = center_x + box_width / 2;	// right
		KFloat top = center_y - box_height / 2;		// top
		KFloat bottom = center_y + box_height / 2;	// bottom

		box.push_back(left);
		box.push_back(right);
		box.push_back(top);
		box.push_back(bottom);

		boxes.push_back(box);
	}
}

float* Yolo3Feeder::m_get_map_pos(int x, int y, int nth_data, int nscale, int nanchor) {
	int idx = nth_data;

	idx = idx * mc_grid_cnt[nscale] + y;
	idx = idx * mc_grid_cnt[nscale] + x;

	if (nscale >= 1) idx += mc_grid_cnt[0] * mc_grid_cnt[0];
	if (nscale >= 2) idx += mc_grid_cnt[1] * mc_grid_cnt[1];

	idx = idx * mc_anc_per_scale + nanchor;
	idx = idx * mc_true_vec_size;

	return m_pTrueMap + idx;
}

KBool Yolo3Feeder::m_informDataIndexes(void* pAux, KIntList nDatIndexs, KInt nRangeStart, KInt nRangeCount) {
	delete[] m_pImages;
	delete[] m_pTrueMap;
	delete[] m_pTrueBox;

	m_pImages = 0;
	m_pTrueMap = 0;
	m_pTrueBox = 0;

	int mb_size = (int)nDatIndexs.size();
	int dat_size = mc_image_size * mc_image_size * 3;

	m_pImages = new float[mb_size * dat_size];
	memset(m_pImages, 0, sizeof(float) * mb_size * dat_size);

	int map_size = mb_size * mc_grid_cell_count * mc_anc_per_scale * mc_true_vec_size;
	m_pTrueMap = new float[map_size];
	memset(m_pTrueMap, 0, sizeof(float) * map_size);

	KaiList imageBoxes;
	
	m_nBoxCountPerImage = 0;

	for (int n = 0; n < mb_size; n++) {
		KFloat mix_ratio = 1.0f;

		int nDataIndex1 = (int)nDatIndexs[n];
		int nDataIndex2 = m_select_mixup_image(nDataIndex1, (int)nRangeStart, (int)nRangeCount, mix_ratio);

		KaiList boxes;
		m_fill_image_pixels(m_pImages + n * dat_size, nDataIndex1);
		m_fill_box_info(n, nDataIndex1, mix_ratio, boxes);

		if (nDataIndex2 >= 0) {
			m_mixup_image_pixels(m_pImages + n * dat_size, nDataIndex2, mix_ratio);
			m_fill_box_info(n, nDataIndex2, 1-mix_ratio, boxes);
		}

		if (boxes.size() > m_nBoxCountPerImage) m_nBoxCountPerImage = (int)boxes.size();

		imageBoxes.push_back(boxes);
	}

	int box_size = mb_size * m_nBoxCountPerImage * 4;
	m_pTrueBox = new float[box_size];
	memset(m_pTrueBox, 0, sizeof(float) * box_size);

	for (int n = 0; n < mb_size; n++) {
		KaiList boxes = imageBoxes[n];
		float* p = m_pTrueBox + n * m_nBoxCountPerImage * 4;
		for (int m = 0; m < (int)boxes.size(); m++) {
			KaiList box = boxes[m];
			for (int k = 0; k < 4; k++) {
				*p++ = box[k];
			}
		}
	}

	return true;
}

KBool Yolo3Feeder::m_feedIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KInt* pnBuffer) {
	return true;
}

KBool Yolo3Feeder::m_feedFreeIntData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KInt*& pnBuffer) {
	return true;
}

KBool Yolo3Feeder::m_feedFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape shape, KIntList nDatIndexs, KFloat* pfBuffer) {
	KInt mb_size = nDatIndexs.size();
	KInt dat_size = shape.total_size();

	if (bInput) {
		memcpy(pfBuffer, m_pImages, sizeof(KFloat) * mb_size * dat_size);
	}
	else if (sFieldName == "true_map") {
		memcpy(pfBuffer, m_pTrueMap, sizeof(KFloat) * mb_size * dat_size);
	}

	return true;
}

KBool Yolo3Feeder::m_feedFreeFloatData(void* pAux, KBool bInput, KString sFieldName, KaiShape& shape, KIntList nDatIndexs, KFloat*& pfBuffer) {
	KInt mb_size = nDatIndexs.size();

	if (sFieldName == "true_box") {
		pfBuffer = m_pTrueBox;
		shape = KaiShape{ mb_size, m_nBoxCountPerImage, 4 };

		/*
		for (KInt n = 0; n < shape.total_size(); n++) {
			if (n % 10 == 0) printf("\n> ");
			printf(" %f", pfBuffer[n]);
		}
		printf("\n\n");
		*/
	}

	return true;
}
