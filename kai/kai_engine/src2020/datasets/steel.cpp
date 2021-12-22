/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "steel.h"

SteelDataset::SteelDataset() : Dataset("steel", "classify") {
	vector<string> header;
	vector<vector<string>> rows = Util::load_csv(KArgs::data_root + "chap03/faults.csv", &header);

	int64 data_cnt = rows.size();
	int64 field_cnt = rows[0].size();
	int64 nom_cnt = 7;
	int64 feat_cnt = field_cnt - nom_cnt;

	input_shape = Shape(feat_cnt);
	output_shape = Shape(nom_cnt);

	m_default_xs = Array<float>::zeros(Shape(data_cnt, feat_cnt));
	m_default_ys = Array<float>::zeros(Shape(data_cnt, nom_cnt));

	for (int n = 0; n < data_cnt; n++) {
		vector<string> row = rows[n];

		for (int k = 0; k < feat_cnt; k++) {
			m_default_xs[Idx(n, k)] = std::stof(row[k]);
		}

		for (int k = 0; k < nom_cnt; k++) {
			m_default_ys[Idx(n, k)] = std::stof(row[feat_cnt + k]);
		}
	}

	m_target_names = vector<string>(header.cbegin()+feat_cnt, header.cend());

	m_shuffle_index(data_cnt);
}

SteelDataset::~SteelDataset() {
}

void SteelDataset::visualize(Value cxs, Value cest, Value cans) {
	m_show_select_results(cest, cans, m_target_names);
}
