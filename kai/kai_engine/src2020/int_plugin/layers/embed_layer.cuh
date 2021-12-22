/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class EmbedLayer : public Layer {
public:
	EmbedLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~EmbedLayer();

	virtual Dict m_get_wvec_param();

protected:
	virtual Dict m_forward_main(Dict hidden);
	virtual Dict m_backprop_main(Dict G_hidden);

	virtual Dict m_forward_cuda_main(Dict hin);
	virtual Dict m_backprop_cuda_main(Dict G_hidden);

	virtual Array<float> m_forward_narr(Array<int64> hidden);
	virtual Array<float> m_backprop_narr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_narr(Array<int64> hidden);
	virtual Array<float> m_backprop_cuda_narr(Array<float> G_hidden);

	virtual int64 dump_structure(int64 depth);

	List m_plugin_names;
	
/*
	int64 m_dic_count;
	int64* m_voc_counts;
	int64* m_cuda_voc_counts;
*/
};

