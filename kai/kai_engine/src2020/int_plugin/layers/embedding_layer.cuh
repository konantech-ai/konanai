/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class EmbeddingLayer : public ComplexLayer {
public:
	EmbeddingLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine);
	virtual ~EmbeddingLayer();

	virtual Dict m_get_wvec_param();

protected:
	virtual Dict m_forward_main(Dict hidden);
	virtual Dict m_backprop_main(Dict G_hidden);

	virtual Dict m_forward_cuda_main(Dict hidden);
	virtual Dict m_backprop_cuda_main(Dict G_hidden);

	virtual Array<float> m_forward_embedding(Array<int64> hint, Array<int64> noms);
	virtual Array<float> m_backprop_embedding(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_embedding(Array<int64> hint, Array<int64> noms);
	virtual Array<float> m_backprop_cuda_embedding(Array<float> G_hidden);

	int64 m_vec_size;

	int64 m_in_cnt;
	int64 m_out_cnt;
};

