/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"
#include "dropout_layer.cuh"

class AttentionLayer : public Layer {
public:
	AttentionLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~AttentionLayer();

protected:
	int m_num_heads;

	float m_coef;

	DropoutLayer* m_pDropoutLayers;

protected:
	Array<float> m_forward_farr(Array<float> hidden);
	Array<float> m_backprop_farr(Array<float> G_hidden);

	Array<float> m_forward_cuda_farr(Array<float> hidden);
	Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
	
	virtual int64 dump_structure(int64 depth);

};
