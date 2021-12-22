/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class DropoutLayer : public Layer {
public:
	DropoutLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~DropoutLayer();

	float m_keep_prob;

	Array<float> create_mask(Shape shape);

protected:
	friend class AttentionLayer;

	Array<float> m_drop_mask;

	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
};
