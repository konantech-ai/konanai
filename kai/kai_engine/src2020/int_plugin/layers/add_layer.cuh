/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class BatchNormalLayer;

class AddLayer : public ComplexLayer {
public:
	AddLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine);
	virtual ~AddLayer();

protected:
	bool m_add_x;
	bool m_b_cnn;

	Shape m_stride;

	string m_actions;

	BatchNormalLayer* m_pBNLayers;

protected:
	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
};
