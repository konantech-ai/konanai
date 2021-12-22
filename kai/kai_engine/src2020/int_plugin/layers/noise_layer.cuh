/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class NoiseLayer : public Layer {
public:
	NoiseLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~NoiseLayer();

protected:
	string m_noise_type;
	float m_mean, m_std, m_ratio;

	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
};
