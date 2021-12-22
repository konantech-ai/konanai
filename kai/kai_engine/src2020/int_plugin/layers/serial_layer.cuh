/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class SerialLayer : public ComplexLayer {
public:
	SerialLayer(Dict options, List layer_subnet, Shape& shape, bool& seq, Engine& engine);
	virtual ~SerialLayer();

protected:
	int m_wrap_axis;
	int m_wrap_size;

	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
};
