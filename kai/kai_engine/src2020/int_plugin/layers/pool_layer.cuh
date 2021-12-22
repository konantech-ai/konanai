/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "conv_layer.cuh"

class PoolLayer : public ConvPoolLayer {
public:
	PoolLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~PoolLayer();

protected:
	bool m_is_simple;
};

class MaxLayer : public PoolLayer {
public:
	MaxLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~MaxLayer();

protected:
	Shape m_hshape;

	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
};

class AvgLayer : public PoolLayer {
public:
	AvgLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~AvgLayer();

protected:
	Array<float> m_mask;

	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
};

class GlobalAvgLayer : public AvgLayer {
public:
	GlobalAvgLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~GlobalAvgLayer();
};
