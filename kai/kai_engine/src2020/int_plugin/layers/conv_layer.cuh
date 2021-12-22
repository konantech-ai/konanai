/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class BatchNormalLayer;

class ConvPoolLayer : public Layer {
public:
	ConvPoolLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~ConvPoolLayer();

protected:
	Shape m_xshape, m_stride, m_ksize;
	int m_xchn, m_ychn;
	string m_padding;

protected:
	Array<float> m_stride_filter(Array<float> hidden);
	Array<float> m_stride_filter_derv(Array<float> G_hidden);

	Array<float> m_get_ext_regions(Array<float> x, float fill);
	Array<float> m_undo_ext_regions(Array<float> G_hidden);
};

class ConvLayer : public ConvPoolLayer {
public:
	ConvLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~ConvLayer();

protected:
	string m_actions;

	BatchNormalLayer* m_pBNLayers;

protected:
	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);

	virtual int64 dump_structure(int64 depth);
};
