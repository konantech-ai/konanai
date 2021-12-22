/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../layer.cuh"

class BatchNormalLayer : public Layer {
public:
	BatchNormalLayer(Dict options, Shape& shape, bool& seq, Engine& engine);
	virtual ~BatchNormalLayer();

	void m_forward_bn_core(CudaConn& cuda, float* cuda_h, Shape hshape);
	void m_backprop_bn_core(CudaConn& cuda, float* cuda_gh, Shape hshape);

	virtual int64 dump_structure(int64 depth);

protected:
	bool m_rescale;
	float m_epsilon;
	float m_momentum;
	Shape m_bn_shape;
	Array<float> m_std;

	virtual Array<float> m_forward_farr(Array<float> hidden);
	virtual Array<float> m_backprop_farr(Array<float> G_hidden);

	virtual Array<float> m_forward_cuda_farr(Array<float> hidden);
	virtual Array<float> m_backprop_cuda_farr(Array<float> G_hidden);
};

