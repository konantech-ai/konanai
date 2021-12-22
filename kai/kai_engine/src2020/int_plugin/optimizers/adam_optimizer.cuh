/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../optimizer.cuh"

class AdamOptimizer : public Optimizer {
public:	// functions
	AdamOptimizer(Dict initArgs);
	virtual ~AdamOptimizer();

	void setup(Dict kwArgs);

	string introduce_extra();

	void m_alloc_affine_param(Dict& param, Shape shape, bool use_bias, Dict kwArgs);
	void m_alloc_embed_param(Dict& param, vector<int64> voc_sizes, int64 vec_size, Dict kwArgs);

	void update_weight(Dict param, Array<float> G_weight);
	void update_bias(Dict param, Array<float> G_bias);
	void update_embed(Dict param, Array<float> G_words, Array<int64> selector);

	void update_weight_cuda(Dict param, Array<float> G_weight);
	void update_bias_cuda(Dict param, Array<float> G_bias);
	void update_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector);

protected: // functions
	Array<float> m_eval_adam_delta(Dict param, Array<float> G_param, float n, int64 row=-1);

public: // variables

protected: // variables
	float m_ro1;
	float m_ro2;
	float m_epsilon;
};
