/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../optimizer.cuh"

class SgdOptimizer : public Optimizer {
public:	// functions
	SgdOptimizer(Dict initArgs);
	virtual ~SgdOptimizer();

	void setup(Dict kwArgs);
	
	void update_weight(Dict pm_w, Array<float> G_weight);
	void update_bias(Dict pm_b, Array<float> G_bias);
	void update_embed(Dict param, Array<float> G_words, Array<int64> selector);

	void update_weight_cuda(Dict param, Array<float> G_weight);
	void update_bias_cuda(Dict param, Array<float> G_bias);
	void update_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector);

protected: // functions

public: // variables

protected: // variables
};
