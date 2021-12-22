/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../session/kcommon.h"
#include "../math/karray.h"

class KaiArrayMath {
public:
	KInt eval_arr_index(KaiShape shape, KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8);

protected:
	bool m_arr_index_step(KaiShape shape, KInt axis, KInt cood, KInt& index);
};

extern KaiArrayMath karr_math;
