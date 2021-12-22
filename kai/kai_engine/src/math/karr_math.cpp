/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "karr_math.h"

KaiArrayMath karr_math;

KInt KaiArrayMath::eval_arr_index(KaiShape shape, KInt nth1, KInt nth2, KInt nth3, KInt nth4, KInt nth5, KInt nth6, KInt nth7, KInt nth8) {
	if (shape.size() == 0) return 0;

	KInt index = 0;

	if (m_arr_index_step(shape, 0, nth1, index)) return index;
	if (m_arr_index_step(shape, 1, nth2, index)) return index;
	if (m_arr_index_step(shape, 2, nth3, index)) return index;
	if (m_arr_index_step(shape, 3, nth4, index)) return index;
	if (m_arr_index_step(shape, 4, nth5, index)) return index;
	if (m_arr_index_step(shape, 5, nth6, index)) return index;
	if (m_arr_index_step(shape, 6, nth7, index)) return index;
	if (m_arr_index_step(shape, 7, nth8, index)) return index;

	return index;
}

bool KaiArrayMath::m_arr_index_step(KaiShape shape, KInt axis, KInt cood, KInt& index) {
	if (shape.size() == axis) {
		if (cood >= 0) throw KaiException(KERR_ARRAY_TOO_MANY_AXIS_COODS);
		return true;
	}
	if (cood < 0 || cood >= shape[axis]) throw KaiException(KERR_ARRAY_BAD_AXIS_COOD);
	index = (axis > 0) ? index * shape[axis] + cood : cood;
	return false;
}
