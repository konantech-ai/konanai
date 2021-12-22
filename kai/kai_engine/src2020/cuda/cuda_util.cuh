/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/common.h"
#include "../cuda/cuda_conn.cuh"

class CudaUtil {
public:
	static Array<float> WaveFFT(Array<float> wave_buffer, int64 step_width, int64 step_cnt, int64 fft_width, int64 freq_cnt);
};
