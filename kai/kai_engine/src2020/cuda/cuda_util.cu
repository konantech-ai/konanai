/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../cuda/cuda_util.cuh"
#include "../cuda/cuda_math.h"
#include "../cuda/cuda_kernels.h"

#include "../core/array.h"
//hs.cho
#include <iostream>
Array<float> CudaUtil::WaveFFT(Array<float> wave_buffer, int64 step_width, int64 step_cnt, int64 fft_width, int64 freq_cnt) {
	CudaConn conn("WaveFFT", NULL);

	int64 mb_size = wave_buffer.axis_size(0);
	int64 fetch_width = step_width * (step_cnt - 1) + fft_width;

	assert(fetch_width == wave_buffer.axis_size(1));

	Array<float> fftResult(Shape(mb_size, step_cnt, freq_cnt));

	int num_gpus;
	size_t free, total;
	cudaGetDeviceCount(&num_gpus);

	for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
		cudaSetDevice(gpu_id);
		int id;
		cudaGetDevice(&id);
		cudaMemGetInfo(&free, &total);
		cout << "GPU " << id << " memory: free=" << free << ", total=" << total << endl;
	}

	int64 max_size = free / (sizeof(float) * (fetch_width + step_cnt * (4 * fft_width + freq_cnt)));

	logger.Print("max_size = %lld\n", max_size);
	//int64 piece_size = 1;
	//int64 split = 10;
	int64 piece_size = max_size; // 100;
	int64 split = 1;

	Shape wavShape(piece_size, fetch_width);
	Shape bufShape(piece_size, step_cnt, fft_width, 2);
	Shape fftShape(piece_size, step_cnt, freq_cnt);

	float* cuda_wave = conn.alloc_float_mem(wavShape);
	float* cuda_buf1 = conn.alloc_float_mem(bufShape);
	float* cuda_buf2 = conn.alloc_float_mem(bufShape);
	float* cuda_fft = conn.alloc_float_mem(fftShape);

	int64 bsize = bufShape.total_size();
	int64 fsize = fftShape.total_size();

	float* pWave = wave_buffer.data_ptr();
	float* pFFT = fftResult.data_ptr();

	int64 rest_size = mb_size;

	while (rest_size > 0) {
		logger.Print("fft loop rest_size = %lld", rest_size);

		//CudaConn conn("WaveFFT", NULL);

		int64 slice_size = (rest_size >= piece_size) ? piece_size : rest_size;

		/*
		Shape wavShape(slice_size, fetch_width);
		Shape bufShape(slice_size, step_cnt, fft_width, 2);
		Shape fftShape(slice_size, step_cnt, freq_cnt);

		float* cuda_wave = conn.alloc_float_mem(wavShape);
		float* cuda_buf1 = conn.alloc_float_mem(bufShape);
		float* cuda_buf2 = conn.alloc_float_mem(bufShape);
		float* cuda_fft = conn.alloc_float_mem(fftShape);

		int64 bsize = bufShape.total_size();
		int64 fsize = fftShape.total_size();
		*/

		cudaMemcpy(cuda_wave, pWave, sizeof(float)* slice_size * fetch_width, cudaMemcpyHostToDevice);
		cu_call(ker_wave_slices_to_complex, bsize, (bsize, cuda_buf1, cuda_wave, step_width, step_cnt, fft_width, fetch_width));
		cmath.fft_core_split(cuda_buf1, cuda_buf2, cuda_fft, fft_width, freq_cnt, fsize, bsize, split);
		cudaMemcpy(pFFT, cuda_fft, sizeof(float) * slice_size * step_cnt * freq_cnt, cudaMemcpyDeviceToHost);

		pWave += piece_size * fetch_width;
		pFFT += piece_size * step_cnt * freq_cnt;

		rest_size -= slice_size;
	}

	return fftResult;
}
