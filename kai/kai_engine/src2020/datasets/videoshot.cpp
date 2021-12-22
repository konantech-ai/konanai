/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "videoshot.h"
#include "../core/util.h"
#include "../core/random.h"
#include "../core/log.h"

VideoShotDataset::VideoShotDataset(string movie_path, string cache_name, int64 sample_cnt, int64 timesteps, Shape frame_shape) : Dataset("videoshot", "binary") {
	logger.Print("dataset loading...");

	string data_path = KArgs::data_root + movie_path;
	string cache_path = KArgs::cache_root + "videoshot";

	Util::mkdir(KArgs::cache_root);
	Util::mkdir(cache_path);

	cache_path += "/" + cache_name;

#ifdef KAI2021_WINDOWS
	try {
		load_cache(cache_path);
	}
	catch (exception err) {
		create_cache(data_path, sample_cnt, timesteps, frame_shape);
		save_cache(cache_path);
	}
#else
	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 0);
#endif

	Shape xshape = m_default_xs.shape();

	int64 total_count = xshape[0], tm_steps = xshape[1], height = xshape[2], width = xshape[3], chn = xshape[4];

	m_timesteps = tm_steps;

	input_shape = Shape(height, width, chn);
	output_shape = Shape(1);

	m_shuffle_index(total_count);

	logger.Print("dataset prepared...");
}

VideoShotDataset::~VideoShotDataset() {
}

#ifdef KAI2021_WINDOWS
void VideoShotDataset::load_cache(string cache_path) {
	logger.Print("load_cache start time: %lld", time(NULL));
	Util::load_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 0);
	logger.Print("load_cache end time: %lld", time(NULL));
}

void VideoShotDataset::save_cache(string cache_path) {
	logger.Print("save_cache start time: %lld", time(NULL));
	Util::save_kodell_dump_file(cache_path, m_default_xs, m_default_ys, &m_target_names, 0);
	logger.Print("save_cache end time: %lld", time(NULL));
}

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

void VideoShotDataset::create_cache(string movie_path, int64 sample_cnt, int64 timesteps, Shape frame_shape) {
	logger.Print("create_cache start time: %lld", time(NULL));

	vector<string> file_names = Util::list_dir(movie_path);

	vector<Array<float>> slices;
	vector<size_t> cat_idxs;

	for (vector<string>::iterator it = file_names.begin(); it != file_names.end(); it++) {
		string file_path = movie_path + '/' + *it;
#ifdef KAI2021_WINDOWS
		std::replace(file_path.begin(), file_path.end(), '/', '\\');
#endif
		VideoCapture cap(file_path.c_str(), CAP_FFMPEG);

		if (!cap.isOpened()) {
			continue;
		}

		int64 frame_Number = (int64) cap.get(CAP_PROP_FRAME_COUNT);

		Shape load_shape = frame_shape.add_front(timesteps).add_front(sample_cnt);
		Array<float> frames(load_shape);

		float* pBuf = frames.data_ptr();

		for (int64 n = 0; n < sample_cnt; n++) {
			int64 left_size = Random::dice(timesteps - 1) + 1;
			int64 right_size = timesteps - left_size;

			int64 left_pos = Random::dice(frame_Number - left_size);
			int64 right_pos = Random::dice(frame_Number - right_size);

			cat_idxs.push_back(left_size);

			Mat frame;

			cap.set(CAP_PROP_POS_FRAMES, (double) left_pos);

			for (int64 m = 0; m < timesteps; m++) {
				if (m == left_size) cap.set(CAP_PROP_POS_FRAMES, (double) right_pos);

				if (!cap.read(frame)) {
					logger.Bookeep("Cannot read %d-th(%d, %d, %d) frame in %s\n", m, left_pos, right_pos, left_size, *it);
					throw KaiException(KERR_ASSERT);
				}
				
				cv::resize(frame, frame, cv::Size((int)frame_shape[0], (int)frame_shape[1]), 0, 0, cv::INTER_CUBIC);
				
				int cn = frame.channels();
				assert(cn == frame_shape[2]);

				for (int i = 0; i < frame.cols; i++) {
					for (int j = 0; j < frame.rows; j++) {
						cv::Vec3b intensity = frame.at<cv::Vec3b>(j, i);
						for (int k = 0; k < cn; k++) {
							*pBuf++ = (float)intensity.val[k];
						}
					}
				}
			}
		}

		assert(pBuf - frames.data_ptr() == frames.total_size());

		cap.release();

		slices.push_back(frames);
	}

	int64 total_count = (int) slices.size() * sample_cnt;

	m_default_xs = hmath.vstack(slices); // .merge_time_axis();
	m_default_ys = Array<float>::zeros(Shape(total_count, timesteps, 1));

	for (int64 n = 0; n < total_count; n++) {
		m_default_ys[Idx(n, cat_idxs[n], 0)] = 1.0f;
	}

	logger.Print("create_cache end time: %lld", time(NULL));
}
#else
#endif

void VideoShotDataset::visualize(Value cxs, Value cest, Value cans) {
	m_show_seq_binary_results(cest, cans);
}
