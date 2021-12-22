/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class VideoShotDataset : public Dataset {
public:
    VideoShotDataset(string movie_path, string cache_name, int64 sample_cnt, int64 timesteps, Shape frame_shape);
    virtual ~VideoShotDataset();

    virtual bool input_seq() { return true; }
    virtual bool output_seq() { return true; }

    virtual int64 input_timesteps() { return m_timesteps; }
    virtual int64 output_timesteps() { return m_timesteps; }

    void visualize(Value xs, Value estimates, Value answers);

protected:
#ifdef KAI2021_WINDOWS
    void create_cache(string movie_path, int64 sample_cnt, int64 timesteps, Shape frame_shape);
    void save_cache(string cache_path);
    void load_cache(string cache_path);
#else
#endif

    int64 m_timesteps;

    Array<int64> m_frame_num;
    Array<bool> m_is_next_frame;

    vector<string> m_target_names;
};
