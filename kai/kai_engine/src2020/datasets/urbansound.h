/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class _WavedataPicker {
public:
    _WavedataPicker(WaveInfo* pWaveInfo, int64 start_offset, int64 nFetchRate);
    virtual ~_WavedataPicker();

    float fetch(int64 nth);

protected:
    float m_get_sample(int64 nth);

    int64 m_start_offset;
    int64 m_fetch_rate;

    WaveInfo* m_pWaveInfo;
};

class UrbanSoundDataset : public Dataset {
public:
    UrbanSoundDataset(string data_name, string cache_name, int64 step_cnt=200, int64 step_win=10, int64 freq_win=16384, int64 freq_cnt=128);
    virtual ~UrbanSoundDataset();

    virtual bool input_seq() { return true; }
    virtual bool output_seq() { return false; }

    virtual int64 input_timesteps() { return m_timesteps; }

    //virtual void gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);

    void visualize(Value xs, Value estimates, Value answers);

protected:
#ifdef KAI2021_WINDOWS
    void create_cache(string data_path, int64 step_cnt, int64 step_win, int64 freq_win, int64 freq_cnt);
    void save_cache(string cache_path);
    void load_cache(string cache_path);

    void m_load_wave_data(string wav_path, float* pWaveBuffer, int64 fetch_width, int64 sample_rate);
    Array<float> m_ftt_analize(string wav_path, int64 step_cnt, int64 step_win, int64 freq_win, int64 freq_cnt);
#else
#endif

    int64 m_data_cnt, m_timesteps, m_timefeats;
    int64 m_target_cnt;

    //Array<float> m_xs;
    //Array<float> m_ys;

    vector<string> m_target_names;
};
