/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class FlowerDataset : public Dataset {
public:
    FlowerDataset(string data_name, string cache_name, Shape resolution=Shape(100,100), Shape data_shape=Shape(30000));
    virtual ~FlowerDataset();

    //void generate_data(int* data_idxs, int size, Value& xs, Value& ys);
    void visualize(Value xs, Value estimates, Value answers);

protected:
#ifdef KAI2021_WINDOWS
    void create_cache(string data_path, Shape resolution, Shape data_shape);
    void save_cache(string cache_path);
    void load_cache(string cache_path);
#else
#endif
    //Array<float> m_xs;
    //Array<float> m_ys;

    Shape m_resolution;

    vector<string> m_file_paths;
    vector<int> m_target_idx;

    vector<string> m_target_names;
};
