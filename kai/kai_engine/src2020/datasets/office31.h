/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class Office31Dataset : public Dataset {
public:
    Office31Dataset(string data_name, string cache_name, Shape resolution = Shape(100, 100), Shape data_shape = Shape(30000));
    virtual ~Office31Dataset();

    virtual Dict forward_postproc(Dict xs, Dict ys, Dict outs, string mode);
    virtual Dict backprop_postproc(Dict ys, Dict outs, string mode);
    virtual Dict eval_accuracy(Dict xs, Dict ys, Dict outs, string mode);

    virtual void log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2);
    virtual void log_test(string name, Dict acc, int64 tm1, int64 tm2); 
    
    void visualize(Value xs, Value estimates, Value answers);

protected:
//hs.cho
//#ifdef KAI2021_WINDOWS
    void create_cache(string data_path, Shape resolution, Shape data_shape);
    void save_cache(string cache_path);
    void load_cache(string cache_path);
//#else
//#endif

    Shape m_resolution;

    vector<string> m_file_paths;
    vector<int> m_target_idx;

    int64 m_domain_cnt, m_product_cnt;

    vector<string> m_target_names[2];
};
