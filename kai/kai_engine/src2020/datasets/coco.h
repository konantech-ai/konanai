/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class CocoDataset : public Dataset {
public:
    CocoDataset(string name, string datetime = "");
    virtual ~CocoDataset();

    //void generate_data(int* data_idxs, int size, Value& xs, Value& ys);
    void visualize(Value xs, Value estimates, Value answers);

protected:
    /*
    //Array<float> m_xs;
    //Array<float> m_ys;

    Shape m_resolution;

    vector<string> m_file_paths;
    vector<int> m_target_idx;

    vector<string> m_target_names;
    */
};
