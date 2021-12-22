/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class PulsarDataset : public Dataset {
public:
    PulsarDataset();
    virtual ~PulsarDataset();

    //void generate_data(int* data_idxs, int size, Value& xs, Value& ys);
    void visualize(Value xs, Value estimates, Value answers);

protected:
    //Array<float> m_xs;
    //Array<float> m_ys;
};

class PulsarSelectDataset : public Dataset {
public:
    PulsarSelectDataset();
    virtual ~PulsarSelectDataset();

    //void generate_data(int* data_idxs, int size, Value& xs, Value& ys);
    void visualize(Value xs, Value estimates, Value answers);

protected:
    //Array<float> m_xs;
    //Array<float> m_ys;

    vector<string> m_targe_name;
};
