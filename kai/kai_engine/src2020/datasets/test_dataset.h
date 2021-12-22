/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class TestDataset : public Dataset {
public:
    TestDataset();
    virtual ~TestDataset();

    void generate_data(int* data_idxs, int size, Array<float>& xs, Value& ys);
    void visualize(Array<float> xs, Value estimates, Value answers);

protected:
};
