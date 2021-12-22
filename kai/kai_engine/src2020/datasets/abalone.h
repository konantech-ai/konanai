/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class AbaloneDataset : public Dataset {
public:
    AbaloneDataset();
    virtual ~AbaloneDataset();

    //void gen_data(int data_idx, int64 xsize, float* px, int64 ysize, float* py);
    //void generate_data(int* data_idxs, int size, Value& xs, Value& ys); // depreciated
    void visualize(Value xs, Value estimates, Value answers);

protected:
};
