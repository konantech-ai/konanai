/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class DummyDataset : public Dataset {
public:
    DummyDataset(string name, string mode, Shape ishape, Shape oshape);
    virtual ~DummyDataset();

    void generate_data(int* data_idxs, int size, Value& xs, Value& ys);
    void visualize(Value xs, Value estimates, Value answers);

protected:
};
