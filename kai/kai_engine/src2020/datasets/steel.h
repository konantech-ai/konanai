/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class SteelDataset : public Dataset {
public:
    SteelDataset();
    virtual ~SteelDataset();

    void visualize(Value xs, Value estimates, Value answers);

protected:
    vector<string> m_target_names;
};
