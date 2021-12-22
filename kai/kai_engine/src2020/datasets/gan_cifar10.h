/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class GanCifar10Dataset : public GanDataset {
public:
    GanCifar10Dataset(string name);
    virtual ~GanCifar10Dataset();

    virtual void visualize(Gan* model, Dict real_xs, Dict fake_xs);

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return false; }

protected:
    //Array<float> m_xs;
};
