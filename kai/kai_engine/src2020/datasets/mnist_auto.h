/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class MnistAutoDataset : public AutoencodeDataset {
public:
    MnistAutoDataset(float ratio = 1.0);
    virtual ~MnistAutoDataset();

    virtual void gen_plain_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y);
    virtual void gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);

    virtual void visualize(Value xs, Value estimates, Value answers);
    virtual void visualize_autoencode(Dict xs, Dict code, Dict repl, Dict outs, Dict ys);
    virtual void visualize_hash(Array<int64> rank1, Array<int64> rank2, Array<int64> key_labels, Array<int64> dat_label, Array<float> distance, Array<float> keys, Array<float> repl, Array<float> xs);

protected:
    Array<unsigned char> m_images;
    Array<unsigned char> m_labels;

    vector<string> m_target_names;
};
