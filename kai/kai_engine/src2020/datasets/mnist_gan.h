/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class GanDatasetPicture : public GanDataset {
public:
    GanDatasetPicture(string name, string filenam, string cache_name);
    virtual ~GanDatasetPicture();

    virtual void visualize(Gan* model, Dict real_xs, Dict fake_xs);

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return false; }

protected:
    //Array<float> m_xs;
};

class GanDatasetMnist : public GanDataset {
public:
    GanDatasetMnist(string name, string nums="");
    virtual ~GanDatasetMnist();

    virtual void visualize(Gan* model, Dict mixed_xs, Dict mixed_ys);

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return false; }

protected:
    //Array<unsigned char> m_images;
    Array<unsigned char> m_labels;

    vector<string> m_target_names;

    int m_length;
};
