/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "dummy.h"

DummyDataset::DummyDataset(string name, string mode, Shape ishape, Shape oshape) : Dataset(name, mode) {
    input_shape = ishape;
    output_shape = oshape;
}

DummyDataset::~DummyDataset() {
}

void DummyDataset::generate_data(int* data_idxs, int size, Value& xs, Value& ys) {
    throw KaiException(KERR_ASSERT);
}
void DummyDataset::visualize(Value xs, Value estimates, Value answers) {
    throw KaiException(KERR_ASSERT);
}
