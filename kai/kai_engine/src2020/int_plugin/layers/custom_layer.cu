/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "custom_layer.cuh"

CustomLayer::CustomLayer(Dict options, Shape& shape, bool& seq, Engine& engine) : ComplexLayer(options, List(), shape, seq, engine) {
    string name = get_option("name");
    //Dict args = get_option("args", Dict());

    List hconfigs = m_engine.get_macro(name, options);
    List pms;
   
    engine.build_hidden_net(hconfigs, shape, seq, m_layers, pms);

    /*
    Layer* pPlayer = Layer::CreateLayer(hconfig, shape, seq, m_engine);

    m_layers.push_back(pPlayer);
    m_pms.push_back(pPlayer->m_pm);
    */

    m_param["pms"] = pms;

    m_output_shape = shape;
}

CustomLayer::~CustomLayer() {
}

Array<float> CustomLayer::m_forward_farr(Array<float> hidden) {
    for (vector<Layer*>::iterator it = m_layers.begin(); it != m_layers.end(); it++) {
        Layer* pLayer = *it;
        hidden = pLayer->forward_subnet(hidden);
    }
    return hidden;
}

Array<float> CustomLayer::m_backprop_farr(Array<float> G_hidden) {
    for (vector<Layer*>::reverse_iterator it = m_layers.rbegin(); it != m_layers.rend(); it++) {
        Layer* pLayer = *it;
        G_hidden = pLayer->backprop_subnet(G_hidden);
    }

    return G_hidden;
}

Array<float> CustomLayer::m_forward_cuda_farr(Array<float> hidden) {
    return m_forward_farr(hidden);
}

Array<float> CustomLayer::m_backprop_cuda_farr(Array<float> G_hidden) {
    return m_backprop_farr(G_hidden);
}
