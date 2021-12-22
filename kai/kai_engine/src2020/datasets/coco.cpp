/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "coco.h"
#include "../core/engine.h"
#include "../core/log.h"

#ifdef KAI2021_WINDOWS
#else
#include <unistd.h>
#endif

/*
CocoDataset::CocoDataset(string name, string datetime) : Dataset(name, "classify", false, false, datetime) {
    logger.Print("dataset loading...");

    input_shape = Shape(416, 416, 3);
    output_shape = Shape(nvec_ancs);

    m_target_num = 0;
    m_image_num = 0;

    mp_grid_offsets = new Array<float>[3];
    for (int n = 0; n < num_scales; n++) {
        mp_grid_offsets[n].init_grid(Shape(grid_cnts[n], grid_cnts[n]));
        mp_grid_offsets[n] = CudaConn::ToCudaArray(mp_grid_offsets[n], "yolo_offset");
    }

    m_anchor_size = Array<int>(Shape(3, 3, 2));
    memcpy(m_anchor_size.data_ptr(), anchor_size, sizeof(int) * m_anchor_size.total_size());
    m_anchor_size = CudaConn::ToCudaArray(m_anchor_size, "yolo_anchors");

    m_grid_cnts = Array<int>(Shape(3));
    memcpy(m_grid_cnts.data_ptr(), grid_cnts, sizeof(int) * m_grid_cnts.total_size());
    m_grid_cnts = CudaConn::ToCudaArray(m_grid_cnts, "grid_cnts");

    mp_target_ids = NULL;
    mp_image_ids = NULL;
    mp_image_widths = NULL;
    mp_image_heights = NULL;

    mp_target_names = NULL;
    mp_box_info = NULL;

    input_shape = Shape(416, 416, 3);
    output_shape = Shape(nvec_ancs);

    assert(name == "coco");

    string inst_path = KArgs::data_root + "coco/annotations/instances_train2014.json";
    string box_cache_path = KArgs::cache_root + "coco_boxes.cache";

    if (mode == "step_debug") {
        m_load_cache(box_cache_path, true);
        logger.Print("empty dataset for step_debug was prepared...");
        return;
    }

    if (!m_load_cache(box_cache_path)) {
        Dict inst_info = Value::parse_json_file(inst_path);  // LR-파싱으로 빠르게 처리 가능할 듯
        m_create_box_info(box_cache_path, inst_info);
    }

    m_shuffle_index(m_image_num);

    logger.Print("tr_cnt = %d, va_cnt = %d, vi_cnt = %d, te_cnt = %d", m_data_count[data_channel::train], m_data_count[data_channel::validate], m_data_count[data_channel::visualize], m_data_count[data_channel::test]);

    logger.Print("dataset prepared...");
}

YoloCocoDataset::~YoloCocoDataset() {
    delete[] mp_target_ids;
    delete[] mp_image_ids;
    delete[] mp_image_widths;
    delete[] mp_image_heights;

    delete[] mp_target_names;
    delete[] mp_box_info;
    delete[] mp_grid_offsets;
}

*/