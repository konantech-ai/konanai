/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "yolo.h"
#include "../core/engine.h"
#include "../core/random.h"
#include "../core/log.h"

#ifdef KAI2021_WINDOWS
#else
#include <unistd.h>
//hs.cho
#define sprintf_s snprintf
#define _fseeki64 fseeko64
#endif

YoloCocoDataset::YoloCocoDataset(string name, string mode, string datetime) : Dataset(name, "yolo", false, false, datetime) {
    logger.Print("dataset loading...");

    m_trace = false;

    m_mode = mode;

    m_target_num = 0;
    m_image_num = 0;

    mp_grid_offsets = new Array<float>[3];
    for (int64 n = 0; n < num_scales; n++) {
        mp_grid_offsets[n].init_grid(Shape(grid_cnts[n], grid_cnts[n]));
        mp_grid_offsets[n] = CudaConn::ToCudaArray(mp_grid_offsets[n], "yolo_offset");
    }

    m_anchor_size = Array<int64>(Shape(3, 3, 2));
    memcpy(m_anchor_size.data_ptr(), anchor_size, sizeof(int64)* m_anchor_size.total_size());
    m_anchor_size = CudaConn::ToCudaArray(m_anchor_size, "yolo_anchors");

    m_grid_cnts = Array<int64>(Shape(3));
    memcpy(m_grid_cnts.data_ptr(), grid_cnts, sizeof(int64)* m_grid_cnts.total_size());
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

    /*
    for (int64 n = 0; n < num_scales; n++) {
        m_grid_offsets[n].init_grid(Shape(grid_cnts[n], grid_cnts[n]));
        m_grid_offsets[n].print("m_grid_offsets");
    }
    */

    // fill map<int64, List> m_box_info; here

    m_shuffle_index(m_image_num);

    logger.Print("tr_cnt = %d, va_cnt = %d, vi_cnt = %d, te_cnt = %d", m_data_count[data_channel::train], m_data_count[data_channel::validate], m_data_count[data_channel::visualize], m_data_count[data_channel::test]);

    logger.Print("dataset prepared...");
}

YoloCocoDataset::~YoloCocoDataset() {
    delete [] mp_target_ids;
    delete [] mp_image_ids;
    delete [] mp_image_widths;
    delete [] mp_image_heights;

    delete [] mp_target_names;
    delete [] mp_box_info;
    delete [] mp_grid_offsets;
}

void YoloCocoDataset::step_debug() {
    m_step_debug_forward();
    // m_step_debug_visualize();
}

void YoloCocoDataset::m_step_debug_forward() {
    Dict xs, ys, estimate;

    int64 grid_cnt = 13;

    for (int64 nscale = 0; nscale < 3; nscale++) {
        Array<float> fmap(Shape(6, grid_cnt, grid_cnt, nvec_ancs));
        float* bp = fmap.data_ptr();
        int64 dsize = fmap.total_size();

        string fname = KArgs::data_root + "yolotest/train/fmaps" + to_string(nscale + 1) + ".npy";

        FILE* fid = Util::fopen(fname.c_str(), "rb");
        fseek(fid, 0, SEEK_END);

        int64 last_offset = ftell(fid);
        int64 start_offset = last_offset - dsize * sizeof(float);

        _fseeki64(fid, start_offset, SEEK_SET);
        if ((int64)fread(bp, sizeof(float), dsize, fid) != dsize) throw KaiException(KERR_ASSERT);
        fclose(fid);

        string map_name = "feature_map_" + to_string(nscale + 1);

        estimate[map_name] = Value::wrap_dict("data", CudaConn::ToCudaArray(fmap, "fmap" + to_string(nscale + 1)));

        grid_cnt *= 2;
    }

    gen_minibatch_data(data_channel::train, NULL, 6, xs, ys);

    Dict losses = forward_postproc(xs, ys, estimate, m_mode);

    //logger.Print("losses = %s", Value::description(losses).c_str());

    Dict grads = backprop_postproc(ys, estimate, m_mode);

    //logger.Print("grads = %s", Value::description(grads).c_str());
}

void YoloCocoDataset::m_step_debug_visualize() {
    Dict xs, ys, estimate;
    string mode = "yolo";

    int64 grid_cnt = 13;

    for (int64 nscale = 0; nscale < 3; nscale++) {
        Array<float> fmap(Shape(1, grid_cnt, grid_cnt, nvec_ancs));
        float* bp = fmap.data_ptr();

        int64 dsize = fmap.total_size();

        string fname = KArgs::data_root + "yolotest/fmaps" + to_string(nscale +1) + ".npy";

        FILE* fid = Util::fopen(fname.c_str(), "rb");

        fseek(fid, 0, SEEK_END);

        int64 last_offset = ftell(fid);
        int64 start_offset = last_offset - dsize * sizeof(float);

        _fseeki64(fid, start_offset, SEEK_SET);
        if ((int64) fread(bp, sizeof(float), dsize, fid) != dsize) throw KaiException(KERR_ASSERT);
        fclose(fid);

        string map_name = "feature_map_" + to_string(nscale + 1);

        //fmap.print("fmap" + to_string(nscale + 1));
        estimate[map_name] = Value::wrap_dict("data", CudaConn::ToCudaArray(fmap, "fmap" + to_string(nscale + 1)));

        grid_cnt *= 2;

        Array<int64> img_info(Shape(1, 3));
        List lst_image_path;

        img_info[Idx(0, 0)] = 0;
        img_info[Idx(0, 1)] = 1296;
        img_info[Idx(0, 2)] = 729;

        string image_path = "messi.jpg";

        lst_image_path.push_back(image_path);

        ys["image_info"] = img_info;
        ys["image_path"] = lst_image_path;
    }

    visualize_main(xs, ys, estimate);
}

int64 YoloCocoDataset::m_extract_mb_size(Dict outs) {
    Dict dict_est = outs["feature_map_1"];
    Array<float> fmap_est = dict_est["data"];
    return fmap_est.shape()[0];
}

void YoloCocoDataset::visualize_main(Dict xs, Dict ys, Dict outs) {
    logger.Print("YoloCocoDataset::visualize_main() function is called...");

    if (m_mode == "map_test") {
        Dict mAP_dict = m_evaluate_mAP(ys, outs, true);
        float mAP = mAP_dict["mAP"];
        logger.Print("mAP = %f", mAP);
        throw KaiException(KERR_ASSERT);
    }

    Dict est_info = m_predict(outs);

    Array<float> nms_boxes = est_info["nms_boxes"];
    Array<int64> img_info = ys["image_info"];

    Array<int64> arr_img_info = CudaConn::ToHostArray(img_info, "img_info");
    Array<float> arr_nms_boxes = CudaConn::ToHostArray(nms_boxes, "nms_boxes");

    int64 box_count = arr_nms_boxes.axis_size(0);

    int64 img_cnt = arr_img_info.axis_size(0);
    int64 obj_cnt = 0;

    float* p_nms_boxes = arr_nms_boxes.data_ptr();

    for (int64 n = 0; n < box_count; n++) {
        float* p_nms_box = p_nms_boxes + n * 8;
        if (p_nms_box[7] == 1) obj_cnt++;
    }

    List lst_image_path;

    if (ys.find("image_path") != ys.end()) {
        lst_image_path = ys["image_path"];
    }

    logger.Print("Visualize: %lld objects are detected in %lld images", obj_cnt, img_cnt);

    for (int64 n = 0; n < box_count; n++) {
        float* p_nms_box = p_nms_boxes + n * 8;
        if (p_nms_box[7] != 1) continue;

        int64 img_idx = (int64)p_nms_box[6];

        string image_path;
        if (img_idx == 0) image_path = (string)lst_image_path[img_idx];
        else {
            char path[128];
            sprintf_s(path,128, "COCO_train2014_%012lld.jpg", img_idx);
            image_path = path;
        }

        int64 image_width = arr_img_info[Idx(img_idx, 1)];
        int64 image_height = arr_img_info[Idx(img_idx, 2)];
        int64 max_size = MAX(image_width, image_height);

        float conf_score = p_nms_box[4];

        float left = p_nms_box[0];
        float top = p_nms_box[1];
        float right = p_nms_box[2];
        float bottom = p_nms_box[3];

        float ratio = (float)image_size / (float)max_size;

        float dw = ((float)image_size - (float)image_width * ratio) / 2.0f;
        float dh = ((float)image_size - (float)image_height * ratio) / 2.0f;

        float img_left = (left - dw) / ratio;
        float img_top = (top - dh) / ratio;
        float img_right = (right - dw) / ratio;
        float img_bottom = (bottom - dh) / ratio;

        string obj_name = mp_target_names[(int64)p_nms_box[5]];

        logger.Print("    [%lld] %s %dx%d %s conf:%f (%f, %f, %f, %f) => (%f, %f, %f, %f)", n, image_path.c_str(), image_width, image_height, obj_name.c_str(), conf_score,
            left, top, right, bottom, img_left, img_top, img_right, img_bottom);
    }
}

bool YoloCocoDataset::m_load_cache(string box_cache_path, bool no_image) {
    FILE* fid = Util::fopen(box_cache_path.c_str(), "rb");
    if (fid == NULL) return false;

    Value::serial_load(fid, m_target_id_map);
    Value::serial_load(fid, m_image_id_map);

    m_target_num = (int64)m_target_id_map.size();
    m_image_num = (int64)m_image_id_map.size();

    mp_target_ids = new int64[m_target_num];
    mp_image_ids = new int64[m_image_num];
    mp_image_widths = new int64[m_image_num];
    mp_image_heights = new int64[m_image_num];

    mp_target_names = new string [m_target_num];
    mp_box_info = new List [m_image_num];

    for (int64 n = 0; n < m_target_num; n++) {
        Value::serial_load(fid, mp_target_names[n]);
        //logger.Print("mp_target_names[%d] = %s", n, mp_target_names[n].c_str());
    }

    for (int64 n = 0; n < m_image_num; n++) {
        Value::serial_load(fid, mp_box_info[n]);
    }

    Value::serial_load(fid, mp_target_ids, m_target_num);
    Value::serial_load(fid, mp_image_ids, m_image_num);
    Value::serial_load(fid, mp_image_widths, m_image_num);
    Value::serial_load(fid, mp_image_heights, m_image_num);

    fclose(fid);
    return true;
}

void YoloCocoDataset::m_create_box_info(string box_cache_path, Dict& inst_info) {
    List categories = inst_info["categories"];
    List images = inst_info["images"];
    List annotations = inst_info["annotations"];

    m_target_num = (int64)categories.size();
    m_image_num = (int64)images.size();

    mp_target_ids = new int64[m_target_num];
    mp_image_ids = new int64[m_image_num];
    mp_image_widths = new int64[m_image_num];
    mp_image_heights = new int64[m_image_num];

    mp_target_names = new string[m_target_num];
    mp_box_info = new List[m_image_num];

    for (int64 n = 0; n < m_target_num; n++) {
        Dict cat_info = categories[n];
        m_target_id_map[cat_info["id"]] = n;
        mp_target_ids[n]= cat_info["id"];
        mp_target_names[n] = (string) cat_info["name"];
    }

    for (int64 n = 0; n < m_image_num; n++) {
        Dict image_info = images[n];
        m_image_id_map[image_info["id"]] = n;
        mp_image_ids[n] = image_info["id"];
        mp_image_widths[n] = image_info["width"];
        mp_image_heights[n] = image_info["height"];
    }

    for (int64 n = 0; n < (int64)annotations.size(); n++) {
        Dict annotation = annotations[n];
        int64 image_id = annotation["image_id"];
        int64 cat_id = annotation["category_id"];

        Dict box_info;
        box_info["cat"] = cat_id;
        box_info["box"] = annotation["bbox"];

        int64 nth = m_image_id_map[image_id];

        mp_box_info[nth].push_back(box_info);
    }

    FILE* fid = Util::fopen(box_cache_path.c_str(), "wb");
    
    if (fid == NULL) throw KaiException(KERR_ASSERT);
    
    Value::serial_save(fid, m_target_id_map);
    Value::serial_save(fid, m_image_id_map);

    for (int64 n = 0; n < m_target_num; n++) {
        Value::serial_save(fid, mp_target_names[n]);
    }

    for (int64 n = 0; n < m_image_num; n++) {
        Value::serial_save(fid, mp_box_info[n]);
    }

    Value::serial_save(fid, mp_target_ids, m_target_num);
    Value::serial_save(fid, mp_image_ids, m_image_num);
    Value::serial_save(fid, mp_image_widths, m_image_num);
    Value::serial_save(fid, mp_image_heights, m_image_num);

    fclose(fid);
}

int64 YoloCocoDataset::m_seek_image_id_idx(int64 image_id) {
    for (int64 n = 0; n < m_image_num; n++) {
        if (mp_image_ids[n] == image_id) return n;
    }
    logger.Print("m_seek_image_id_idx() failed: m_image_num = %lld, image_id = %lld", m_image_num, image_id);
    return 0;
}

Array<int64> YoloCocoDataset::m_get_image_idxs(int64* data_idxs, int64 size, Array<float> mix_ratio) {
    Array<int64> image_idxs(Shape(size, 2)); // [idx1, idx2] idx2 = 0 when mixed_up is not used

    int64* p_img_idxs = image_idxs.data_ptr();
    float* p_mix = mix_ratio.data_ptr();

    if (m_trace) logger.Print("m_mode = %s, data_idxs = 0x%llx", m_mode.c_str(), (unsigned long long) data_idxs);

    if (m_mode == "step_debug" && data_idxs == NULL) { // step_debug() for forward test 위한 임시 코드임
        assert(size == 6);
        static int64 test_1st_ids[6] = { 359156, 109, 91120, 91120, 309953, 91120 };
        //static int64 test_2nd_ids[6] = { 400848, -1, 359156, -1, -1, 375016  };
        //static float test_mix_ratio[6] = { 0.796078230933f, 1.0f, 0.512578859436f, 1.0f, 1.0f, 0.785506435471f };

        for (int64 n = 0; n < 6; n++) {
            *p_img_idxs++ = m_seek_image_id_idx(test_1st_ids[n]);
            //*p_img_idxs++ = (test_2nd_ids[n] >= 0) ? m_seek_image_id_idx(test_2nd_ids[n]) : -1;
            //*p_mix++ = test_mix_ratio[n];
            *p_img_idxs++ = -1;
            *p_mix++ = 1.0f;
        }

        if (m_trace) image_idxs.print("image_idxs");

        return image_idxs;
    }

    for (int64 n = 0; n < size; n++) {
        int64 idx2 = -1;
        float ratio = 1.0f;

        if (m_use_mixed && Random::uniform() < 0.5) {
            idx2 = (int64) Random::dice_except(m_image_num, data_idxs[n]);
            ratio = Random::beta(1.5, 1.5);
        }

        *p_img_idxs++ = data_idxs[n];
        *p_img_idxs++ = idx2;
        *p_mix++ = ratio;
    }

    if (m_trace) image_idxs.print("image_idxs");

    return image_idxs;
}

void YoloCocoDataset::gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys) {
    if (m_mode == "step_debug") {
        static bool only_once = true;
        if (m_trace) logger.Print("gen_minibatch_data() is called");
        if (only_once) only_once = false;
        else throw KaiException(KERR_ASSERT);
    }

    Array<float> mix_ratio(size);
    Array<int64> image_idxs = m_get_image_idxs(data_idxs, size, mix_ratio);
    
    if (m_trace) mix_ratio.print("mix_ratio");

    if (m_mode != "step_debug") {
        Array<float> xarr = m_load_images(image_idxs, mix_ratio);
        xs["default"] = Value::wrap_dict("data", xarr);
    }

    ys = m_create_scale_maps(image_idxs, mix_ratio);
}

Array<float> YoloCocoDataset::m_load_images(Array<int64> image_idxs, Array<float> mix_ratio) {
    int64 mb_size = image_idxs.axis_size(0);

    Array<unsigned char> buffer(Shape(mb_size, image_size, image_size, image_depth));
    Array<unsigned char> mixbuf(Shape(1, image_size, image_size, image_depth));

    unsigned char* bp = buffer.data_ptr();
    unsigned char* mp = mixbuf.data_ptr();

    int64 dsize = buffer.total_size() / mb_size;

    string dat_format = KArgs::data_root + "coco/train2014_dat/COCO_train2014_%012d.dat.npy";
    const char* format = dat_format.c_str();
    char fname[256];

    float* p_mix = mix_ratio.data_ptr();

    //logger.Print("gen_minibatch_data: size = %d", size);
    for (int64 n = 0; n < mb_size; n++) {
        int64 image_id1 = mp_image_ids[image_idxs[Idx(n, 0)]];

        snprintf(fname, 256, format, image_id1);

        //logger.Print("fname[%d]: %s", n, buf);

        FILE* fid = Util::fopen(fname, "rb");
        fseek(fid, 128, SEEK_SET);
        if (fread(bp, dsize, 1, fid) != 1) throw KaiException(KERR_ASSERT);
        fclose(fid);

        if (image_idxs[Idx(n, 1)] >= 0) {   // mixed_up data
            int64 image_id2 = mp_image_ids[image_idxs[Idx(n, 1)]];
            snprintf(fname, 256, format, image_id2);
            FILE* fid = Util::fopen(fname, "rb");
            fseek(fid, 128, SEEK_SET);
            if (fread(mp, dsize, 1, fid) != 1) throw KaiException(KERR_ASSERT);
            fclose(fid);

            float ratio = p_mix[n];

            for (int64 k = 0; k < dsize; k++) {
                bp[k] = (unsigned char) (bp[k] * ratio + mp[k] * (1 - ratio));
            }
        }

        bp += dsize;
    }

    Array<float> xarr = kmath->to_float(buffer) / 255.0f;

    return CudaConn::ToCudaArray(xarr, "xs");
}

/*
void YoloCocoDataset::m_select_scale_anchor_for_box(bool scale_anchor[][3], float width, float height, float ratio) {
    for (int64 ns = 0; ns < num_scales; ns++) {
        bool found = false;
        int64 best_idx = -1;
        float best_iou = -1.0;

        float sw = width * ratio;
        float sh = height * ratio;

        for (int64 na = 0; na < anchor_per_scale; na++) {
            scale_anchor[ns][na] = false;

            float mw = MIN(sw, (float)anchor_size[ns][na][0]);
            float mh = MIN(sh, (float)anchor_size[ns][na][1]);

            float iou = mw * mh / (sw* sh + (float)anchor_size[ns][na][0] * (float)anchor_size[ns][na][1] - mw * mh);

            if (iou > true_box_thr) {
                found = true;
                scale_anchor[ns][na] = true;
            }
            else if (iou > best_iou) {
                best_idx = na;
                best_iou = iou;
            }
        }

        if (!found) {
            scale_anchor[ns][best_idx] = true;
        }
    }
}
*/

void YoloCocoDataset::visualize(Value xs, Value estimates, Value answers) {
    throw KaiException(KERR_ASSERT);
}

Dict YoloCocoDataset::forward_postproc(Dict xs, Dict ys, Dict outs, string mode) {
    Dict losses;

    bool trace = (m_mode == "map_test") || (m_mode == "debug");

    Array<int64> img_info = ys["img_info"];
    Array<float> box_info = ys["box_info"];

    if (trace) {
        Value::print_dict_keys(ys, "ys");
        Value::print_dict_keys(outs, "outs");
        
        img_info.print("img_info", true);
        box_info.print("box_info", true);
    }

    for (int64 nscale = 0; nscale < num_scales; nscale++) {
        string key_name = "boxes_" + to_string(nscale + 1);

        if (ys.find(key_name) == ys.end()) {
            logger.Print("no box scale found...");
            continue; // 해당 scale에 걸친 박스 데이터가 전무한 경우
        }

        string map_name = "feature_map_" + to_string(nscale + 1);
        Dict dict_est = outs[map_name];
        Array<float> fmap_est = dict_est["data"];
        Array<int64> scale_boxes = ys[key_name];

        if (trace) {
            scale_boxes.print("scale_boxes", true);
        }

        //Dict pred_est = m_conv_layer_estimate(nscale, fmap_est, trace);
        Dict loss = m_evaluate_layer_loss(nscale, fmap_est, img_info, box_info, scale_boxes, trace);

        Value::dict_accumulate(losses, loss);
    }

    logger.Bookeep("losses = %s\n", Value::description(losses).c_str());

    return losses;
}

Dict YoloCocoDataset::backprop_postproc(Dict ys, Dict outs, string mode) {
    Dict G_output;

    bool trace = (m_mode == "map_test") || (m_mode == "debug");

    Array<int64> img_info = ys["img_info"];
    Array<float> box_info = ys["box_info"];

    if (trace) {
        Value::print_dict_keys(ys, "ys");
        Value::print_dict_keys(outs, "outs");

        img_info.print("img_info", true);
        box_info.print("box_info", true);
    }

    Dict G_fmap;

    for (int64 nscale = 0; nscale < num_scales; nscale++) {
        string key_name = "boxes_" + to_string(nscale + 1);

        if (ys.find(key_name) == ys.end()) {
            logger.Print("no box scale found...");
            continue; // 해당 scale에 걸친 박스 데이터가 전무한 경우
        }

        string map_name = "feature_map_" + to_string(nscale + 1);
        Dict dict_est = outs[map_name];
        Array<float> fmap_est = dict_est["data"];
        Array<int64> scale_boxes = ys[key_name];

        if (trace) {
            scale_boxes.print("scale_boxes", true);
        }

        //Dict pred_est = m_conv_layer_estimate(nscale, fmap_est, false);
        G_fmap = m_evaluate_layer_grad(nscale, fmap_est, img_info, box_info, scale_boxes, trace);

        G_output[map_name] = G_fmap;
    }

    //logger.Print("outs: %s", Value::description(outs).c_str());

    G_output["default"] = G_fmap; // YOLO 모델에서는 기본출력을 이용 않고 feature_map_N으로 이름 붙여 사용하지만 일관성 유지 위해 추가

    return G_output;
}

Dict YoloCocoDataset::eval_accuracy(Dict xs, Dict ys, Dict outs, string mode) {
    Dict acc = m_evaluate_mAP(ys, outs, false);
    return acc;
}

void YoloCocoDataset::log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2) {
    float loss_mean_coods = loss_mean["coods"];
    float loss_mean_sizes = loss_mean["sizes"];
    float loss_mean_confs = loss_mean["confs"];
    float loss_mean_probs = loss_mean["probs"];

    float acc_mAP = acc["mAP"];
    float acc_mean_mAP = acc_mean["mAP"];

    float loss_mean_sum = loss_mean_coods + loss_mean_sizes + loss_mean_confs + loss_mean_probs;

    if (batch_count == 0)
        logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
    else
        logger.PrintWait("    Batch %lld/%lld(in Epoch %d): ", batch, batch_count, epoch);

    logger.Print("loss=%16.9e(%16.9e+%16.9e+%16.9e+%16.9e), accuracy=(loss mean: %16.9e, validate:%16.9e) (%lld/%lld secs)",
        loss_mean_sum, loss_mean_coods, loss_mean_sizes, loss_mean_confs, loss_mean_probs,
        acc_mean_mAP, acc_mAP, tm1, tm2);
}

void YoloCocoDataset::log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2) {
    float loss_mean_coods = loss_mean["coods"];
    float loss_mean_sizes = loss_mean["sizes"];
    float loss_mean_confs = loss_mean["confs"];
    float loss_mean_probs = loss_mean["probs"];

    float acc_mean_mAP = acc_mean["mAP"];

    float loss_mean_sum = loss_mean_coods + loss_mean_sizes + loss_mean_confs + loss_mean_probs;

    if (batch_count == 0)
        logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
    else
        logger.PrintWait("    Batch %ld/%lld(in Epoch %lld): ", batch, batch_count, epoch);

    logger.Print("loss=%16.9e(%16.9e+%16.9e+%16.9e+%16.9e), accuracy=%16.9e (%lld/%lld secs)",
        loss_mean_sum, loss_mean_coods, loss_mean_sizes, loss_mean_confs, loss_mean_probs,
        acc_mean_mAP, tm1, tm2);
}

void YoloCocoDataset::log_test(string name, Dict acc, int64 tm1, int64 tm2) {
    float acc_mAP = acc["mAP"];

    logger.Print("Model %s test report: accuracy = %16.9e, (%lld/%lld secs)\n", name.c_str(), acc_mAP, tm2, tm1);
    logger.Print("");
}

/*
Dict YoloCocoDataset::m_extract_preds_on_layer(int64 ns, Array<float> fmap, List true_boxes) {
    vector<int64> chns;

    chns.push_back(2);
    chns.push_back(2);
    chns.push_back(1);
    chns.push_back(class_num);
    chns.push_back(1);

    vector<Array<float>> slices = kmath->hsplit(fmap, chns);

    Array<float> pred_xy = kmath->sigmoid(slices[0]);
    Array<float> pred_wh = kmath->exp(slices[1]);
    Array<float> pred_conf = slices[2];
    Array<float> pred_class = slices[3];
    Array<float> mix_w = slices[4];

    Array<float> ignore_mask = kmath->zeros(pred_conf.shape());

    int64 mb_size = ignore_mask.axis_size(0);

    Array<float> pred_box_xy = (pred_xy + mp_grid_offsets[ns]) * (float)grid_size[ns];
    Array<float> pred_box_wh = pred_wh * (float)grid_size[ns];

    for (int64 nd = 0; nd < mb_size; nd++) {
        Dict img_true_boxes = true_boxes[nd];
        if (img_true_boxes.size() == 0) {
            ignore_mask.set_row(nd, 1.0);
            continue;
        }
        Array<float> true_xy = img_true_boxes["box_xy"];
        Array<float> true_wh = img_true_boxes["box_wh"];
        Array<float> iou = m_eval_box_iou(pred_box_xy, pred_box_wh, true_xy, true_wh);
        Array<float> best_iou = kmath->max(iou, -1);
        Array<bool> ignore = best_iou < 0.5;
        ignore_mask.set_row(nd, ignore.to_float());
    }

    Dict preds;

    preds["box_xy"] = pred_xy;
    preds["box_wh"] = pred_wh;
    preds["conf"] = pred_conf;
    preds["class"] = pred_class;
    preds["mix_w"] = mix_w;
    preds["ignore_mask"] = ignore_mask;

    return preds;
}

Array<float> YoloCocoDataset::m_eval_box_iou(Array<float> pred_xy, Array<float> pred_wh, Array<float> true_xy, Array<float> true_wh) {
    // shape: [13, 13, 3, 1, 2]
    pred_xy = pred_xy.add_axis(-2);
    pred_wh = pred_wh.add_axis(-2);

    // shape: [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
    Array<float> intersect_mins = kmath->maximum(pred_xy - pred_wh / 2.0f, true_xy - true_wh / 2.0f);
    Array<float> intersect_maxs = kmath->maximum(pred_xy + pred_wh / 2.0f, true_xy + true_wh / 2.0f);

    // shape: [13, 13, 3, V, 2]
    Array<float> intersect_wh = kmath->maximum(intersect_maxs - intersect_mins, 0);

    // shape: [13, 13, 3, V]
    Array<float> intersect_area = intersect_wh.mult_dim(-1);
    // shape: [13, 13, 3, 1]
    Array<float> pred_box_area = pred_wh.mult_dim(-1);
    // shape: [V]
    Array<float> true_box_area = true_wh.mult_dim(-1);

    // shape: [1, V]
    true_box_area = true_box_area.add_axis(0);

    // shape: [13, 13, 3, V]
    Array<float> iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10f);

    return iou;
}

float YoloCocoDataset::m_eval_xy_loss_on_layer(int64 ns, Dict preds, Dict trues) {
    Array<float> mix_w = preds["mix_w"];
    Array<float> object_mask = trues["object_mask"];

    Array<float> true_xy = trues["box_xy"];
    Array<float> pred_xy = preds["box_xy"];

    Array<float> box_loss_scale = trues["box_loss_scale"];

    int64 mb_size = trues["mb_size"];

    float loss = kmath->sum(kmath->square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / (float) mb_size;

    return loss;
}
float YoloCocoDataset::m_eval_wh_loss_on_layer(int64 ns, Dict preds, Dict trues) {
    Array<float> mix_w = preds["mix_w"];
    Array<float> object_mask = trues["object_mask"];

    Array<float> true_wh = trues["box_wh"];
    Array<float> pred_wh = preds["box_wh"];

    Array<float> box_loss_scale = trues["box_loss_scale"];

    int64 mb_size = trues["mb_size"];

    float loss = kmath->sum(kmath->square(true_wh - pred_wh) * object_mask * box_loss_scale * mix_w) / (float) mb_size;

    return loss;
}

float YoloCocoDataset::m_eval_conf_loss_on_layer(int64 ns, Dict preds, Dict trues) {
    Array<float> mix_w = preds["mix_w"];
    Array<float> object_mask = trues["object_mask"];
    Array<float> ignore_mask = preds["ignore_mask"];

    Array<float> conf_pos_mask = object_mask;
    Array<float> conf_neg_mask = (1.0f - object_mask) * ignore_mask;
        
    Array<float> pred_conf= preds["conf"];

    Array<float> entropy = kmath->sigmoid_cross_entropy_with_logits(object_mask, pred_conf);

    Array<float> conf_loss_pos = conf_pos_mask * entropy;
    Array<float> conf_loss_neg = conf_neg_mask * entropy; 

    // TODO: may need to balance the pos-neg by multiplying some weights
    Array<float> conf_loss = conf_loss_pos + conf_loss_neg;

    if (m_use_focal_loss) {
        float alpha = 1.0;
        float gamma = 2.0;
        // TODO: alpha should be a mask array if needed
        Array<float> focal_mask = kmath->power(kmath->abs(object_mask - kmath->sigmoid(pred_conf)), gamma) * alpha;
        conf_loss *= focal_mask;
    }

    int64 mb_size = trues["mb_size"];
    float loss = kmath->sum(conf_loss * mix_w) / (float) mb_size;
    return loss;
}

float YoloCocoDataset::m_eval_class_loss_on_layer(int64 ns, Dict preds, Dict trues) {
    Array<float> mix_w = preds["mix_w"];
    Array<float> object_mask = trues["object_mask"];

    Array<float> true_class = trues["class"];
    Array<float> pred_class = preds["class"];

    if (m_use_smooth_onehot) {
        float deta = 0.01f;
        true_class = true_class * (1 - deta) + deta / (float) class_num;
    }

    Array<float> full_entropy = kmath->sigmoid_cross_entropy_with_logits(true_class, pred_class);
    Array<float> valid_entropy = object_mask * full_entropy * mix_w;

    int64 mb_size = trues["mb_size"];
    float loss = kmath->sum(valid_entropy) / (float) mb_size;
    return loss;
}
*/

