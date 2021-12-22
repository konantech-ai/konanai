/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/corpus.h"
#include "../cuda/cuda_kernels.h"

class YoloCocoDataset : public Dataset {
public:
    YoloCocoDataset(string name, string mode, string datetime="");
    virtual ~YoloCocoDataset();

    //virtual Value get_ext_param(string key);

    virtual void gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys);

    virtual void visualize_main(Dict xs, Dict ys, Dict outs);
    void visualize(Value xs, Value estimates, Value answers);

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return false; }

    virtual Dict forward_postproc(Dict xs, Dict ys, Dict estimate, string mode);
    virtual Dict backprop_postproc(Dict ys, Dict estimate, string mode);
    virtual Dict eval_accuracy(Dict x, Dict y, Dict out, string mode);

    virtual void log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2);
    virtual void log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2);
    virtual void log_test(string name, Dict acc, int64 tm1, int64 tm2);

    void step_debug();

public:
    const int64 num_scales = 3;
    const int64 anchor_per_scale = 3;

    const bool m_use_smooth_onehot = true;
    const bool m_use_focal_loss = true;
    const bool m_use_mixed = true;

    const int64 class_num = 80;
    const int64 nvec_pbox = PRED_SIZE;
    const int64 nvec_ancs = anchor_per_scale * nvec_pbox;

    const int64 image_size = 416;
    const int64 image_depth = 3;
    const int64 grid_cnts[3] = { 13, 26, 52 };
    const int64 grid_size[3] = { 32, 16,  8 };

    const int64 anchor_size[3][3][2] = {
        {{116,90}, {156,198}, {373,326}},
        {{30,61}, {62,45}, {59,119}},
        {{10,13}, {16,30}, {33,23}}
    };

    const float pred_conf_thr = 0.5f;
    const float score_thresh = 0.3f;

protected:
    string m_mode;

    int64 m_target_num;
    int64 m_image_num;

    map<int64, int64> m_target_id_map;
    map<int64, int64> m_image_id_map;

    int64* mp_target_ids;
    int64* mp_image_ids;
    int64* mp_image_widths;
    int64* mp_image_heights;

    Array<int64> m_anchor_size;
    Array<int64> m_grid_cnts;

    string* mp_target_names;
    List* mp_box_info;

    Array<float>* mp_grid_offsets;

    bool m_trace;

protected:
    Array<int64> m_get_image_idxs(int64* data_idxs, int64 size, Array<float> mix_ratio);
    int64 m_seek_image_id_idx(int64 image_id);

    bool m_load_cache(string box_cache_path, bool no_image=false);
    void m_create_box_info(string box_cache_path, Dict& inst_info);

    Array<float> m_load_images(Array<int64> image_idxs, Array<float> mix_ratio);

    Dict m_create_scale_maps(Array<int64> image_idxs, Array<float> mix_ratio);
    //List m_create_boxes(int64* data_idxs, int64 size, Array<float>& xarr);

    Dict m_extract_estimate(int64 nscale, Array<float> fmap, bool trace);

    Dict m_evaluate_layer_loss(int64 nscale, Array<float> fmap, Array<int64> img_info, Array<float> box_info, Array<int64> scale_boxes, bool trace);

    void m_select_scale_anchor_for_box(bool scale_anchor[][3], float width, float height, float ratio);

    Dict m_extract_preds_on_layer(int64 ns, Array<float> fmap, List true_boxes);
    
    Array<float> m_eval_box_iou(Array<float> pred_xy, Array<float> pred_wh, Array<float> true_xy, Array<float> true_wh);

    /*
    float m_eval_xy_loss_on_layer(int64 ns, Dict preds, Dict trues);
    float m_eval_wh_loss_on_layer(int64 ns, Dict preds, Dict trues);
    float m_eval_conf_loss_on_layer(int64 ns, Dict preds, Dict trues);
    float m_eval_class_loss_on_layer(int64 ns, Dict preds, Dict trues);
    */

    Dict m_evaluate_mAP(Dict ys, Dict outs, bool need_detail);

    //Dict m_backprop_layer_loss(int64 nscale, Dict est, Array<int64> box_info, Array<float> box_rect, Array<int64> scale_boxes);
    Dict m_evaluate_layer_grad(int64 nscale, Array<float> fmap, Array<int64> img_info, Array<float> box_info, Array<int64> scale_boxes, bool trace);

    Dict m_predict(Dict outs);
    
    int64 m_extract_mb_size(Dict outs);

    void m_step_debug_forward();
    void m_step_debug_visualize();
};
