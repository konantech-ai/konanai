/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <assert.h>

#include "../../datasets/yolo.h"

#include "../cuda_conn.cuh"
#include "../cuda_math.h"
#include "../cuda_note.h"

#include "../../core/array.h"
#include "../../core/func_timer.h"
#include "../../core/log.h"

Dict YoloCocoDataset::m_create_scale_maps(Array<int64> arr_img_idxs, Array<float> mix_ratio) {
    CudaConn cuda("yolo_create_scale_maps", NULL);

    //bool m_trace = false;
    int64 isize = arr_img_idxs.total_size();
    
    int64 img_cnt = 0;
    int64 box_cnt = 0;

    // 미니배치 데이터 전체에 있는 박스 레이블링 정보의 갯수를 조사
    int64* p_img_idx = arr_img_idxs.data_ptr();

    for (int64 n = 0; n < isize; n++) {
        if (p_img_idx[n] < 0) continue;
        img_cnt++;
        List boxes = mp_box_info[p_img_idx[n]];
        for (int64 m = 0; m < (int64)boxes.size(); m++) {
            Dict box_info = boxes[m];
            int64 cat_id = box_info["cat"];
            if (cat_id >= 0 && cat_id < class_num) box_cnt++;
        }
    }

    // 미니배치 데이터 전체에 있는 박스 정보 수집
    Array<int64> arr_img_info(Shape(img_cnt, IMG_INFO_SIZE));     // (ndata, image_id, width, height) for images
    Array<float> arr_box_info(Shape(box_cnt, BOX_INFO_SIZE));   // (nimginfo, cat_id, center_x, center_y, width, height, mixed)

    int64* p_img_info = arr_img_info.data_ptr();
    float* p_box_info = arr_box_info.data_ptr();
    float* p_mix_ratio = mix_ratio.data_ptr();

    for (int64 n = 0, img_idx = 0; n < isize; n++) {
        int64 didx = p_img_idx[n];

        if (didx < 0) continue;

        List boxes = mp_box_info[didx];

        if ((int64)boxes.size() == 0) continue;

        int64 ndata = n / 2;
        int64 image_id = mp_image_ids[didx];
        int64 width = mp_image_widths[didx];
        int64 height = mp_image_heights[didx];
        
        float mixed = (n % 2 == 0) ? p_mix_ratio[ndata] : 1.0f - p_mix_ratio[ndata];

        *p_img_info++ = ndata;
        *p_img_info++ = image_id;
        *p_img_info++ = width;
        *p_img_info++ = height;

        float ratio = (float)image_size / (float)(MAX(width, height));

        int64 dx = (image_size - (int64)((float)width * ratio)) / 2;
        int64 dy = (image_size - (int64)((float)height * ratio)) / 2;

        for (int64 m = 0; m < (int64)boxes.size(); m++) {
            Dict box_info = boxes[m];

            if (m_trace) logger.Print("    box_info[%lld: %lldx%lld]: %s", n, width, height, Value::description(box_info).c_str());

            int64 cat_id = box_info["cat"];

            if (cat_id < 0 || cat_id >= class_num) continue;

            List bbox = box_info["box"]; // (left, top, width, hight) in original image_size

            *p_box_info++ = (float)(img_idx);
            *p_box_info++ = (float)cat_id;

            *p_box_info++ = ((float)bbox[0] * ratio + (float)dx) + ((float)bbox[2] * ratio) / 2.0f;  // center_x = (left + 2 / width) in (416x416) scale
            *p_box_info++ = ((float)bbox[1] * ratio + (float)dy) + ((float)bbox[3] * ratio) / 2.0f;  // center_y = (right + 2 / height) in (416x416) scale
            *p_box_info++ = (float)bbox[2] * ratio;  // width in (416x416) scale
            *p_box_info++ = (float)bbox[3] * ratio;  // height in (416x416) scale
            
            if (m_trace) logger.Print("    box_info ==> [%f, %f, %f, %f]", p_box_info[-4], p_box_info[-3], p_box_info[-2], p_box_info[-1]);
            
            *p_box_info++ = mixed;
        }

        if (m_trace) logger.Print("");

        img_idx++;
    }

    if (m_trace) arr_img_info.print("arr_img_info");

    assert(p_box_info - arr_box_info.data_ptr() == arr_box_info.total_size());

    // (스케일, 앵커) 조합별로 각 박스의 iou 계산
    Shape sshape(box_cnt, num_scales, anchor_per_scale);

    float* cuda_score = cuda.alloc_float_mem(sshape, "score");
    float* cuda_selected = cuda.alloc_float_mem(sshape, "selected"); 

    int64* cuda_img_info = cuda.copy_to_buffer(arr_img_info, "img_info");
    float* cuda_box_info = cuda.copy_to_buffer(arr_box_info, "box_info");
    int64* cuda_selected_cnt = cuda.alloc_int64_mem(Shape(num_scales), "selected_cnt");

    int64* cuda_anchors = m_anchor_size.data_ptr();

    int64 ssize = sshape.total_size();
    int64 bsize = box_cnt;

    cu_call(ker_yolo_eval_true_box_score, ssize, (ssize, cuda_score, cuda_box_info, cuda_anchors, num_scales, anchor_per_scale));
    cu_call(ker_yolo_eval_true_box_select, bsize, (bsize, cuda_selected, cuda_score, num_scales, anchor_per_scale));
    cu_call(ker_yolo_eval_true_count_selected, num_scales, (num_scales, cuda_selected_cnt, cuda_selected, ssize, num_scales, anchor_per_scale));

    if (m_trace) {
        cuda.DumpArr(cuda_box_info, "box_info", Shape(), true);
        //cuda.DumpArr(cuda_score, "cuda_score", Shape(), true);
        //cuda.DumpArr(cuda_selected, "cuda_selected", Shape(), true);
        cuda.DumpArr(cuda_selected_cnt, "cuda_selected_cnt", Shape(), true);

        cuda.DumpArr(cuda_img_info, "cuda_img_info", Shape(), true);
        cuda.DumpArr(cuda_box_info, "cuda_box_info", Shape(), true);

        //cuda.DumpArr(cuda_score, "cuda_score", Shape(), true);
        //cuda.DumpArr(cuda_selected, "cuda_selected", Shape(), true);
        cuda.DumpArr(cuda_selected_cnt, "cuda_selected_cnt", Shape(), true);
    }

    Dict trues;
    
    trues["img_info"] = cuda.detach(cuda_img_info, "img_info");
    trues["box_info"] = cuda.detach(cuda_box_info, "box_info");

    for (int64 ns = 0; ns < num_scales; ns++) {
        string key_name = "boxes_" + to_string(ns + 1);

        int64 scale_box_cnt = cuda.get_nth_element(cuda_selected_cnt, ns);

        if (scale_box_cnt > 0) {
            int64* cuda_box_scale = cuda.alloc_int64_mem(Shape(scale_box_cnt, SCALE_BOX_SIZE), "cuda_box_label_info");

            cu_call(ker_yolo_eval_true_lookup_scale_box, 1, (1, cuda_box_scale, cuda_selected, ns, ssize, num_scales, anchor_per_scale));
            cu_call(ker_yolo_eval_true_eval_box_cood, scale_box_cnt, (scale_box_cnt, cuda_box_scale, cuda_box_info, grid_size[ns]));

            //cuda.DumpArr(cuda_box_scale, "box_scale", Shape(), true);

            trues[key_name] = cuda.detach(cuda_box_scale, "box_scale");
        }
    }

    return trues;
}

Dict YoloCocoDataset::m_extract_estimate(int64 nscale, Array<float> fmap, bool trace) {
    CudaConn cuda("yolo_conv_layer_estimate", NULL);

    m_trace = false;

    Shape pshape = fmap.shape().remove_end().append(3).append(nvec_pbox);

    float* cuda_fmap = fmap.data_ptr();
    float* cuda_pred = cuda.alloc_float_mem(pshape, "pred");

    int64* cuda_anchors = m_anchor_size.data_ptr() + nscale * anchor_per_scale * 2;

    int64 msize = fmap.total_size() / nvec_pbox;

    cu_call(ker_yolo_conv_fmap, msize, (msize, cuda_pred, cuda_fmap, cuda_anchors, image_size, grid_cnts[nscale], anchor_per_scale, class_num));

    Array<float> pred = cuda.detach(cuda_pred, "pred");
    Dict est = Value::wrap_dict("pred", pred);
    return est;
}

Dict YoloCocoDataset::m_evaluate_layer_loss(int64 nscale, Array<float> fmap, Array<int64> img_info, Array<float> box_info, Array<int64> scale_boxes, bool trace) {
    CudaConn cuda("yolo_evaluate_layer_loss", NULL);

    int64 mb_size = fmap.axis_size(0);
    int64 box_cnt = scale_boxes.axis_size(0);

    //logger.Print("fmap.shape = %s", fmap.shape().desc().c_str());

    Shape mshape = fmap.shape().replace_end(anchor_per_scale);
    Shape ushape = mshape.append(box_cnt);
    Shape fshape = fmap.shape();

    int64 msize = mshape.total_size();
    int64 usize = ushape.total_size();
    int64 fsize = fshape.total_size();

    if (m_trace) {
        logger.Print("fmap.shape(): %s", fmap.shape().desc().c_str());
        logger.Print("img_info.shape(): %s", img_info.shape().desc().c_str());
        logger.Print("box_info.shape(): %s", box_info.shape().desc().c_str());
        logger.Print("scale_boxes.shape(): %s", scale_boxes.shape().desc().c_str());
        logger.Print("mshape: %s", mshape.desc().c_str());
        logger.Print("ushape: %s", ushape.desc().c_str());
    }

    float* cuda_fmap = fmap.data_ptr();
    float* cuda_box_info = box_info.data_ptr();

    int64* cuda_img_info = img_info.data_ptr();
    int64* cuda_scale_boxes = scale_boxes.data_ptr();
    int64* cuda_anchors = m_anchor_size.data_ptr() + nscale * anchor_per_scale * 2;

    int64* cuda_iou_box = cuda.alloc_int64_mem(ushape, "iou_box");
    int64* cuda_best_box = cuda.alloc_int64_mem(mshape, "best_box");

    float* cuda_iou = cuda.alloc_float_mem(ushape, "iou");
    float* cuda_best_iou = cuda.alloc_float_mem(mshape, "best_iou");
    //float* cuda_loss = cuda.alloc_float_mem(mshape, "loss");
    float* cuda_losses = cuda.alloc_float_mem(fshape, "losses");

    if (m_trace || trace) cuda.DumpArr(cuda_scale_boxes, "scale_boxes", Shape(), true);

    cu_call(ker_yolo_eval_iou, usize, (usize, cuda_iou, cuda_iou_box, cuda_fmap, cuda_img_info, cuda_box_info, cuda_scale_boxes, cuda_anchors,
        anchor_per_scale, image_size, grid_cnts[nscale], box_cnt));
    cu_call(ker_yolo_select_best_iou, msize, (msize, cuda_best_box, cuda_best_iou, cuda_iou, cuda_iou_box, box_cnt));

    /*
    int64 idxs[100];

    if (m_trace || trace) {
        int64 valid_cnt = cuda.DumpSparse(cuda_iou, "iou_sparse", idxs, 100, 100);
        int64 dump_cnt = (valid_cnt < 100) ? valid_cnt : 100;

        logger.Print("valid_cnt for iou_sparse = %lld", valid_cnt);

        cuda.DumpSparse(cuda_iou_box, "iou_box_sparse", NULL, 100);

        //cuda.Print_selected_rows(cuda_pred, "cuda_pred", idxs, dump_cnt, 85);

        //cuda.Print_rows(cuda_iou, "iou", 0, 3, box_cnt);
        //cuda.Print_rows(cuda_iou_box, "iou_box", 0, 3, box_cnt);

        valid_cnt = cuda.DumpSparse(cuda_best_iou, "best_iou", idxs, 100, 100, 1);
        dump_cnt = (valid_cnt < 100) ? valid_cnt : 100;

        logger.Print("valid_cnt for best_iou = %lld", valid_cnt);

        cuda.Print_selected_rows(cuda_best_box, "best_box", idxs, dump_cnt, 1);

        //cuda.DumpSparse(cuda_best_box, "best_box");
        //cuda.Print_rows(cuda_best_iou, "best_iou", 0, 3, 1);
        //cuda.Print_rows(cuda_best_box, "best_box", 0, 3, 1);
    }

    float* cuda_fmap = fmap.data_ptr();

    cu_call(ker_yolo_loss_cood, msize, (msize, cuda_loss, cuda_fmap, cuda_box_info, cuda_best_box, image_size, anchor_per_scale, grid_cnts[nscale]));

    trace = true;

    if (trace || m_trace) {
        int64 valid_cnt = cuda.DumpSparse(cuda_loss, "cuda_loss", idxs, 100, 100, 1);
        logger.Print("valid_cnt for cuda_loss = %lld", valid_cnt);
    }

    for (int64 range = ADD_RANGE, ssize = msize; ssize > 1; range *= ADD_RANGE) {
        ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
        cu_call(ker_sum, ssize, (ssize, cuda_loss, msize, range));
        if (0 && (m_trace || trace)) cuda.DumpSparse(cuda_loss, "cuda_loss-sum loop", NULL, 100, 0, 1);
    }

    float loss_coods = cuda.get_nth_element(cuda_loss, 0) / (float)mb_size;

    if (m_trace || trace) logger.Print("loss_coods = %f", loss_coods);

    int64* cuda_anchors = m_anchor_size.data_ptr() + nscale * anchor_per_scale * 2;

    cu_call(ker_yolo_loss_size, msize, (msize, cuda_loss, cuda_fmap, cuda_box_info, cuda_best_box, cuda_anchors, image_size, anchor_per_scale, grid_cnts[nscale]));

    if (0 && (m_trace || trace) && nscale == 2) {
        //int64 valid_cnt = cuda.DumpSparse(cuda_loss, "size_loss", idxs, 100, 100, 1);
        //logger.Print("valid_cnt for size_loss = %lld", valid_cnt);
    }

    for (int64 range = ADD_RANGE, ssize = msize; ssize > 1; range *= ADD_RANGE) {
        ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
        cu_call(ker_sum, ssize, (ssize, cuda_loss, msize, range));
        if (0 && (m_trace || trace)) cuda.DumpSparse(cuda_loss, "size_loss-sum loop", NULL, 100, 0, 1);
    }

    float loss_sizes = cuda.get_nth_element(cuda_loss, 0) / (float)mb_size;

    if (m_trace || trace) logger.Print("loss_sizes = %f", loss_sizes);

    cu_call(ker_yolo_loss_conf, msize, (msize, cuda_loss, cuda_fmap, cuda_box_info, cuda_best_box, cuda_best_iou, m_use_focal_loss));

    if (0 && (m_trace || trace)) {
        int64 valid_cnt = cuda.DumpSparse(cuda_loss, "confs_loss", idxs, 20, 20, 1);
        logger.Print("valid_cnt for confe_loss = %lld", valid_cnt);
        int64 zero_cnt = cuda.DumpZeros(cuda_loss, "confs_loss", NULL, 0, 0, 1);
        logger.Print("zero_cnt for confe_loss = %lld", zero_cnt);
    }

    for (int64 range = ADD_RANGE, ssize = msize; ssize > 1; range *= ADD_RANGE) {
        ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
        cu_call(ker_sum, ssize, (ssize, cuda_loss, msize, range));
        if (0 && (m_trace || trace)) cuda.DumpSparse(cuda_loss, "conf_loss-sum loop", NULL, 20, 0, 1);
    }

    float loss_confs = cuda.get_nth_element(cuda_loss, 0) / (float)mb_size;

    if (m_trace || trace) logger.Print("loss_confs = %f", loss_confs);

    cu_call(ker_yolo_loss_class, msize, (msize, cuda_loss, cuda_fmap, cuda_box_info, cuda_best_box, class_num, m_use_smooth_onehot));

    if (0 && (m_trace || trace)) {
        int64 valid_cnt = cuda.DumpSparse(cuda_loss, "class_loss", idxs, 100, 100, 1);
        logger.Print("valid_cnt for size_loss = %lld", valid_cnt);
    }

    for (int64 range = ADD_RANGE, ssize = msize; ssize > 1; range *= ADD_RANGE) {
        ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
        cu_call(ker_sum, ssize, (ssize, cuda_loss, msize, range));
        if (0 && (m_trace || trace)) cuda.DumpSparse(cuda_loss, "class_loss-sum loop", NULL, 100, 0, 1);
    }

    float loss_probs = cuda.get_nth_element(cuda_loss, 0) / (float)mb_size;

    if (m_trace || trace) logger.Print("loss_probs = %f", loss_probs);
    */

    cu_call(ker_yolo_eval_losses, fsize, (fsize, cuda_losses, cuda_fmap, cuda_box_info, cuda_best_box, cuda_best_iou, cuda_anchors,
        mb_size, image_size, grid_cnts[nscale], anchor_per_scale, class_num, m_use_focal_loss, m_use_smooth_onehot));

    if (0 && (trace || m_trace)) {
        int64 idxs[100];
        int64 valid_cnt = cuda.DumpSparse(cuda_losses, "cuda_losses", idxs, 100, 100, 1);
        logger.Print("valid_cnt for cuda_losses = %lld", valid_cnt);
    }

    //cuda.DumpArr(cuda_losses, "cuda_losses");

    for (int64 range = ADD_RANGE, ssize = msize; ssize > 1; range *= ADD_RANGE) {
        ssize = (ssize + ADD_RANGE - 1) / ADD_RANGE;
        int64 vsize = ssize * nvec_pbox;
        cu_call(ker_sum_rows, vsize, (vsize, cuda_losses, msize, range, nvec_pbox));
        //if (m_trace || trace) cuda.DumpSparse(cuda_losses, "class_losses-sum loop", NULL, 100, 0, 1);
    }

    float* loss_sums = new float[nvec_pbox];

    cuda.get_nth_row(loss_sums, cuda_losses, 0, nvec_pbox);

    float loss_coods = loss_sums[0] + loss_sums[1];
    float loss_sizes = loss_sums[2] + loss_sums[3];
    float loss_confs = loss_sums[4];
    float loss_probs = 0;
    
    for (int64 n = 0; n < class_num; n++) {
        loss_probs += loss_sums[n + 5];
    }

    delete[] loss_sums;

    if (0 && (trace || m_trace)) {
        logger.Print("loss_coods = %f", loss_coods);
        logger.Print("loss_sizes = %f", loss_sizes);
        logger.Print("loss_confs = %f", loss_confs);
        logger.Print("loss_probs = %f", loss_probs);
    }

    Dict loss;

    loss["coods"] = loss_coods;
    loss["sizes"] = loss_sizes;
    loss["confs"] = loss_confs;
    loss["probs"] = loss_probs;

    return loss;
}

Dict YoloCocoDataset::m_evaluate_layer_grad(int64 nscale, Array<float> fmap, Array<int64> img_info, Array<float> box_info, Array<int64> scale_boxes, bool trace) {
    CudaConn cuda("yolo_evaluate_layer_grad", NULL);

    int64 mb_size = fmap.axis_size(0);
    int64 box_cnt = scale_boxes.axis_size(0);

    //logger.Print("fmap.shape = %s", fmap.shape().desc().c_str());

    Shape mshape = fmap.shape().replace_end(anchor_per_scale);
    Shape ushape = mshape.append(box_cnt);
    Shape fshape = fmap.shape();

    int64 msize = mshape.total_size();
    int64 usize = ushape.total_size();
    int64 fsize = fshape.total_size();

    if (m_trace) {
        logger.Print("fmap.shape(): %s", fmap.shape().desc().c_str());
        logger.Print("img_info.shape(): %s", img_info.shape().desc().c_str());
        logger.Print("box_info.shape(): %s", box_info.shape().desc().c_str());
        logger.Print("scale_boxes.shape(): %s", scale_boxes.shape().desc().c_str());
        logger.Print("mshape: %s", mshape.desc().c_str());
        logger.Print("ushape: %s", ushape.desc().c_str());
    }

    float* cuda_fmap = fmap.data_ptr();
    float* cuda_box_info = box_info.data_ptr();

    int64* cuda_img_info = img_info.data_ptr();
    int64* cuda_scale_boxes = scale_boxes.data_ptr();
    int64* cuda_anchors = m_anchor_size.data_ptr() + nscale * anchor_per_scale * 2;

    int64* cuda_iou_box = cuda.alloc_int64_mem(ushape, "iou_box");
    int64* cuda_best_box = cuda.alloc_int64_mem(mshape, "best_box");

    float* cuda_iou = cuda.alloc_float_mem(ushape, "iou");
    float* cuda_best_iou = cuda.alloc_float_mem(mshape, "best_iou");
    //float* cuda_loss = cuda.alloc_float_mem(mshape, "loss");
    float* cuda_grads = cuda.alloc_float_mem(fshape, "grads");

    if (m_trace || trace) cuda.DumpArr(cuda_scale_boxes, "scale_boxes", Shape(), true);

    cu_call(ker_yolo_eval_iou, usize, (usize, cuda_iou, cuda_iou_box, cuda_fmap, cuda_img_info, cuda_box_info, cuda_scale_boxes, cuda_anchors,
        anchor_per_scale, image_size, grid_cnts[nscale], box_cnt));
    cu_call(ker_yolo_select_best_iou, msize, (msize, cuda_best_box, cuda_best_iou, cuda_iou, cuda_iou_box, box_cnt));

    cu_call(ker_yolo_eval_grads, fsize, (fsize, cuda_grads, cuda_fmap, cuda_box_info, cuda_best_box, cuda_best_iou, cuda_anchors,
        mb_size, image_size, grid_cnts[nscale], anchor_per_scale, class_num, m_use_focal_loss, m_use_smooth_onehot));

    if (0) {
        int64 idxs[20];
        int64 valid_cnt = cuda.DumpSparse(cuda_grads, "cuda_grads", idxs, 20, 20, 1);
        logger.Print("valid_cnt for cuda_grads = %lld", valid_cnt);
    }

    Array<float> G_fmap = cuda.detach(cuda_grads, "geads");

    return Value::wrap_dict("data", G_fmap);
}

Dict YoloCocoDataset::m_evaluate_mAP(Dict ys, Dict outs, bool need_detail) {
#ifdef KAI2021_WINDOWS
    throw KaiException(KERR_ASSERT);
    return Dict();
#else
    //hs.cho
    throw KaiException(KERR_ASSERT);
    return Dict();
#endif
#if FALSE
    Dict acc_dic;
    acc_dic["mAP"] = 3.141592f;
    if (m_mode != "map_test") return acc_dic;

    CudaConn cuda("yolo_evaluate_mAP", NULL);

    m_trace = true;

    //  m_comment("conf 값에 따른 소팅 방식으로 재정리 필요");

    // 1. 처리 대상 실제 박스 정보 준비
    //    box_info(nscale, cat 활용), box_rect(사각정보 활용)를 이용
    Array<int64> box_info = ys["box_info"];
    Array<float> box_rect = ys["box_rect"];

    int64* cuda_box_info = box_info.data_ptr();
    float* cuda_box_rect = box_rect.data_ptr();

    int64 tbox_cnt = box_info.axis_size(0);

    if (m_trace && 0) {
        logger.Print("true box cnt = %lld", tbox_cnt);
        cuda.DumpArr(cuda_box_info, "cuda_box_info", Shape(), true);
        cuda.DumpArr(cuda_box_rect, "cuda_box_rect", Shape(), true);
    }

    // 2. 처리 대상 추정 박스 정보 수집 준비: 일괄 처리 위해 스케일 별 맵 정보를 배열에 준비하고 크기 집계
    //    (데이터, 스케일, 좌표, 앵커) 별로 추정이 이루어지기 때문에 전체 피쳐맵 크기를 앵커벡터크기(ancvec_size)로 나눈 크기의 버퍼가 필요하다

    int64 fsize = 0;
    int64 mb_size = 0;
    int64 msizes[num_scales];

    float* cuda_fmaps[num_scales];

    for (int64 nscale = 0; nscale < num_scales; nscale++) {
        string map_name = "feature_map_" + to_string(nscale + 1);
        Dict dict_est = outs[map_name];
        Array<float> fmap_est = dict_est["data"];

        Array<float> temp = cuda.ToHostArray(fmap_est, "temp");
        temp = temp.reshape(Shape(-1, nvec_pbox));
        temp.print_rows("fmap_est", 0, 2);

        cuda_fmaps[nscale] = fmap_est.data_ptr();

        msizes[nscale] = fmap_est.total_size() / nvec_pbox;
        fsize += msizes[nscale];

        mb_size = fmap_est.axis_size(0);
    }

    if (m_trace) {
        logger.Print("fsize = %lld, mb_size = %lld", fsize, mb_size);
    }

    // 3. 처리 대상 추정 박스 정보 수집
    //    (데이터, 스케일, 좌표, 앵커) 별로 확률로 변환(sigmoid)한 conf 값이 pred_box_thr(0.50) 이상인 추정 박스들을 cat 별로 플래그 표시
    //    가변적인 수의 스케일 처리를 위해 루프 처리가 불가피, 
    //    스케일별 루프 처리를 1회로 마치기 위해 피쳐맵에서 (데이터번호, 혹률정보, 사각정보)도 여기에서 함께 수집한다.

    int64 flag_bytes = (class_num + 7) / 8; // category 정보를 비트 플래그로 표시하기 위한 unsigned char 버퍼의 크기

    unsigned char* cuda_flag = cuda.alloc_byte_mem(Shape(fsize, flag_bytes), "flag");
    int64* cuda_ndata = cuda.alloc_int_mem(Shape(fsize, 1), "ndata");
    float* cuda_conf_rects = cuda.alloc_float_mem(Shape(fsize, 5), "conf_rects");

    int64 out_base = 0;

    for (int64 nscale = 0; nscale < num_scales; nscale++) {
        int64 msize = msizes[nscale];
        float* cuda_fmap = cuda_fmaps[nscale];

        int64* cuda_anchors = m_anchor_size.data_ptr() + nscale * anchor_per_scale * 2;

        cu_call(ker_yolo_select_pred_boxes, msize, (msize, cuda_flag, cuda_ndata, cuda_conf_rects, cuda_fmap, cuda_anchors, pred_conf_thr, out_base, image_size, grid_cnts[nscale], anchor_per_scale, class_num));

        out_base += msize;
    }

    if (m_trace) {
        cuda.DumpArr(cuda_flag, "cuda_flag");
        cuda.DumpArr(cuda_ndata, "cuda_ndata");
        cuda.DumpArr(cuda_conf_rects, "cuda_conf_rects");
    }

    throw KaiException(KERR_ASSERT);

    //    1.2. 확률로 변환(sigmoid)한 conf 값이 pred_box_thr(0.50) 이상인 추정 박스들을 cat 별로 분류 수집
    //         - 합격 박스들은 cuda_flag 해당 위치에 1로, 불합격 상자는 0으로 표시된다
    //         - 합격 박스들의 (데이터번호, 분류번호)는 cuda_cats 해당위치에, 렉트 정보는 cuda_rects 해당 위치에 저장된다.
    /*
    Shape fshape(fsize, 1);    // 1 for valid boxes
    Shape cshape(fsize, 2); // (nd, cat_id) for valid boxes
    Shape rshape(fsize, 4); // box_rects for valid boxes

    int64* cuda_cats = cuda.alloc_int_mem(cshape, "cats");
    float* cuda_rects = cuda.alloc_float_mem(rshape, "rects");

    int64 out_base = 0;

    for (int64 nscale = 0; nscale < num_scales; nscale++) {
        int64 msize = msizes[nscale];
        float* cuda_fmap = cuda_fmaps[nscale];
        
        int64* cuda_anchors = m_anchor_size.data_ptr() + nscale * anchor_per_scale * 2;

        cu_call(ker_yolo_select_pred_boxes, msize, (msize, cuda_flag, cuda_cats, cuda_rects, cuda_fmap, cuda_anchors, pred_conf_thr, out_base, image_size, grid_cnts[nscale], anchor_per_scale, class_num));

        out_base += msize;
    }

    // 2. (실제박스, 추정박스) 쌍들에 대해 iou 값 계산
    //    2.1. 1단계에서 선택된 추정 박스들만을 압축 추출하여 처리 대상으로 삼음
    //         - cuda_flag 값이 1인 상자들의 위치 정보들을 cuda_fidxs 배열에 압축 저장
    //    2.2. 데이터 번호와 분류카테고리 번호가 일치하는 순서쌍에 한해 iou 계산
    //         - 유효한 쌍 여부가 cuda_pair_info[0]에 유효하면 1, 아니면 0으로 저장된다
    //         - 유효 여부와 관계 없이 추정된 cat_id가 cuda_pair_info[1]에 저장된다
    //         - 유효한 쌍의 경우 iou 값이 cuda_ious에 저장된다
    */

    /*
    Array<int64> fidxs = ms_cuda_compress_flags(cuda_flag, fsize, 1);
    int64* cuda_fidxs = cuda.attach_int(fidxs, "fidxs");
    
    int64 pbox_cnt = fidxs.axis_size(0);

    Shape pshape(pbox_cnt, tbox_cnt, 2); // (flag, cat_id) for matched pair
    Shape ishape(pbox_cnt, tbox_cnt, 1); // (iou) for matched pair

    int64 isize = ishape.total_size();

    int64* cuda_pair_info = cuda.alloc_int_mem(pshape, "pair_info");
    float* cuda_ious = cuda.alloc_float_mem(ishape, "ious");

    cu_call(ker_yolo_eval_box_pair_ious, isize, (isize, cuda_pair_info, cuda_ious, cuda_fidxs, cuda_cats, cuda_rects, cuda_box_info, cuda_box_rect, tbox_cnt));

    // 3. 박스 쌍들에 대한 iou 값에 따라 임계값에 따른 카테고리 및 데이터별 precision 정보 집계
    //    3.1. 클래스 별로 정답 속의 상자 수를 cuda_tbox_cnt 배열에 집계한다.
    //    3.2. 클래스 별로 추정 속의 상자 수를 cuda_pbox_cnt 배열에 집계한다.
    //    3.3. (클래스, 임계_iou값) 별로 유효한 순서쌍이면서 iou 값이 임게값 이상인 상자쌍 수를 cuda_match_cnt 배열에 집계한다.
    //    3.4. 세 가지 count 정보로부터 (클래스, 임계_iou값) 별로 precision과 recall을 게산한다.

    int64 iou_thr_cnt = 10;
    float iou_thr_from = 0.5;
    float iou_thr_step = 0.05;

    Shape cshape(class_num);
    Shape qshape(class_num, iou_thr_cnt);

    int64* cuda_tbox_cnt = cuda.alloc_int_mem(cshape, "tbox_cnt");
    int64* cuda_pbox_cnt = cuda.alloc_int_mem(cshape, "cbox_cnt");
    int64* cuda_match_cnt = cuda.alloc_int_mem(qshape, "match_cnt");

    float* cuda_precision = cuda.alloc_float_mem(qshape, "precision");
    float* cuda_recall = cuda.alloc_float_mem(qshape, "recall");

    int64 csize = cshape.total_size();
    int64 qsize = qshape.total_size();

    cu_call(ker_yolo_count_true_boxes, csize, (csize, cuda_tbox_cnt, cuda_box_info, tbox_cnt));
    cu_call(ker_yolo_count_pred_boxes, csize, (csize, cuda_pbox_cnt, cuda_pair_info, pbox_cnt, tbox_cnt));

    cu_call(ker_yolo_count_matched_box_pairs, qsize, (qsize, cuda_match_cnt, cuda_pair_info, cuda_ious, iou_thr_cnt, iou_thr_from, iou_thr_step, pbox_cnt, tbox_cnt));

    float* cuda_recall = cuda.alloc_float_mem(qshape, "recall");
    cu_call(ker_yolo_eval_prec_recall, qsize, (qsize, cuda_precision, cuda_recall, cuda_tbox_cnt, cuda_pbox_cnt, cuda_match_cnt, iou_thr_cnt));

    // 3. 박스 쌍들에 대한 iou 값에 따라 임계값에 따른 카테고리 및 데이터별 precision 정보 집계
    //    3.1. 클래스 별로 정답 속의 상자 수를 cuda_tbox_cnt 배열에 집계한다.
    //    3.2. 클래스 별로 추정 속의 상자 수를 cuda_pbox_cnt 배열에 집계한다.
    //    3.3. (클래스, 임계_iou값) 별로 유효한 순서쌍이면서 iou 값이 임게값 이상인 상자쌍 수를 cuda_match_cnt 배열에 집계한다.
    //    3.4. 세 가지 count 정보로부터 (클래스, 임계_iou값) 별로 precision과 recall을 게산한다.

    //int64* cuda_cats = cuda.alloc_int_mem(cshape, "cats");

    // 2. iou_thr_range: 0.05~0.95, 0.10 간격

    // 3. cat 정보 별로 실제 박스와 추정 박스들을 비교 처리
    //    3.1. (실제박스, 추정박스) 쌍들에 대해 iou 값 계산
    //    3.2. iou_thr in iou_thr_range 별로 통게정보 집계
    //         3.2.1. 추정박스 별로 iou 최대치가 0 이상 여부에 따라 tp, fp, fn 집계
    //         3.2.2. 이 때 겹침 판정된 실제박스는 다른 추정박스에 중복 겹침 반영 안되도록 처리 
    //         3.2.3. freqs[iou_thr]['fn'] 형태의 정수값 정보들이 얻어진다.
    //    3.3. 박스별로 얻어진 정보들을 (iou_thr, case) 별로 모두 합산한다. 

    // 4. iou_thr 별로 precision, recall 값을 구한다.
    // 5. recall_level(0.0~1.0, 0.1 간격 11단뎨) 별로 prec 계산해 수집
    //    5.1. recall 값이 recall_level 이상인 항들 중 precision 최대값을 구함
    // 6. avg_prec 값을 게산: 수집된 prec 값들의 평ㄱㅍㄴ

    return 3.141592f;
    */

    /*
    Dict losses;

    if (m_trace) Value::print_dict_keys(ys, "ys");
    if (m_trace) Value::print_dict_keys(outs, "outs");

    Array<int64> box_info = ys["box_info"];
    Array<float> box_rect = ys["box_rect"];

    if (m_trace) box_info.print("box_info");
    if (m_trace) box_rect.print("box_rect");

    for (int64 nscale = 0; nscale < num_scales; nscale++) {
        string key_name = "boxes_" + to_string(nscale + 1);

        if (ys.find(key_name) == ys.end()) continue; // 해당 scale에 걸친 박스 데이터가 전무한 경우

        string map_name = "feature_map_" + to_string(nscale + 1);
        Dict dict_est = outs[map_name];
        Array<float> fmap_est = dict_est["data"];
        Array<int64> scale_boxes = ys[key_name];

        if (m_trace) fmap_est.print("fmap_est");
        if (m_trace) scale_boxes.print("scale_boxes");

        Dict est = m_conv_layer_estimate(nscale, fmap_est);
        Dict loss = m_evaluate_layer_loss(nscale, est, box_info, box_rect, scale_boxes);

        Value::dict_accumulate(losses, loss);
    }

    Dict loss_mean = Value::dict_mean_reset(losses);

    logger.Bookeep("loss_mean = %s\n", Value::description(loss_mean).c_str());

    return loss_mean;
    */

    /*
        if output is None:
            output, _ = model.forward_neuralnet(x)

        est_grids = self.get_estimate(model, output)

        #print('y[2][0].shape', y[2][0].shape)
        assert len(y[2][0].shape) == 5

        box_map = self.get_object_boxes(y[2], est_grids)

        iou_thr_range = np.arange(0.05,1.00,0.10)

        #print('iou_thr_range', iou_thr_range)
        #print('box_map.keys()', box_map.keys())

        prec_at_rec = []

        for cat in box_map:
            #print('cat', cat)
            cat_name = self.target_names[cat]

            freqs = {}
            for iou_thr in iou_thr_range:
                freqs[iou_thr] = {'tp':0, 'fp':0, 'fn':0}

            for nid in box_map[cat]:
                binfo = box_map[cat][nid]
                true_boxes = []
                pred_boxes = []
                if 'true' in binfo: true_boxes = binfo['true']
                if 'pred' in binfo: pred_boxes = binfo['pred']
                #print("{}-{}: true {}, pred {}".format(cat_name, nid, len(true_boxes), len(pred_boxes)))

                freq = self.eval_ap_on_ious(true_boxes, pred_boxes, iou_thr_range)
                for iou_thr in freqs:
                    for case in freqs[iou_thr]:
                        freqs[iou_thr][case] += freq[iou_thr][case]

            recalls, precisions = [], []

            for iou_thr in freqs:
                precision, recall = 0.0, 0.0
                sum1 = freqs[iou_thr]['tp'] + freqs[iou_thr]['fp']
                sum2 = freqs[iou_thr]['tp'] + freqs[iou_thr]['fn']
                if sum1 > 0: precision = freqs[iou_thr]['tp'] / sum1
                if sum2 > 0: recall    = freqs[iou_thr]['tp'] / sum2
                #if freqs[iou_thr]['tp'] > 0:
                #    print('cat, iou_thr, precision, recall', cat, iou_thr, precision, recall)
                recalls.append(recall)
                precisions.append(precision)

            for recall_level in np.linspace(0.0, 1.0, 11):
                try:
                    #print('recalls', recalls)
                    #print('recall_level', recall_level)
                    idxs = np.argwhere(recalls >= recall_level).flatten()
                    #print('idxs', idxs)
                    #print('precisions', precisions)
                    prec = max(np.asarray(precisions)[idxs])
                    #print('prec', prec)
                except ValueError:
                    prec = 0.0
                prec_at_rec.append(prec)

        if len(prec_at_rec) == 0: return 0

        avg_prec = np.mean(prec_at_rec)
        #print('avg_prec', avg_prec)

        return avg_prec
    */

    Dict acc;
    acc["mAP"] = 0.3141592f;
    return acc;
#endif
}

Dict YoloCocoDataset::m_predict(Dict outs) {
#ifdef KAI2021_WINDOWS
    throw KaiException(KERR_ASSERT);
    return Dict();
#else
    CudaConn cuda("predict", NULL);

    bool trace = false;

    int64 acc_count = 0;
    float* cuda_boxes = NULL;

    for (int64 nscale = 0; nscale < num_scales; nscale++) {
        string map_name = "feature_map_" + to_string(nscale + 1);
        Dict dict_est = outs[map_name];
        Array<float> fmap_est = dict_est["data"];

        Dict est = m_extract_estimate(nscale, fmap_est, trace);

        Array<float> pred = est["pred"];

        Shape sshape = pred.shape().remove_end().append(anchor_per_scale).append(class_num);

        float* cuda_pred = pred.data_ptr();
        float* cuda_score = cuda.alloc_float_mem(sshape, "score");

        int64 ssize = sshape.total_size();

        cu_call(ker_yolo_eval_predict_score, ssize, (ssize, cuda_score, cuda_pred, class_num, score_thresh));

        int64* idxs = new int64[ssize];
        int64 cnt = cuda.DumpSparse(cuda_score, "score", idxs, 0, ssize, 1);

        if (cnt > 0) {
            int64* cuda_idxs = cuda.alloc_int64_mem(idxs, cnt, "idxs");
            acc_count += cnt;
            Shape bshape = Shape(acc_count, RES_BOX_SIZE);

            float* old_boxes = cuda_boxes;
            
            cuda_boxes = cuda.alloc_float_mem(bshape, "boxes");

            cu_call(ker_yolo_get_boxes, acc_count, (acc_count, cuda_boxes, old_boxes, cuda_idxs, cuda_score, cuda_pred, acc_count-cnt, grid_cnts[nscale], anchor_per_scale, class_num));
        }
    }

    int64 nms_max_boxes = 200;
    float nms_iou_thr = 0.45f;

    cu_call(ker_yolo_non_max_suppression, class_num, (class_num, cuda_boxes, acc_count, nms_max_boxes, nms_iou_thr));

    Array<float> nms_boxes = cuda.detach(cuda_boxes, "nms_boxes");

    //nms_boxes.print("nms_boxes");

    return Value::wrap_dict("nms_boxes", nms_boxes);
#endif
}

