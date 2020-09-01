from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import pairwise_iou, Boxes
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.comm import get_world_size

from .utils.loader import build_test_loader
from .utils.sa_utils import sa_setup_project_dir, sa_write_status_lists, sa_format_annotations

from torch.nn import functional as F
import numpy as np
import torch
import copy
import logging
import datetime
import time

def get_gt_data(inputs):

    gt_instances  = inputs['instances']
    gt_boxes = gt_instances.gt_boxes
    gt_classes = gt_instances.gt_classes
    gt_mislabeled = gt_instances.gt_mislabeled

    return gt_boxes, gt_classes, gt_mislabeled

def get_pred_data(inputs):

    pred_instances = inputs["instances"]
    pred_boxes = pred_instances.pred_boxes
    pred_scores = pred_instances.softmax_scores
    pred_boxes.tensor = pred_boxes.tensor.to("cpu")

    return pred_boxes, pred_scores

def match_gt_to_pred(gt_boxes, pred_boxes, iou_thresh):

    match_quality_matrix = pairwise_iou(gt_boxes, pred_boxes)
    matched_vals, matches = match_quality_matrix.max(dim=1)
    valid_gt_ids = matched_vals > iou_thresh 
    matched_gt_ids = np.where(valid_gt_ids)[0]
    matched_pred_ids = matches[valid_gt_ids]
    
    return matched_gt_ids, matched_pred_ids

def match_pred_to_gt(gt_boxes, pred_boxes, iou_thresh):

    match_quality_matrix = pairwise_iou(pred_boxes, gt_boxes)
    matched_vals, matches = match_quality_matrix.max(dim=1)
    valid_pred_ids = matched_vals < iou_thresh
    matched_pred_ids = np.where(valid_pred_ids)[0]
        
    return matched_pred_ids

def label_mismatch_score(gt_one_hot, pred_softmax):
    return torch.norm(pred_softmax - gt_one_hot).pow(2)/2

def detect_mislabeled_annotations_per_image(inputs, model, iou_thresh = 0.3):

    gt_boxes, gt_classes, gt_mislabeled = get_gt_data(inputs[0])
    gt_mismatch_scores = torch.ones(len(gt_boxes))

    with torch.no_grad():
        outputs = model(inputs)
   
    pred_boxes, pred_scores = get_pred_data(outputs[0])
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return gt_mismatch_scores, gt_classes, gt_boxes, gt_mislabeled
   
    matched_gt_ids, matched_pred_ids = match_gt_to_pred(gt_boxes, pred_boxes, iou_thresh)
    
    for gt_idx, pred_idx in zip(matched_gt_ids, matched_pred_ids):
        gt_one_hot = F.one_hot(gt_classes[gt_idx], num_classes = 20)
        pred_softmax = pred_scores[pred_idx, :].to("cpu")
        gt_mismatch_scores[gt_idx] = label_mismatch_score(gt_one_hot, pred_softmax)
    
    return gt_mismatch_scores, gt_classes, gt_boxes, gt_mislabeled

def detect_mislabeled_annotations(dataset_name, cfg, mismatch_thresh = 0.3):

    class_names = MetadataCatalog.get(dataset_name).thing_classes
    sa_json_dir = sa_setup_project_dir(dataset_name, class_names)
    qa = [] 
    completed = []
    
    data_loader = build_test_loader(cfg, dataset_name)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    n = len(data_loader)
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info(
        "Start mislabel detection on {} images".format(n)
    )
    
    num_warmup = min(5, n - 1)
    start_time = time.perf_counter()
    total_compute_time = 0


    for idx, inputs in enumerate(data_loader):
        
        if idx == num_warmup:
            start_time = time.perf_counter()
            total_compute_time = 0

        start_compute_time = time.perf_counter()

        gt_mismatch_scores, gt_classes, gt_boxes, _ = detect_mislabeled_annotations_per_image(inputs, model)
        mislabeled_gt_ids = gt_mismatch_scores > mismatch_thresh
       
        gt_class_info = [(class_id.item(), class_names[class_id]) for class_id in gt_classes]
        
        if torch.any(mislabeled_gt_ids):
            qa.append(inputs[0]["file_name"])
        else:
            completed.append(inputs[0]["file_name"])
        
        sa_format_annotations(inputs[0]['image_id'], sa_json_dir, gt_boxes, gt_class_info, mislabeled_gt_ids)
     
        total_compute_time += time.perf_counter() - start_compute_time
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_img = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_img > 5:
            total_seconds_per_img = (
                time.perf_counter() - start_time
            ) / iters_after_start
            eta = datetime.timedelta(
                seconds=int(total_seconds_per_img * (n - idx - 1))
            )
            log_every_n_seconds(
                logging.INFO,
                "Proessed {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, n, seconds_per_img, str(eta)
                ),
                n=5,
            )
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    
    logger.info(
        "Total detection time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (n - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total detection pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (n - num_warmup), num_devices
        )
    )

    sa_write_status_lists(dataset_name, qa, completed)

def eval_mislabel_detection(dataset_name, cfg, mismatch_thresh = 0.3, augment =False):

    data_loader = build_test_loader(cfg, dataset_name)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    n = len(data_loader)
    
    tp = torch.zeros(n)
    fp = torch.zeros(n)

    total = torch.zeros(n)
    npos = torch.zeros(n)

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info(
        "Start mislabel evaluation on {} images".format(n)
    )
    
    num_warmup = min(5, n - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    for idx, inputs in enumerate(data_loader):
        if idx == num_warmup:
            start_time = time.perf_counter()
            total_compute_time = 0

        start_compute_time = time.perf_counter()
        
        gt_mismatch_scores, _, _, gt_mislabeled_ids = detect_mislabeled_annotations_per_image(inputs, model)
        pred_mislabeled_ids = gt_mismatch_scores > mismatch_thresh

        total[idx] = gt_mismatch_scores.shape[0]
        npos[idx] = torch.sum(gt_mislabeled_ids).int()

        tp[idx] = torch.sum(torch.logical_and(gt_mislabeled_ids, pred_mislabeled_ids)).int()
        fp[idx] = torch.sum(torch.logical_and(torch.logical_not(gt_mislabeled_ids), pred_mislabeled_ids)).int()
        
        total_compute_time += time.perf_counter() - start_compute_time
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_img = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_img > 5:
            total_seconds_per_img = (
                time.perf_counter() - start_time
            ) / iters_after_start
            eta = datetime.timedelta(
                seconds=int(total_seconds_per_img * (n - idx - 1))
            )
            log_every_n_seconds(
                logging.INFO,
                "Processed {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, n, seconds_per_img, str(eta)
                ),
                n=5,
            )
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    
    logger.info(
        "Total evaluation time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (n - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total evaluation pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (n - num_warmup), num_devices
        )
    )
            
    # recall = torch.sum(tp)/ torch.sum(npos)
    # precision = 1 - (torch.sum(tp) + torch.sum(fp))/torch.sum(total)

    # return recall.item(), precision.item()
    return torch.sum(tp).item(), torch.sum(fp).item(), torch.sum(npos).item(), torch.sum(total).item()

def detect_missing_annotations_per_image(inputs, model, iou_thresh = 0.2):
    gt_boxes, gt_classes, _ = get_gt_data(inputs[0])

    with torch.no_grad():
        outputs = model(inputs)

    pred_boxes, pred_scores = get_pred_data(outputs[0])
    
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.empty((0,)), torch.empty((0,)), torch.empty((0,))

    matched_pred_ids = match_pred_to_gt(gt_boxes, pred_boxes, iou_thresh)
    miss_candidate_boxes = pred_boxes[matched_pred_ids]
    miss_candidate_scores = pred_scores[matched_pred_ids, :]
    

    if len(miss_candidate_boxes) == 0:
        return torch.empty((0,)), torch.empty((0,)), torch.empty((0,))

    miss_label_scores, miss_class_ids = torch.max(miss_candidate_scores, dim = 1)
    
    return miss_label_scores, miss_candidate_boxes, miss_class_ids

def detect_missing_annotations(dataset_name, cfg, skip_thresh = 0.9):
    
    class_names = MetadataCatalog.get(dataset_name).thing_classes
    sa_json_dir = sa_setup_project_dir(dataset_name, class_names)
    qa = [] 
    completed = []
    
    data_loader = build_test_loader(cfg, dataset_name)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    n = len(data_loader)
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info(
        "Start skipped annotation detection on {} images".format(n)
    )
    
    num_warmup = min(5, n - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    for idx, inputs in enumerate(data_loader):
        
        if idx == num_warmup:
            start_time = time.perf_counter()
            total_compute_time = 0

        start_compute_time = time.perf_counter()
       
        missed_label_scores, missed_boxes, missed_class_ids = detect_missing_annotations_per_image(inputs, model)
        if len(missed_boxes) == 0:
           completed.append(inputs[0]["file_name"])
           continue 

        skip_ids = missed_label_scores > skip_thresh
        skip_boxes = missed_boxes[skip_ids]
        skip_class_ids = missed_class_ids[skip_ids]

        skip_class_info = [(class_id.item(), class_names[class_id]) for class_id in skip_class_ids]
        if len(skip_boxes) == 0:
            completed.append(inputs[0]["file_name"])
            continue
        else:
            qa.append(inputs[0]["file_name"])
       
        sa_format_annotations(inputs[0]['image_id'], sa_json_dir, skip_boxes, skip_class_info, torch.ones(len(skip_boxes), dtype = torch.bool))
       
        total_compute_time += time.perf_counter() - start_compute_time
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_img = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_img > 5:
            total_seconds_per_img = (
                time.perf_counter() - start_time
            ) / iters_after_start
            eta = datetime.timedelta(
                seconds=int(total_seconds_per_img * (n - idx - 1))
            )
            log_every_n_seconds(
                logging.INFO,
                "Proessed {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, n, seconds_per_img, str(eta)
                ),
                n=5,
            )
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    
    logger.info(
        "Total skip detection time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (n - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total skip detection pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (n - num_warmup), num_devices
        )
    )

    sa_write_status_lists(dataset_name, qa, completed)

def detect_risky_annotations(dataset_name, cfg, mismatch_thresh = 0.3, skip_thresh = 0.9):
    
    class_names = MetadataCatalog.get(dataset_name).thing_classes
    sa_json_dir = sa_setup_project_dir(dataset_name, class_names)
    qa = [] 
    completed = []
    
    data_loader = build_test_loader(cfg, dataset_name)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    n = len(data_loader)
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info(
        "Start risky annotations detection on {} images".format(n)
    )
    
    num_warmup = min(5, n - 1)
    start_time = time.perf_counter()
    total_compute_time = 0


    for idx, inputs in enumerate(data_loader):
        
        if idx == num_warmup:
            start_time = time.perf_counter()
            total_compute_time = 0

        start_compute_time = time.perf_counter()
        
        #detect mislabeled annotations
        gt_mismatch_scores, gt_classes, gt_boxes, _ = detect_mislabeled_annotations_per_image(inputs, model)
        mislabeled_gt_ids = gt_mismatch_scores > mismatch_thresh
        gt_class_info = [(class_id.item(), class_names[class_id]) for class_id in gt_classes]

        #detect missing annotations
        missed_label_scores, missed_boxes, missed_class_ids = detect_missing_annotations_per_image(inputs, model)
        if len(missed_boxes):
            skip_ids = missed_label_scores > skip_thresh
            skip_boxes = missed_boxes[skip_ids]
            skip_class_ids = missed_class_ids[skip_ids]
            skip_class_info = [(class_id.item(), class_names[class_id]) for class_id in skip_class_ids]

            gt_boxes = Boxes.cat([gt_boxes, skip_boxes])
            mislabeled_gt_ids = torch.cat((mislabeled_gt_ids, torch.ones(len(skip_boxes), dtype = torch.bool)))
            gt_class_info.extend(skip_class_info)

        if torch.any(mislabeled_gt_ids):
            qa.append(inputs[0]["file_name"])
        else:
            completed.append(inputs[0]["file_name"])

        sa_format_annotations(inputs[0]['image_id'], sa_json_dir, gt_boxes, gt_class_info, mislabeled_gt_ids)
     
        total_compute_time += time.perf_counter() - start_compute_time
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_img = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_img > 5:
            total_seconds_per_img = (
                time.perf_counter() - start_time
            ) / iters_after_start
            eta = datetime.timedelta(
                seconds=int(total_seconds_per_img * (n - idx - 1))
            )
            log_every_n_seconds(
                logging.INFO,
                "Proessed {}/{}. {:.4f} s / img. ETA={}".format(
                    idx + 1, n, seconds_per_img, str(eta)
                ),
                n=5,
            )
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    
    logger.info(
        "Total detection time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (n - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total detection pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (n - num_warmup), num_devices
        )
    )

    sa_write_status_lists(dataset_name, qa, completed)
