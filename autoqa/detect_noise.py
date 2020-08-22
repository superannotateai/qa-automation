from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import pairwise_iou


from .utils.loader import build_test_loader
from .utils.ao_utils import ao_setup_project_dir, ao_format_annotations
from .utils.tensor_aug import add_salt_and_pepper, gaussian_blur, change_contrast

from torch.nn import functional as F
import numpy as np
import torch
import copy

def add_augmentations(inputs):

    orig_image_tensor = inputs['image']
    
    salty_image_tensor = add_salt_and_pepper(orig_image_tensor)
    salty_inputs = copy.deepcopy(inputs)
    salty_inputs['image'] = salty_image_tensor

    blurred_image_tensor = gaussian_blur(orig_image_tensor)
    blurred_inputs = copy.deepcopy(inputs)
    blurred_inputs['image'] = blurred_image_tensor

    contrast_image_tensor = change_contrast(orig_image_tensor)
    contrast_inputs = copy.deepcopy(inputs)
    contrast_inputs['image'] = contrast_image_tensor

    aug_inputs = [inputs, salty_inputs, blurred_inputs, contrast_inputs]
    return aug_inputs

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

def label_mismatch_score(gt_one_hot, pred_softmax):
    return torch.norm(pred_softmax - gt_one_hot).pow(2)/2

def detect_mislabeled_annotations_per_image(inputs, model, augment = False, iou_thresh = 0.3):
    gt_boxes, gt_classes, gt_mislabeled = get_gt_data(inputs[0])
    if augment:
       inputs = add_augmentations(inputs[0])
    
    with torch.no_grad():
        outputs = model(inputs)
    
    gt_label_scores = torch.ones((len(gt_boxes), len(inputs)))

    for aug_idx, predictions in enumerate(outputs):
        pred_boxes, pred_scores = get_pred_data(predictions)
        matched_gt_ids, matched_pred_ids = match_gt_to_pred(gt_boxes, pred_boxes, iou_thresh)
        
        for gt_idx, pred_idx in zip(matched_gt_ids, matched_pred_ids):
            gt_one_hot = F.one_hot(gt_classes[gt_idx], num_classes = 20)
            pred_softmax = pred_scores[pred_idx, :].to("cpu")

            gt_label_scores[gt_idx, aug_idx] = label_mismatch_score(gt_one_hot, pred_softmax)
    
    if augment:
        top2_values, _ = gt_label_scores.topk(2, dim = 1, largest = False)
        return top2_values[:,1], gt_classes, gt_boxes, gt_mislabeled
    return gt_label_scores.flatten(), gt_classes, gt_boxes, gt_mislabeled

def detect_mislabeled_annotations(dataset_name, cfg, mismatch_thresh = 0.4, augment = False):

    class_names = MetadataCatalog.get(dataset_name).thing_classes
    ao_json_dir = ao_setup_project_dir(dataset_name, class_names)
    
    data_loader = build_test_loader(cfg, dataset_name)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    for inputs in data_loader:
        gt_scores, gt_classes, gt_boxes, _ = detect_mislabeled_annotations_per_image(inputs, model, augment)
        mislabeled_gt_inds = gt_scores > mismatch_thresh
       
        gt_class_info = [(class_id.item(), class_names[class_id]) for class_id in gt_classes]
        ao_format_annotations(inputs[0]['image_id'], ao_json_dir, gt_boxes, gt_class_info, mislabeled_gt_inds)
     
def eval_mislabel_detection(dataset_name, cfg, mismatch_thresh = 0.4, augment =False):

    data_loader = build_test_loader(cfg, dataset_name)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    n = len(data_loader)
    
    tp = torch.zeros(n)
    fp = torch.zeros(n)

    total = torch.zeros(n)
    npos = torch.zeros(n)

    for idx, inputs in enumerate(data_loader):
        gt_scores, _, _, gt_mislabeled_ids = detect_mislabeled_annotations_per_image(inputs, model, augment)
        pred_mislabeled_ids = gt_scores > mismatch_thresh
        
        total[idx] = gt_scores.shape[0]
        npos[idx] = torch.sum(gt_mislabeled_ids).int()

        tp[idx] = torch.sum(torch.logical_and(gt_mislabeled_ids, pred_mislabeled_ids)).int()
        fp[idx] = torch.sum(torch.logical_and(torch.logical_not(gt_mislabeled_ids), pred_mislabeled_ids)).int()

    recall = torch.sum(tp)/ torch.sum(npos)
    precision = torch.sum(tp)/ (torch.sum(tp) + torch.sum(fp))
    qa = (torch.sum(tp) + torch.sum(fp))/torch.sum(total)
    return recall.item(), precision.item(), qa.item()
    
    

    
    


        
    
    



