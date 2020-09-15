import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import PascalVOCDetectionEvaluator

__all__ = ["load_noisy_voc_instances", "register_pascal_voc"]

CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

def load_noisy_voc_instances(
    dirname: str,
    split: str,
    class_names: Union[List[str], Tuple[str, ...]],
    add_noise=False
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(
        os.path.join(dirname, "ImageSets", "Main", split + ".txt")
    ) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(
        os.path.join(dirname, "Annotations/")
    )
    
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")


        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            
            #process annotation bias
            mislabeled = obj.find("biased")
            
            isMislabeled = False
            if mislabeled is not None:
                isMislabeled = True

           
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [
                float(bbox.find(x).text)
                for x in ["xmin", "ymin", "xmax", "ymax"]
            ]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {
                    "category_id": class_names.index(cls),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "mislabeled": isMislabeled  
                }
            )
        r["annotations"] = instances
        dicts.append(r)

    return dicts


def register_pascal_voc(name, dirname, split, year = 2007, class_names=CLASS_NAMES):
    DatasetCatalog.register(
        name, lambda: load_noisy_voc_instances(dirname, split, class_names)
    )
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names),
        dirname=dirname,
        year=year,
        split=split
    )


def split_pascal_voc(voc_root):
    splits = ["PART1", "PART2"]
    
    fileids = np.array([filename.split(".")[0] for filename in os.listdir(os.path.join(voc_root, "Annotations"))])
    np.random.shuffle(fileids)

    split_bar = round(len(fileids)/2)
    first_half = os.path.join(voc_root, "ImageSets", "Main", "{}.txt".format(splits[0]))
    np.savetxt(first_half, fileids[:split_bar], delimiter=" ", fmt="%s" )
    second_half = os.path.join(voc_root, "ImageSets", "Main", "{}.txt".format(splits[1]))
    np.savetxt(second_half, fileids[split_bar:], delimiter=" ", fmt="%s")

class PascalVOCTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create VOC evaluator(s) for a given dataset.
        """
        return PascalVOCDetectionEvaluator(dataset_name)
      

def train_fastrcnn_on_pascal_voc_split(train_dataset_name, test_dataset_name, batch_size, num_iter, output_dir = "."):
    #setup cfg
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
    cfg.DATASETS.TRAIN= (train_dataset_name, )
    cfg.DATASETS.TEST= (test_dataset_name, )
   
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.MAX_ITER = num_iter 
    cfg.SOLVER.STEPS = [round(num_iter * 3/4), ]
    
    cfg.OUTPUT_DIR = os.path.join(output_dir, "OUTPUT_{}".format(train_dataset_name))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #set training
    trainer = PascalVOCTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg

def train_fastrcnn_on_pascal_voc(voc_root, batch_size, num_iter, output_dir = "."):
    
    #split voc into two parts
    splits = ["PART1", "PART2"]
    split_pascal_voc(voc_root)

    #register splits and define configs 
    part1_dataset_name = "VOC2007_{}".format(splits[0])
    register_pascal_voc(part1_dataset_name, voc_root, splits[0])
    part2_dataset_name = "VOC2007_{}".format(splits[1])
    register_pascal_voc(part2_dataset_name, voc_root, splits[1])

    #set fastrcnn training
    cfg1 = train_fastrcnn_on_pascal_voc_split(part1_dataset_name, part2_dataset_name, batch_size, num_iter)
    cfg2 = train_fastrcnn_on_pascal_voc_split(part2_dataset_name, part1_dataset_name, batch_size, num_iter)
    
    return [cfg1, cfg2]








