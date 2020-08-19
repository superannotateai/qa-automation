from fvcore.common.file_io import PathManager
import xml.etree.ElementTree as ET
from typing import Dict

from tqdm import tqdm
import numpy as np
import os

def bias_pascal_voc(
    dirname: str,
    noise_ratio: float,
    bias_rule: Dict[str, str]
):
    """
    Add Noise to Pascal VOC detection annotations.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        noise_ratio: Noise ratio of biased annotations
        bias_rule: Assymetric mislabel rules between classes 
    """

    annotation_dirname = PathManager.get_local_path(
        os.path.join(dirname, "Annotations/")
    )

    annotation_files = np.array(PathManager.ls(annotation_dirname))
    
    num_biased_files = round(len(annotation_files) * noise_ratio)
    np.random.shuffle(annotation_files)
    biased_files = set(annotation_files[:num_biased_files])

    bias_stats = dict.fromkeys(["total", "mislabeled", "skipped"], 0)

    for filename in tqdm(annotation_files):
        anno_file = os.path.join(annotation_dirname, filename)

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)
        
        instances = tree.findall("object")
        num_instances = len(instances)
        bias_stats['total'] += num_instances

        if filename in biased_files:
            mislabel_ratio = round(num_instances*0.7)
            np.random.shuffle(instances)
            biased_instances = instances[:mislabel_ratio]
            for instance in biased_instances:
                cls_name = instance.find("name")
                
                if cls_name.text in bias_rule.keys():
                    biased_cls_name = bias_rule[cls_name.text]
                    cls_name.text = biased_cls_name
                    mislabel_attr = ET.SubElement(instance, "mislabeled")
                    mislabel_attr.text = '1'
                    bias_stats['mislabeled'] += 1
                else:
                    skipped_attr = ET.SubElement(instance, "skipped")
                    skipped_attr.text = '1'
                    bias_stats['skipped'] +=1
                  
            tree.write(anno_file)


    return bias_stats





