import os
import json
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm

def sa_setup_project_dir(dirname, classes):

    os.makedirs(dirname, exist_ok = True)
    
    #generate classes.json
    sa_classes = []
    for idx, class_name in enumerate(classes):
        color = np.random.choice(range(256), size=3)
        hexcolor = "#%02x%02x%02x" % tuple(color)
        sa_class = {
            "id": idx,
            "name": class_name,
            "color": hexcolor,
            "attribute_groups": [] }
        sa_classes.append(sa_class)

    with open(os.path.join(dirname, "classes.json"), "w+") as classes_json:
        classes_json.write(json.dumps(sa_classes, indent = 2))

    ann_dir = os.path.join(dirname, "jsons")
    os.makedirs(ann_dir, exist_ok = True)

    return ann_dir
  
def sa_write_status_lists(dirname, qa, completed):
    
    with open(os.path.join(dirname, "qa.json"), "w+") as qa_json:
        qa_json.write(json.dumps(qa, indent = 2))

    with open(os.path.join(dirname, "completed.json"), "w+") as completed_json:
        completed_json.write(json.dumps(completed, indent = 2))

def sa_format_annotations(image_id, sa_json_dir, boxes, classes, mislabeled):
      
    sa_base_element = {'type': "bbox", 'classId': None, 'className': None, 'probability': 100, 'points': [], 'attributes': [], 'attributeNames': []}
       
    instances = []

    for box, classInfo, isMislabeled in zip(boxes, classes, mislabeled):
        box = box.tolist()
        instance = dict(sa_base_element)
        instance["classId"] = classInfo[0]
        instance["className"] = classInfo[1]     
        instance["points"] = {"x1": box[0], "x2":box[2], "y1":box[1], "y2":box[3]}

        if isMislabeled:
            instance["error"] = True
        instances.append(instance)
   
    annpath = os.path.join(sa_json_dir, "{}.jpg___objects.json".format(image_id))

    anndata = []
    if os.path.isfile(annpath):
        anndata = json.load(open(annpath))
       
    with open(annpath, "w+") as annfile:
        anndata.extend(instances)
        annfile.write(json.dumps(anndata))


def sa_convert_from_voc(voc_root, sa_root):
    classes = set()
    
    os.makedirs(sa_root, exist_ok = True)
    
    annotation_dirname = os.path.join(os.path.join(voc_root, "Annotations/"))

    annotation_files = np.array(os.listdir(annotation_dirname))

    sa_annotation_dir = os.path.join(sa_root, "jsons")
    os.makedirs(sa_annotation_dir, exist_ok = True)

    for filename in tqdm(annotation_files):
        anno_file = os.path.join(annotation_dirname, filename)
        instances = []
        with open(anno_file) as f:
            tree = ET.parse(f)
        sa_base_element = {'type': "bbox", 'classId': None, 'className': None, 'probability': 100, 'points': [], 'attributes': [], 'attributeNames': []}

        objects = tree.findall("object")
        for obj in objects:
            class_name = obj.find("name").text
            classes.add(class_name)
            bbox = obj.find("bndbox")
            bbox = [
                float(bbox.find(x).text)
                for x in ["xmin", "ymin", "xmax", "ymax"]
            ]

            instance = dict(sa_base_element)
            instance["classId"] = list(classes).index(class_name)
            instance["className"] = class_name
            instance["points"] = {"x1": bbox[0], "x2":bbox[2], "y1":bbox[1], "y2":bbox[3]}
            instances.append(instance)
        image_id = filename.split(".")[0]
        annpath = os.path.join(sa_annotation_dir, "{}.jpg___objects.json".format(image_id))
        
        with open(annpath, "w+") as annfile:
            annfile.write(json.dumps(instances, indent = 2))
    #generate classes json
    sa_classes = []
    for idx, class_name in enumerate(classes):
        color = np.random.choice(range(256), size=3)
        hexcolor = "#%02x%02x%02x" % tuple(color)
        sa_class = {
            "id": idx,
            "name": class_name,
            "color": hexcolor,
            "attribute_groups": [] }
        sa_classes.append(sa_class)

    with open(os.path.join(sa_root, "classes.json"), "w+") as classes_json:
        classes_json.write(json.dumps(sa_classes, indent = 2))





