import os
import json
import numpy as np


def ao_setup_project_dir(dirname, classes):

    os.makedirs(dirname, exist_ok = True)
    
    #generate classes.json
    ao_classes = []
    for idx, class_name in enumerate(classes):
        color = np.random.choice(range(256), size=3)
        hexcolor = "#%02x%02x%02x" % tuple(color)
        ao_class = {
            "id": idx,
            "name": class_name,
            "color": hexcolor,
            "attribute_groups": None }
        ao_classes.append(ao_class)

    with open(os.path.join(dirname, "classes.json"), "w+") as classes_json:
        classes_json.write(json.dumps(ao_classes, indent = 2))

    ann_dir = os.path.join(dirname, "jsons")
    os.makedirs(ann_dir, exist_ok = True)
    return ann_dir
  


def ao_format_annotations(image_id, ao_json_dir, boxes, classes, mislabeled):
      
    ao_base_element = {'type': "bbox", 'classId': None, 'className': None, 'probability': 100, 'points': [], 'attributes': [], 'attributeNames': []}
       
    instances = []

    for box, classInfo, isMislabeled in zip(boxes, classes, mislabeled):
        box = box.tolist()
        instance = dict(ao_base_element)
        instance["classId"] = classInfo[0]
        instance["className"] = classInfo[1]     
        instance["points"] = {"x1": box[0], "x2":box[2], "y1":box[1], "y2":box[3]}

        if isMislabeled:
            instance["error"] = True
        instances.append(instance)
    anndata = json.dumps(instances)

    annpath = os.path.join(ao_json_dir, "{}.jpg___objects.json".format(image_id))
    with open(annpath, "w+") as annfile:
        annfile.write(anndata)
