import os
import json
import numpy as np


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
