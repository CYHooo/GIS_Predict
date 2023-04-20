import os
import json

# ---------------------- other file package ------------------------ #
# from tif_img import *

dir = os.getcwd() + "/result/"


def convert_dict(idx, image_id, category_id, segmentation):
    dic = {"id": idx,
           "image_id": image_id,
           "category_id": category_id,
           "segmentation": [segmentation],
           "area": 0, "bbox": [], "iscrowd": 0, "attributes": {"occluded": False}}
    return dic


def load_data(data):
    '''
    in json of coco format, load data divided into parts. in annotations, there are the polygon of image coordinate system.

    :param data: coco format json data
    :return: data list that we are read the json file.
    '''

    # bring the class_name in CATEGORIES
    classes = [d['name'] for d in data["categories"]]

    # to bring image_id and name, i make the type of variable as dictionary.
    imgs = dict()
    img_size = []

    for d in data["images"]:
        name = d["file_name"]
        id = d["id"]

        imgs[id] = name
        img_size.append([d["height"], d["width"]])

    # sort by image_id.
    anno = sorted(data["annotations"], key=lambda x: x["image_id"])

    annotations = [[]]
    curr_id = anno[0]["image_id"]

    for a in anno:
        # segmentation 각 polygon vertices [x,y]로 묶기
        a["segmentation"] = [[[a["segmentation"][j][i], a["segmentation"][j][i + 1]] for i in range(0, len(a["segmentation"][j]), 2)] for j in range(len(a["segmentation"]))]

        # ----------- image id를 통한 list 분리 ---------- #
        if a['image_id'] != curr_id:
            annotations.append([])
        annotations[-1].append(a)
        curr_id = a['image_id']

    annotations = [sorted(annotations[i], key=lambda x: x["category_id"]) for i in range(len(annotations))]

    return annotations, imgs, classes, img_size
