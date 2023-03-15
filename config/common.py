# ------------------ basic package ------------------------- #
import os
import random
import datetime

# ------------------ type of variable ------------------------- #
from typing import List, Dict


def _init(img_path: os.path) -> Dict:
    """
    initial configuration

    :param img_path: default path
    :return: configuration
    """

    # base path to save result
    base_dir = img_path.rsplit("\\", 1)[0]
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    result_dir = img_path.replace(".tif", "") + "/result/"
    cfg = {"status": 0,
           "base_dir": base_dir,
           "img_name": img_name,
           "ext": ext,
           "result_dir": result_dir,
           "pred_path": result_dir + 'predict_mask/',
           "geo_path": result_dir + 'geo_Json/',
           "class_name": [CATEGORIES[i]["name"] for i in range(4)],
           }

    return cfg


def _setting(cfg: Dict) -> None:
    """
    prepare processing by creating directories for task

    :param cfg : config with the path to create
        pred_path : path to save predicted image
        geo_path : path to save made by geojson from detected polygon
    :return : None
    """
    os.makedirs(cfg["pred_path"], exist_ok=True)
    os.makedirs(cfg["geo_path"], exist_ok=True)



def print_status(progress: int, rand: List) -> int:
    """
    print present status and send to js through flush

    :param progress : number of previous status
    :param rand : random range to add present status
    :return : present status
    """
    # add previous status and random int
    status = progress + random.randint(rand[0], rand[1])
    # if added number is over the 100, limit to not exceed 100.
    if status > 100:
        status = 100

    # print present status as flush
    print('{"type": "status", "value": ' + str(status) + '}', end="", flush=True)
    return status


def random_color() -> List:
    """
    make random RGB color

    :return : RGB color made randomly
    """
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


CATEGORIES = [
    {
        "id": 1,
        "name": "building",
        "supercategory": "building",
    },
    {
        "id": 2,
        "name": "vinyl house",
        "supercategory": "vinyl_house",
    },
    {
        "id": 3,
        "name": "paved road",
        "supercategory": "paved_road",
    },
    {
        "id": 4,
        "name": "unpaved road",
        "supercategory": "unpaved_road",
    },
    {
        "id": 5,
        "name": "bush",
        "supercategory": "bush",
    },
    {
        "id": 6,
        "name": "tree",
        "supercategory": "tree",
    },
    {
        "id": 7,
        "name": "paddy",
        "supercategory": "paddy",
    },
    {
        "id": 8,
        "name": "field",
        "supercategory": "field",
    },
    {
        "id": 9,
        "name": "grave",
        "supercategory": "grave",
    },
]

############################################
#      COCO Json Base Info
############################################
now = datetime.datetime.now()
INFO = dict(
    description="Ko-mapper building dataset",
    url=None,
    version='1.0',
    year=now.year,
    contributor='Lastmile',
    date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
)

LICENSES = [
    dict(
        url=None,
        id=0,
        name=None,
    )
]

output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": [],
}

epsilon = 30
find_contour_threshold = {"building": 3, "road": 30}  # default, road
area_threshold = 2000
