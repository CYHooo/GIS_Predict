# ------------------ basic package ------------------------- #
import os
import cv2
import random
import datetime
import numpy as np

from src.cocojson.pycococreatortools.pycococreatortools import binary_mask_to_polygon

# ------------------ type of variable ------------------------- #
from typing import List, Dict

class DetransformCoord:
    def __init__(self):
        self.prev_pos = np.array([0, 0])
        self.pos_shape = np.array([0, 0])
        self.prev_pos_shape = np.array([0, 0])

    def contour2contour(self, contours, position, image_shape):
        position_y, position_x = position
        position_coor = []

        if self.prev_pos[1] == position_y:
            self.pos_shape[0] += self.prev_pos_shape[0]
        else:
            self.pos_shape[1] += self.prev_pos_shape[1]
            self.pos_shape[0] = 0

        for i in range(len(contours)):
            if contours[i] is None:
                continue

            output = (contours[i].reshape(-1, 2) + self.pos_shape).reshape(-1, 1, 2)
            position_coor.append(output)

        self.prev_pos = position[::-1]
        self.prev_pos_shape = image_shape[:2][::-1]

        return position_coor

    def mask2contour(self, masks_coor, position, masks_shape):
        '''
        @23/03/11 Yeongho:
        :param masks_coor: crop predict mask info (type: np.bool_)
        :param positon: position from original image
        :param masks_shape: input mask's shape  (h,w)
        :return: building predict result xy coordinate from original image's coordinate
        '''
        position_y, position_x = position
        position_coor = []

        if self.prev_pos[1] == position_y:
            self.pos_shape[0] += self.prev_pos_shape[0]
        else:
            self.pos_shape[1] += self.prev_pos_shape[1]
            self.pos_shape[0] = 0

        if masks_coor is None:
            return None

        for i in range(masks_coor.shape[-1]):
            # crop_coor = masks_coor[:, :, i].astype('uint8')
            # crop_coor = resize_binary_mask(masks_coor[:, :, i], masks_shape)  # TODO: np.uint로 바꾸고, 다시 bool로 바꾸는 이유가?
            crop_coor = cv2.resize(masks_coor[:, :, i].astype(np.uint8) * 255, masks_shape[::-1])
            crop_coor = binary_mask_to_polygon(crop_coor, tolerance=2)  # (x,y) # TODO: 만약 list가 2개인 결과가 나올 경우는?

            # coor = [[crop_coor[0][a + 1], crop_coor[0][a]] for a in range(0, len(crop_coor[0]), 2)]

            # x = np.array(coor)[:, 1] + position_x * masks_shape[1]
            # y = np.array(coor)[:, 0] + position_y * masks_shape[0]
            # output = [val for pair in zip(x, y) for val in pair]
            output = (np.array(crop_coor).reshape(-1, 2) + self.pos_shape).reshape(1, -1).tolist()[0]  # x,y # TODO: 간편화
            position_coor.append(output)

            """
            m = np.zeros((2880, 2319), dtype=np.uint8); m = cv2.line(m, (0,999*2),(10000,999*2), color=255); m = cv2.line(m, (0,999),(10000,999), color=255); m = cv2.line(m, (999,0), (999, 10000), color=255); m = cv2.line(m, (999*2,0), (999*2, 10000), color=255)
            m = cv2.fillConvexPoly(m, np.array([output]).astype(np.int32).reshape(-1,2), color=255)
            import matplotlib.pyplot as plt; plt.imshow(m, cmap="gray"); plt.show()
            """

        self.prev_pos = position[::-1]
        self.prev_pos_shape = masks_shape[::-1]

        return position_coor

def DynamicEpsilonbyContourarea(contour, class_id):
    """
    set epsilon of rdp algorithm optimized by image shape
    :param contour:
    :param class_id:
    :return: epsilon optimized by image shape
    """
    contourArea = cv2.contourArea(np.array([contour[0]])) / 10000

    # 건물
    if class_id in [1, 2]:
        if contourArea >= 10:
            epsilon = 20
        elif contourArea >= 3:
            epsilon = 15
        else:
            epsilon = 8
    else:
        if contourArea >= 30:
            epsilon = 30
        elif contourArea >= 10:
            epsilon = 20
        elif contourArea >= 5:
            epsilon = 15
        elif contourArea >= 2:
            epsilon = 10
        else:
            epsilon = 8

    return epsilon

def DynamicAreaThresholdbyImageshape(image_shape, class_id):
    """
    set Threshold of contour area optimized by image shape
    :param image_shape: (height, width)
    :param class_id:
    :return: area threshold optimized by image shape
    """

    cutline = image_shape[0] * image_shape[1] * 0.0001
    # 도로
    if class_id in [1,2]:
        params = 0.4
    # 건물
    else:
        params = 1.1

    area_threshold = int(cutline * params)
    return area_threshold


def _init(img_path: os.path) -> Dict:
    """
    initial configuration

    :param img_path: default path
    :return: configuration
    """

    # base path to save result
    abspath = os.path.abspath(img_path)
    base_dir = abspath.rsplit(os.sep, 1)[0]
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

def create_coco_annotation(annotation_id, image_id, category_info, contour, is_crowd):
    '''
    @23/03/12 Yeongho:
    change create_coco_annotation(img_id, annotation_id) --> create_coco_annotation(annotation_id, img_id) as pycococreatetool()
    add check annotation_id is None or not
    '''
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "segmentation": contour,
    }

    return annotation_info

epsilon = 30
find_contour_threshold = {"building": 3, "road": 30}  # default, road
area_threshold = 2000
