"""
    python version: 3.8.10 (python >= 3.8)
"""

# ------------------ basic package ------------------------- #
import os
import warnings
import datetime
import time

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
# os.environ["PROJ_LIB"] = "C:\\Users\\CYH\\anaconda3\\envs\\rcnn-AI\\Lib\\site-packages\\rasterio\\proj_data"

warnings.filterwarnings(action="ignore")

# ------------------ tensorflow ---------------------------- #
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf

# ------------------ CV Package ---------------------------- #
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ------------------ other files ------------------ #
# MODEL
from mask_rcnn import MASK_RCNN
from unet import Unet

# UTIL
from src.create_coco import create_json
from src.tif_img import *
from src.format_gsd import format_gsd, check_gsd, save_img
from src.cocojson.pycococreatortools import pycococreatortools

# CONFIG
from config.common import _init, _setting, print_status, ComposeCoordinate
from config.common import *

# whether to use gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

'''
class     : building,       vinyl house,   paved road,     unpaved road
BGR-color : [244,67,54],    [233,30,99],   [64,64,215],    [30,215,124]
'''


def main(img_path: os.path):
    """
    load the tif image and process that and convert polygon extracted by deep-learning to geoJson format about road and building.

    :param img_path : image path for processing
    :return : None
    :exception: i think that it's better to put tif for exporting to GEO polygon from polygon detected by AI.
                you can also put png or jpg as input for detecting some object,
                but you cannot convert GEO polygon because there is no coordinate system in png or jpg.
    """

    start = datetime.datetime.now()

    # initial setting
    cfg = _init(img_path)
    _setting(cfg=cfg)

    # print status for sending to js
    status = print_status(cfg["status"], [0, 0])

    # mask_rcnn is detection model for building
    mask_rcnn = MASK_RCNN()
    # UNet is detection model for road
    uNet = Unet()

    ## coor init
    deformer = ComposeCoordinate()

    # check image's GSD value
    unit_pixel = check_gsd(os.path.join(cfg["base_dir"], cfg["img_name"]) + cfg["ext"])

    # format gsd to 0.25m
    imgs, positions = format_gsd(unit_pixel, os.path.join(cfg["base_dir"], cfg["img_name"]) + cfg["ext"])

    # list for crop image's predict masks (if you don't need to crop image, info still append to list )
    road_masks = []
    building_masks = []

    image_id = 1
    annotation_id = 1

    # open the original image for size of image
    img = Image.open(img_path)

    # save image size as variable
    img_size = img.size

    # input image path and image size
    image_info = pycococreatortools.create_image_info(image_id, os.path.basename(img_path), img_size)
    output["images"].append(image_info)

    for img, position in zip(imgs, positions):
        # detecting for building
        bbox_img, building_mask, colors, rcnn_info = mask_rcnn.detect_image(img, position)

        # detecting for road
        total_mask, road_mask, road_color = uNet.detect_image(img, building_mask)

        # merge colors for building and road
        colors.append(road_color)

        json_data, ann_id = create_json(building_mask, road_mask, rcnn_info, colors, position, 
                                deformer, annotation_id, image_id)

        # get all crop image predict result mask
        road_masks.append(road_mask)
        building_masks.append(building_mask)
        
        # annotations id
        annotation_id = ann_id

    # save json file
    # import json
    # with open(cfg["result_dir"] + 'result.json', 'w') as f:
    #     json.dump(json_data,f)


    # save predict result
    save_img(img_path, building_masks, road_masks, cfg["pred_path"], unit_pixel)

    status = print_status(status + 10, [0, 3])  # 10~13

    # print status for sending to js
    status = print_status(status + 30, [0, 3])  # 10~13 + 30 + 0~3 = 40~46

    """ temporal code for test
    total_mask = np.asarray(Image.open(os.path.join(cfg["pred_path"], cfg["img_name"] + '_mask.png')))
    classes = {'building': [54, 67, 244], 'vinyl house': [99, 30, 233], 'paved road': [215, 64, 64], 'unpaved road': [124, 215, 30]}
    """

    # print status for sending to js
    print_status(status + 45, [5, 8])  # 40~46 + 45 + 5~8 = 90~99

    # convert from data formed json format to geojson
    tif_img(json_data, cfg["base_dir"], cfg["geo_path"])

    finish = (datetime.datetime.now() - start).seconds
    print(f"\nduring time : {finish} s")

if __name__ == "__main__":
    # path = "img/(B060)정사영상_2018_37612100.tif"
    # path = "img/(B060)정사영상_2018_37616020.tif"
    path = "img/tiled_0.tif"
    main(path)
