'''
    python version: 3.8.10 (python >= 3.8)

    python package:    
        tensorflow == 2.10.0 (cuda==11.5,cudnn==8.3.1)
        silence_tensorflow
        numpy == 1.20.0

        opencv-python == 4.1.2.30 (opencv < 4.2.0)
        scikit-image == 0.16.2 (scikit-image < 0.19.0)
        scipy == 1.10.0
        pillow == 9.3.0

        tqdm

        psutil
        rasterio
        pycocotools-windows
        geojson
        rdp
'''

# ------------------ basic package ------------------------- #
import os
import warnings

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

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

# CONFIG
from config.common import _init, _setting, print_status

# whether to use gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


'''
class     : building,       vinyl house,   paved road,     unpaved road,
BGR-color : [244,67,54],    [233,30,99],   [64,64,215],    [30,215,124],
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

    # initial setting
    cfg = _init(img_path)
    _setting(cfg=cfg)

    # print status for sending to js
    status = print_status(cfg["status"], [0, 0])

    # mask_rcnn is detection model for building
    mask_rcnn = MASK_RCNN()
    # UNet is detection model for road
    uNet = Unet()

    # img = Image.open(os.path.join(cfg["base_dir"], cfg["img_name"]) + cfg["ext"])

    # check image's GSD value
    unit_pixel = check_gsd(os.path.join(cfg["base_dir"], cfg["img_name"]) + cfg["ext"])

    # format gsd to 0.25m
    imgs, positions = format_gsd(unit_pixel, os.path.join(cfg["base_dir"], cfg["img_name"]) + cfg["ext"])
    
    # list for crop image's predict info (if don't need to crop image, info still append to list )
    rcnn_infos = []
    road_masks = []
    building_masks = []

    for img, position in zip(imgs, positions):
        # detecting for building
        bbox_img, building_mask, colors, rcnn_info = mask_rcnn.detect_image(img, position)

        # print status for sending to js
        

        # detecting for road
        total_mask, road_mask, road_color = uNet.detect_image(img, building_mask)

        # merge colors for building and road
        colors.extend(road_color)

        # merge class_name and color correctly
        classes = {label: color for label, color in zip(cfg["class_name"], colors)}

        # got all crop image predict result & info
        road_masks.append(road_mask)
        building_masks.append(building_mask)
        rcnn_infos.append(rcnn_info)

    # save predict result
    save_img(img_path, building_masks, road_masks, cfg["pred_path"], unit_pixel)
    
    status = print_status(status + 10, [0, 3])  # 10~13
    # remove None rcnn infos
    # mask_save_rcnn = rcnn_infos.copy()
    # geojson_rcnn = rcnn_infos.copy()
    # mask_save_rcnn = [[info for info in infos if info is not None] for infos in rcnn_infos]
    # mask_save_rcnn = [x for x in rcnn_infos if x]

    # print status for sending to js
    status = print_status(status + 30, [0, 3])  # 10~13 + 30 + 0~3 = 40~46

    """ temporal code for test
    total_mask = np.asarray(Image.open(os.path.join(cfg["pred_path"], cfg["img_name"] + '_mask.png')))
    classes = {'building': [54, 67, 244], 'vinyl house': [99, 30, 233], 'paved road': [215, 64, 64], 'unpaved road': [124, 215, 30]}
    """
    # create some data formed by json format
    json_data = create_json(img_path, rcnn_infos, road_masks, classes, positions)

    # print status for sending to js
    print_status(status + 45, [5, 8])  # 40~46 + 45 + 5~8 = 90~99

    # convert from data formed json format to geojson
    tif_img(json_data, cfg["geo_path"])
   


if __name__ == "__main__":
    # path = "D:\\data\\광주\\crop\\TEST_A_15.tif"
    # path = "img\\orthophoto-2.tif"
    # path = "img\\02_05_TEST.tif"
    path = 'img\\test_1.tif'
    # path = sys.argv[1]
    main(path)
