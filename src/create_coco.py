import os
import numpy as np
from skimage import measure
from itertools import groupby

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pycocotools import mask 
from src.cocojson.pycococreatortools import pycococreatortools

from src.split_class import *
from src.rdp_algo import *
from config.common import *

from typing import Dict


def create_coco_annotation(annotation_id, image_id, category_info, contour, is_crowd):
    # rdp algorithm is also organization the surroundings of objects but, this is arrangement from the perspective of polygons
    # contour = rdp_algo(contour, epsilon=epsilon)  # TODO: set optimized epsilon

    """ 
    make polygon without rdp
    """
    '''
    @23/03/12 Yeongho:
    change create_coco_annotation(img_id, annotation_id) --> create_coco_annotation(annotation_id, img_id) as pycococreatetool()
    add check annotation_id is None or not
    '''
    if annotation_id is None:
        return None
    
    contour = [np.array(contour[i]).ravel() for i in range(len(contour))]
    score = np.array(1.0).astype('float32').tolist()

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "segmentation": contour,
        "score":score
    }

    return annotation_info

def create_json(img_path : os.path, rcnn_infos : list, road_masks : list, classes : Dict, positions: list):
    """
    convert from detected mask to json with Polygon of object that detected by AI
    information of object is split each.

    :param img_path: original image path
    :param rcnn_infos: a list for each crop image's mask rcnn info
    :param road_masks: a list for each crop image;s unet result mask
    :param postions: input mask's position (ie: [[0,0],[1,0],...]), for format road masks' xy coordinates
    :return: data formed json format that is had image path, each detected object's class name,
                                                 each detected object's position and information for original image.
    """

    ## building & vinyl house info
    box_thres     = [rcnn_info[0] for rcnn_info in rcnn_infos]
    class_thres   = [rcnn_info[1] for rcnn_info in rcnn_infos]
    class_ids     = [rcnn_info[2] for rcnn_info in rcnn_infos]
    masks_args     = [rcnn_info[3] for rcnn_info in rcnn_infos]
    masks_sigmoids = [rcnn_info[4] for rcnn_info in rcnn_infos]

    image_id = 1
    annotation_id = 1

    # saved index for class name of object that we want to detect
    label_idx = [category["id"] for category in CATEGORIES if category["name"] in list(classes.keys())]

    # open the original image for size of image
    img = Image.open(img_path)

    # save image size as variable
    img_size = img.size

    # input image path and image size
    image_info = pycococreatortools.create_image_info(image_id, os.path.basename(img_path), img_size)
    output["images"].append(image_info)

    # iteration about split object to make json for building & vinyl house
    for i in range(len(class_ids)):
        if class_ids[i] is None:
            i += 1
        else:
            for j in range(len(class_ids[i])):
                
                class_id = np.array(class_ids[i][j]).astype('uint64').tolist()
                # contour = masks_sigmoid[:,:,i]
                contour = masks_sigmoids[i][j]
                category_info = {'id': class_id, 'is_crowd': 0}
                # bounding_box = np.array(box_thres[i]).astype('float64').tolist
                score = class_thres[i][j]
                annotation_info = pycococreatortools.create_annotation_info(
                            annotation_id, image_id, category_info, contour, score, image_size=img_size
                            )
                if annotation_info is not None:    
                    output["annotations"].append(annotation_info)
                    annotation_id += 1
        

    # in mask, split each object by colors
    for road_mask, position in zip(road_masks,positions):

    # for i in range(len(road_masks)):
        divided_obj = split_class(road_mask, label_idx, list(classes.values())[2:], position)

        # iteration about split object to make json for paved road & unpaved road
        for obj in divided_obj:
            class_id = obj["id"]
            contour = obj["contour"]
            category_info = {'id': class_id, 'is_crowd': 0}
            annotation_info = create_coco_annotation(
                        annotation_id, image_id, category_info, contour, is_crowd=0
                        )
            if annotation_info is not None:

                output["annotations"].append(annotation_info)

                annotation_id += 1

    return output
