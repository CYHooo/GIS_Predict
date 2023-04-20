import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pycocotools import mask
from src.cocojson.pycococreatortools import pycococreatortools

from src.split_class import *
from src.rdp_algo import *
from config.common import *

from typing import Dict

def create_json(crop : Image, road_mask : Image, rcnn_infos : list, colors : list, position: tuple, 
                deformer, annotation_id, image_id):
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
    # get crop image position for mask coor
    deformer.position_move(position)

    ## building & vinyl house info
    box_thres, class_thres, class_ids, masks_args, masks_sigmoids = [], [], [], [], []


    box_thres.append(rcnn_infos[0])
    class_thres.append(rcnn_infos[1])
    class_ids.append(rcnn_infos[2])
    masks_args.append(rcnn_infos[3])
    masks_sigmoids.append(rcnn_infos[4])

    image_size = Image.fromarray(crop).size

    # iteration about split object to make json for building & vinyl house
    for i in range(len(class_ids)):
        if class_ids[i] is None:
            continue

        for j in range(len(class_ids[i])):
            class_id = int(class_ids[i][j])
            contour = np.asarray(Image.fromarray(masks_sigmoids[i][:,:,j]).convert('1')).astype(np.uint8)
            # contour = masks_sigmoids[i][j]  # array to list
            category_info = {'id': class_id, 'is_crowd': 0}
            score = class_thres[i][j]
            annotation_info = pycococreatortools.create_annotation_info(annotation_id, image_id, category_info, contour, score, 
                                                                        position, deformer, image_size=image_size)
            if annotation_info is not None:
                output["annotations"].append(annotation_info)
                annotation_id += 1
   
    
    # in mask, split each object by colors
    label_idx = [3, 4]
    divided_obj = split_class(road_mask, label_idx, colors=colors[1], position=position, deformer=deformer)

    # iteration about split object to make json for paved road & unpaved road
    for obj in divided_obj:
        class_id = obj["id"]
        contour = obj["contour"]
        category_info = {'id': class_id, 'is_crowd': 0}

        # rdp algorithm is also organization the surroundings of objects but, this is arrangement from the perspective of polygons
        epsilon = DynamicEpsilonbyContourarea(contour, class_id)
        contour = rdp_algo(contour, epsilon=epsilon)
        annotation_info = create_coco_annotation(annotation_id, image_id, category_info, contour, is_crowd=0)

        if annotation_info is not None:
            output["annotations"].append(annotation_info)
            annotation_id += 1


    return output, annotation_id
