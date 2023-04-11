# conda env = maskrcnn
# pip install rasterio

# -------------------------- basic package ------------------------------ #
import os
os.environ["PROJ_LIB"] = "C:\\Users\\jhyoon\\miniconda3\\envs\\dlm\\Lib\\site-packages\\rasterio\\proj_data"

import re

import geojson
import numpy as np

# -------------------------- geometry package ------------------------------ #
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# -------------------------- other file package ------------------------------ #
from src.save_anno import load_data

# -------------------------- visualize package ------------------------------ #
import matplotlib.pyplot as plt

def convert_epsg(src, dst_crs, is_save=True) -> os.path:
    '''
    convert from a coordinate that we have already to dst coordinate
    :param src: existing coordinate(crs)
    :param dst_crs: what crs you want to convert
    :param is_save: variables for whether you save the converted image
    :return: path containing the converted epsg
    '''

    # calculate transform, width, height by original crs and dst crs that you want to convert.
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)

    # copy to maintain meta data in tiff
    meta = src.meta.copy()
    meta.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    dst_name = src.name.replace(".tif", f"_{dst_crs.split(':')[-1]}.tif")

    if not is_save:
        return dst_name

    # save file converted to dst_crs.
    with rasterio.open(os.path.join(os.getcwd(), src.name.replace(".tif", f"_{dst_crs.split(':')[-1]}.tif")), 'w',
                       **meta) as dst:
        for i in range(1, src.count + 1):
            dst_band, dst_transform = reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

    return dst_name

epsgs = ['3857', '4326', '5181', '5186', '5187']
wkts = [re.findall(r'"(.*?)"', rasterio.crs.CRS({"init" : f"epsg:{ep}"}).wkt)[0] for ep in epsgs]

def convert_img_coord_to_crs(annotations, img_path, imgs):
    '''
    convert image coordinate to crs coordinate

    :param annotations: annotations data that loaded from json.
    :param imgs: image file list
    :return: vertices of polygon
    '''
    import re
    vertices = []
    for value, anno in zip(imgs.values(), annotations):
        path = os.path.join(img_path, value)
        src = rasterio.open(path, mode="r+")

        if not "init" in src.crs and re.findall(r'"(.*?)"', src.crs.wkt)[0] in wkts:
            epsg = epsgs[wkts.index(re.findall(r'"(.*?)"', src.crs.wkt)[0])]
            src.crs = rasterio.crs.CRS({"init": f"epsg:{epsg}"})

        # top-left, bottom-right coordinate
        crs_bounds = np.array([[src.bounds[3], src.bounds[0]], [src.bounds[1], src.bounds[2]]], dtype=np.float64)
        # diff top-bottom , right-left
        crs_size = np.array([crs_bounds[0, 0] - crs_bounds[1, 0], crs_bounds[1, 1] - crs_bounds[0, 1]], dtype=np.float64)  # height, width

        # convert epsg
        if src.crs["init"] != "epsg:4326":
            dst_name = convert_epsg(src, "EPSG:4326", is_save=False)
        else:
            dst_name = None

        img_size = src.shape[:2]  # height, width

        # ratio of the actual latitude/longitude per-pixel.
        ratio = (crs_size / img_size)

        # in annotations, we read only segmentation points
        points = [anno[j]['segmentation'] for j in range(len(anno))]

        for point in points:
            for p in range(len(point)):
                point[p] = np.flip(point[p], axis=1)

                # flip the y coordinate, because the image's y-axis direction is under, but the world's y-axis direction is upper.
                point[p] = point[p] * [-1, 1]

                # make the image coordinate to world coordinate and, translate the origin from image's origin point to world's origin.
                point[p] = point[p] * ratio + np.array([crs_bounds[0, 0], crs_bounds[0, 1]])

        vertices.append(points)

    # make two list to one list.
    vertices = sum(vertices, [])
    return vertices, src.crs["init"]


def make_geojson(vertices, result_path, epsg: str):
    '''
    make geojson from json using geopandas

    :param vertices: each point of polygon
    :param result_path: result path
    :param epsg: epsg
    :return: save geojson
    '''
    achive = {"type": 'FeatureCollection', "features": []}

    features = achive["features"]
    for i in range(len(vertices)):
        features.append({})
        features[-1]["type"] = "Feature"
        features[-1]["properties"] = {}
        features[-1]["geometry"] = {}
        features[-1]["geometry"]["coordinates"] = [np.flip(vertices[i][k]).tolist() for k in range(len(vertices[i]))]
        features[-1]["geometry"]["type"] = "Polygon"

    if epsg != "epsg:4326":
        data = [features["geometry"] for features in achive["features"]]
        dst_geometry = rasterio.warp.transform_geom(dict(init=epsg), dict(init="epsg:4326"), data)

        # make tuple to list all of them.
        for d in dst_geometry:
            d["coordinates"] = [list(map(list, d["coordinates"][i])) for i in range(len(d["coordinates"]))]

        # change coordinates transformed by dst crs
        for i in range(len(dst_geometry)):
            achive["features"][i]["geometry"] = dst_geometry[i]

    with open(result_path, 'w') as f:
        geojson.dump(achive, f)

    return {
            "class": result_path.rsplit("/", 1)[-1].split(".")[0].split("_")[-1],
            "geojson": achive
            }

def tif_img(json_data, img_path, save_path):
    """
    convert polygon of image coordinates to world coordinate about epsg in original input tif.

    :param json_data: polygon data formed coco json format of image coordinate system that is detected by AI.
    :param img_path: image path.
    :return: None
    :exception: if you install PostGIS or other program of OSGEO,
                you should check the environment variable for PROJ_LIB(PROJ_DATA - in more recent version, change the name to this.)
    """
    # load data in json data.
    annotations, imgs, classes, _ = load_data(json_data)

    target_ids = [1, 2, 3, 4]  # building, vinyl house, paved road, unpaved road
    result = []
    # I bring only building, paved road, unpaved road if you want to bring other class, you should change the number of class.
    for class_number in target_ids:
        anno_by_cls = [[a for a in anno if a["category_id"] == class_number] for anno in annotations]
        vertices, img_epsg = convert_img_coord_to_crs(anno_by_cls, img_path, imgs)

        # save 2 files that geojson for original crs, and transformed crs.
        result.append(make_geojson(vertices, save_path + f'result_{classes[class_number - 1]}.geojson', img_epsg))

    print('{"type": "status", "value": "100"}', end="", flush=True)
    print('{"type": "result", "value": "' + str(result) + '"}', end="", flush=True)