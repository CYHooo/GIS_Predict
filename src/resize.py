import os
import sys
import cv2
import rasterio
from rasterio.warp import Resampling

def tif_resize(img_path, scaleStr):
    os.environ["PROJ_LIB"] = "D:\\miniconda3\\envs\\dlm\\lib\\site-packages\\rasterio\\proj_data"

    src = rasterio.open(img_path)
    scale = float(scaleStr)
    resize = src.read(out_shape=(src.count, int(src.height * scale), int(src.width * scale)),
                      resampling=Resampling.cubic)

    dst_transform = src.transform * src.transform.scale(
        (src.width / resize.shape[-1]),
        (src.height / resize.shape[-2])
    )

    meta = src.meta.copy()
    meta.update({
        'transform': dst_transform,
        'width': resize.shape[2],
        'height': resize.shape[1]
    })

    with rasterio.open(f"{img_path.replace('.tif', '_down.tif')}", "w", **meta) as dst:
        # iterate through bands
        for i in range(resize.shape[0]):
              dst.write(resize[i].astype(rasterio.uint32), i+1)

def img_resize(img_path, scaleStr):
    from PIL import Image
    if os.path.isfile(img_path):
        img = Image.open(img_path)
        dst_size = img.size * scaleStr
        dst = img.resize(dst_size, resample=Image.CUBIC)
        exif = img.info["exif"]
        dst.save(img_path.replace(".", "_down."), exif=exif)


    elif os.path.isdir(img_path):
        from glob import glob
        img_list = glob(img_path)
        os.makedirs(img_path + "/downsample/", exist_ok=True)
        for i in img_list:
            name = i.split("\\")[-1]
            img = Image.open(i)
            dst_size = img.size * scaleStr
            dst = img.resize(dst_size, resample = Image.CUBIC)
            exif = img.info["exif"]
            dst.save(os.path.join(img_path + "/downsample/", name), exif=exif)

def resizing(img_path, scaleStr, mode):
    if mode == "tif":
        tif_resize(img_path, scaleStr)
    elif mode == "img":
        img_resize(img_path, scaleStr)

if __name__ == "__main__":
    """
    example :   1. python resize.py "D:/data/sample.tif" 0.5 "tif"
                2. python resize.py "D:/data/sample.img" 0.5 "img"
                3. python resize.py "D:/data/" 0.5 "img"
    """
    resizing(sys.argv[1], sys.argv[2], sys.argv[3])
    print("success", flush=True)