from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import rasterio
import numpy as np
import cv2

def check_gsd(img_path):
    '''
    @230310 Yeongho:
        :param img_path: input image path (endwiths *.tif or *.png/*.jpg...)
        :param GSD: image GSD (defualt=0.25m)
        :return: image unit pixel size
    '''
    if os.path.splitext(img_path)[-1] == '.tif':
        unit_meter = [250, 250]
        img = np.array(Image.open(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with rasterio.open(img_path, 'r') as src:
            res = src.res

            #-------- For test pixel size -------------#
            if res[0] == 1.0:
                # res = [res[0]/10,res[1]/10]
                return [1004,1004]
            #------------------------------------------#

            bounds = src.bounds
            bound = [bounds[2] - bounds[0], bounds[3] - bounds[1]]
            # width = src.width
            # height = src.height
            unit_pixel = [int(unit_meter[0] / res[0]), int(unit_meter[1] / res[1])]
            if 0 <= img.shape[0]/unit_pixel[0] <= 1.1 and 0 <= img.shape[1]/unit_pixel[1] <= 1.1:
                return  img.shape
            else:
                return unit_pixel
    else:
        print('Input Image is not .tif File...')
        return [1024, 1024]



def format_gsd(unit_pixel, img_path):
    '''
    @23/03/12 Yeongho:
    update func. method, now return img without non-img black pixel

    :unit_pixel: crop img unit pixel(defualt: unit_pixel*gsd=250m)
    :param img_path: img (*.tif/*.png/*.jpg)
    :param save_path: crop img save path
    :return: img, positions
    '''
    
    img = np.array(Image.open(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # check pixel size, need use '/' to get float type reuslt 
    if 0 <= img.shape[0]/unit_pixel[0] <= 1.1 and 0 <= img.shape[1]/unit_pixel[1] <= 1.1:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'))
        return [img], [[0,0]]
    else:
        crop = []
        position = []
        for i in range((img.shape[0]//unit_pixel[0])+1):
            for j in range((img.shape[1]//unit_pixel[1])+1):
                pic = img[i*unit_pixel[0]: (i+1)*unit_pixel[0], j*unit_pixel[1]: (j+1)*unit_pixel[1], :]
                pic = Image.fromarray(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB).astype('uint8'))
                crop.append(pic)
                position.append([i,j])
    return crop, position

# def save_img(img_path, building_masks, road_masks, save_path, img_name, unit_pixel):
#     '''
#     :param img_path: input image path (*.tif)
#     :param building_masks: mask-rcnn predict result masks
#     :param road_masks: unet predict result masks
#     :param save_path: save path
#     :param img_name: image name
#     :param uint_pixel: crop pixel size
#     :return None
#     '''
#     size = unit_pixel
#     img = np.array(Image.open(img_path))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    
#     masks = []
#     mask_save = np.zeros_like(img)
#     blend = np.zeros_like(img)
#     for building_mask, road_mask in zip(building_masks, road_masks):

#         building = cv2.cvtColor(building_mask, cv2.COLOR_BGR2RGB)
#         mask = Image.fromarray(road_mask + building)
#         masks.append(mask)
#     a = img.shape[0]//size[0]
#     b = img.shape[1]//size[1]

#     if img.shape[0]//size[0] > 1 and img.shape[1]//size[1] > 1 :
#         for j in range(1, img.shape[0]//size[0] + 2):
#             for i in range(1, img.shape[1]//size[1] + 2):
#                 for m in range(len(masks)):
#                     part = cv2.cvtColor(np.array(masks[m]), cv2.COLOR_RGB2BGR)
#                     mask_save[size[0]*(i-1): size[0]*i, size[1]*(j-1): size[1]*j, :] = part

#         mask_save = Image.fromarray(mask_save)
#         mask_save.save(os.path.join(save_path + img_name + '_mask.png'))
#     else:
#         mask.save(os.path.join(save_path, img_name + '_mask.png'))
        



'''
    For crop image with non-img black pixel
'''

# def format_gsd(unit_pixel, img_path):
#     '''
#     :param unit_pixel: crop img unit pixel(defualt: unit_pixel*gsd=250m)
#     :param img_path: img (*.tif/*.png/*.jpg)
#     :param save_path: crop img save path
#     :return: img, positions
#     '''

#     img = Image.open(img_path)

#     # check pixel size, need use '/' to get float type reuslt 
#     if 0 <= img.size[0]/unit_pixel[0] <= 1.1 and 0 <= img.size[1]/unit_pixel[1] <= 1.1:
#         return [img], [[0,0]]
#     else:
#         size = unit_pixel
#         crop = []
#         position = []
#         # Loop through the input image, cutting out tiles of the specified size
#         for i in range(0, img.width, size[0]):
#             for j in range(0, img.height, size[1]):
#                 box = (i, j, i + size[0], j + size[1])
#                 # Check if the last tile needs to be padded
#                 if box[2] > img.width:
#                     # Calculate the amount of padding needed
#                     pad_width = size[0] - (img.width - i)
                    
#                     # Create a new box with the padded width
#                     box = (i, j, i + pad_width, j + size[1])
                    
#                     # Create a new tile by cropping the original image
#                     tile = img.crop(box)
                    
#                     # Create a new image with the padded size
#                     padded_tile = Image.new("RGB", size, color="black")
                    
#                     # Paste the original tile onto the padded image
#                     padded_tile.paste(tile, (0, 0))

#                     crop.append(padded_tile)
#                     position.append([j//size[1],i//size[0]])

#                     ## Calculate the sequential tile name
#                     # tile_name = f"./output_{j//size[1]}_{i//size[0]}.tif"
#                     ## Save the padded tile with the sequential name
#                     # padded_tile.save(tile_name)

#                 else:
#                     # If the tile is a full tile, just crop and save it
#                     tile = img.crop(box)
#                     crop.append(tile)
#                     position.append([j//size[1],i//size[0]])
#                     ## Calculate the sequential tile name
#                     # tile_name = f"./output_{j//size[1]}_{i//size[0]}.tif"
#                     ## Save the tile with the sequential name
#                     # tile.save(tile_name)

#     return crop, position

''' 
    For stack crop masks with non-image black pixel
    Not finished...
'''
# def save_img(img_path, building_masks, road_masks, save_path, img_name, unit_pixel):
#     '''
#     :param img_path   : cut size predict blend img (*.tif/*.png/*.jpg)
#     :param save_path  : original size result blend img save path
#     :return: None
#     '''

#     size = unit_pixel
#     # Open the first output tile to get the size of the output image
#     img = Image.open(img_path)
#     w,h = img.size

#     # Calculate the number of rows and columns in the output image
#     num_cols = (w + size[0] - 1) // size[0] 
#     num_rows = (h + size[1] - 1) // size[1] 

#     # Create a new image with the size of the output image
#     output_image = Image.new("RGB", (w, h), color="black")

#     # Loop through the output tiles and paste them onto the output image
#     for i in range(num_rows):
#         for j in range(num_cols):
            
#             # Open the tile and paste it onto the output image
#             tile = Image.open(tile_name)
#             output_image.paste(tile, (j * size[0], i * size[1]))
#     output_image.save(save_path + "mask.png")

def save_img(img_path, building_masks, road_masks, save_path, unit_pixel):
    '''
    :param img_path: input image path (*.tif)
    :param building_masks: mask-rcnn predict result masks
    :param road_masks: unet predict result masks
    :param save_path: save path
    :param uint_pixel: crop pixel size
    :return None
    '''

    size = unit_pixel
    img_name = os.path.basename(img_path).split(".")[0]
    img = np.asarray(Image.open(img_path).convert("RGB"))
    
    if img.shape[0] < size[0] and img.shape[1] < size[1]:
        mask = Image.fromarray(road_masks[0] + building_masks[0])
        mask.save(os.path.join(save_path, img_name + '_mask.png'))
        return None

    mask_save = np.zeros_like(img)
    blend = np.zeros_like(img)
    x_pos, y_pos = 0, 0

    for building_mask, road_mask in zip(building_masks, road_masks):
        building = cv2.cvtColor(building_mask, cv2.COLOR_BGR2RGB)
        mask_np = np.asarray(Image.fromarray(road_mask + building))
        size = building_mask.shape[:2]  # h, w
        mask_save[y_pos : y_pos + size[0], x_pos : x_pos + size[1]] = mask_np

        x_pos += size[1]

        if x_pos == img.shape[1]:
            y_pos += size[0]
            x_pos = 0

    mask_save_pil = Image.fromarray(mask_save)
    mask_save_pil.save(os.path.join(save_path + img_name + '_mask.png'))

    blend = cv2.addWeighted(mask_save, 0.4, img, 0.6, gamma=0)
    Image.fromarray(blend).save(os.path.join(save_path + img_name + '_blend.png'))

