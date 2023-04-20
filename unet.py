import colorsys
import copy
import time
import os

import cv2
import numpy as np
from PIL import Image

from src.nets.unet import Unet as unet
from src.utils.unet_utils import cvtColor, preprocess_input, resize_image, show_config

from config.UNet import cfg, colors

# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
# --------------------------------------------#
class Unet(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   model path
        # -------------------------------------------------------------------#
        "model_path": cfg["weights"],
        # ----------------------------------------#
        #   class num (class + 1)
        # ----------------------------------------#
        "classes": cfg["classes"],
        # ----------------------------------------#
        #   backbone network：vgg(x) or resnet50   
        # ----------------------------------------#
        "backbone": cfg["backbone"],
        # ----------------------------------------#
        #   input size
        # ----------------------------------------#
        "input_shape": cfg["input_shape"],
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.num_classes = len(self.classes) + 1
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = np.array(colors, dtype=np.uint8)
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        self.generate()

        # show_config(**self._defaults)

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.model = unet([self.input_shape[0], self.input_shape[1], 3], self.num_classes, self.backbone)
        self.model.load_weights(self.model_path)

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, rcnn_mask):
        '''
        input:
            image        : original img
            image_name   : img name
            save_path    : save path
            rcnn_mask    : R-cnn predict mask (type: np.uint8, color: BGR)

        return: 
            blend_image  : R-cnn & Unet predict mask cover to original img
            mask         : R-cnn & Unet predict mask to one mask img
            road_mask    : Unet predict road mask img (type: np.uint8, color: RGB) 
                        < save usage > 
                        Image.fromarray(road_mask).save(os.path.join(savepath + 'filename.png'))
            color        : BGR color
        '''

        '''
        @23/03/12 Yeongho:
        Unet.py can not save predict mask, need debugging.
    
        :param image: input crop image
        :param rcnn_mask: crop mask's R-cnn predict mask (type: np.uint8, color: BGR)

        !! mask-rcnn.py can not save building_mask.png, need debugging !!

        '''
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   归一化+添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        pr = self.model.predict(image_data, verbose=0)[0]

        # ---------------------------------------------------#
        #   将灰条部分截取掉
        # ---------------------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
             int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   resize for pictures
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

        # ---------------------------------------------------#
        #   Get type of each pixel
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)

        target_ids = [1, 2]

        color = [list(self.colors[t_id]) for t_id in target_ids]

        seg_img = self.colors[pr]

        # ------------------------------------------------#
        #   将新图片转换成Image的形式
        # ------------------------------------------------#

        road_mask = np.uint8(seg_img)
        rcnn_mask = cv2.cvtColor(rcnn_mask, cv2.COLOR_BGR2RGB)  ## rcnn_mask cv-BGR to PIL-RGB

        mask = Image.fromarray(road_mask + rcnn_mask)
        # blend_image = Image.blend(old_img, mask, 0.5)

        # mask.save(os.path.join(save_path, img_name + '_mask.png'))
        # blend_image.save(os.path.join(save_path, img_name + '_blend.png'))

        ## for save road mask
        # Image.fromarray(road_mask).save(os.path.join(save_path,image_name + '_road.png'))

        return np.asarray(mask), road_mask, color
