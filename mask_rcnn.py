import colorsys
import os

import cv2
import numpy as np
from PIL import Image

from src.nets.mrcnn import get_predict_model
from src.utils.anchors import get_anchors
from src.utils.config import Config
from src.utils.utils import cvtColor, resize_image
from src.utils.utils_bbox import postprocess
from src.cocojson.pycococreatortools.pycococreatortools import resize_binary_mask, binary_mask_to_polygon

from config.Mask_RCNN import cfg, colors

class MASK_RCNN(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        # Use your own training models to predict, be sure to modify the Model_path and Classes_path!
        # MODEL_PATH points to the right -value file under the LOGS folder, Classes_Path points to the txt under the Model_data
        #
        # After training, there are multiple rights files under the LOGS folder, and you can select the lower loss of the set.
        # The lower loss of verification set does not mean that the MAP is higher, and it only means that the value of the right value is better on the verification set.
        # If the shape does not match, pay attention to the modification of the model_path and class_path parameters during training
        # --------------------------------------------------------------------------#
        "model_path": cfg["weights"],
        "class_names": cfg["classes"],
        # ---------------------------------------------------------------------#
        #   Only the prediction box with greater scores will be retained
        # ---------------------------------------------------------------------#
        "confidence": cfg["conf"],
        # ---------------------------------------------------------------------#
        #   Number of NMS_iou size used
        # ---------------------------------------------------------------------#
        "nms_iou": cfg["nms_iou"],
        # ----------------------------------------------------------------------#
        #   The input shape size
        # Algorithm will fill in the size of the input picture to the size of [Image_max_dim, Image_max_dim]
        # ----------------------------------------------------------------------#
        "IMAGE_MAX_DIM": cfg["max_channels"],
        # ----------------------------------------------------------------------#
        #   It is used to set the size of the first test box. In most cases the default priority box, it is universal and can not be modified.
        # Need to be consistent with the RPN_ANCHOR_SCALES of the training settings.
        # ----------------------------------------------------------------------#
        "RPN_ANCHOR_SCALES": cfg["rpn_anchors"]
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Initialize Mask-RCNN
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   Calculate the total number of the total class
        # ---------------------------------------------------#
        self.num_classes = len(self.class_names) + 1

        # ---------------------------------------------------#
        #   Set different colors of the frame
        # ---------------------------------------------------#
        if self.num_classes <= 81:
            self.colors = np.array(colors, dtype='uint8')
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        class InferenceConfig(Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = self.num_classes
            DETECTION_MIN_CONFIDENCE = self.confidence
            DETECTION_NMS_THRESHOLD = self.nms_iou
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM

        self.config = InferenceConfig()
        self.generate()

    # ---------------------------------------------------#
    #   Generate
    # ---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # -----------------------#
        #   Loading model
        # -----------------------#
        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path, by_name=True)

    # ---------------------------------------------------#
    #   test
    # ---------------------------------------------------#

            

    def detect_image(self, image, position):

        '''@23/03/12 Yeongho:
            :param image: input crop image
            :param position: crop input masks position (ie:[[0,0],[1,0]...])

            !! mask-rcnn.py can not save building_mask.png, need debugging !!
        '''
        
        def _format_coor(masks_coor, position, masks_shape):
            '''
            @23/03/11 Yeongho:
            :param masks_coor: crop predict mask info (type: np.bool_)
            :param positon: position from original image
            :param masks_shape: input mask's shape
            :return: building predcit result xy coordinate from original image's coordinate
            '''
            position_x, position_y = position[1], position[0]
            position_coor = []
            
            if masks_coor is None:
                return None
            
            for i in range(masks_coor.shape[-1]):
                crop_coor = masks_coor[:,:,i].astype('uint8')
                crop_coor = resize_binary_mask(crop_coor,masks_shape)
                crop_coor = binary_mask_to_polygon(crop_coor, tolerance=2)

                coor = [[crop_coor[0][a+1], crop_coor[0][a]] for a in range(0,len(crop_coor[0]),2)]

                x = np.array(coor)[:,1] + position_x * image_shape[1]
                y = np.array(coor)[:,0] + position_y * image_shape[0]
                output = [val for pair in zip(x,y) for val in pair]

                position_coor.append(output)

            return  position_coor
        
        ##------------ move color info at first ------------##
        target_ids = [1, 2]
        colors = [self.colors[t_id][::-1].tolist() for t_id in target_ids]
        ##--------------------------------------------------##

        image_shape = image.size[::-1]

        image = cvtColor(image)
        image_origin = np.array(image, np.uint8)

        image_data, image_metas, windows = resize_image([np.array(image)], self.config)

        anchors = np.expand_dims(get_anchors(self.config, image_data[0].shape), 0)

        detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)
        # ---------------------------------------------------#
        #   The prediction result obtained above is a picture after padding
        #   We need to convert the prediction results to the original picture
        # ---------------------------------------------------#
        box_thres, class_thres, class_ids, masks_arg, masks_sigmoid = postprocess(detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0])
        masks_save = masks_sigmoid

        masks_sigmoid = _format_coor(masks_sigmoid, position, image_shape)

        # np.save(r'D:\msrcnn\AI_code_230308\AI-Detection\img\02_05_TEST\save.npy', masks_sigmoid)
        rcnn_info = [
                box_thres, 
                class_thres, 
                class_ids, 
                masks_arg, 
                masks_sigmoid
            ]

        if box_thres is None:
            non_class_mask = np.zeros_like(image)

            return None, non_class_mask, colors, rcnn_info
        
        
        # ----------------------------------------------------------------------#
        #   masks_class [image_shape[0], image_shape[1]]
        #   Determine the types of each pixel based on the instances of each pixel and whether they meet the needs of the clusive limit.
        # ----------------------------------------------------------------------#
        masks_class = masks_save * (class_ids[None, None, :] + 1)
        masks_class = masks_class.reshape(-1, masks_class.shape[-1])
        masks_class = np.reshape(masks_class[np.arange(masks_class.shape[0]), np.reshape(masks_arg, [-1])], [image_shape[0], image_shape[1]]) ## TODO: ????
        

        # 0 index = background(black)
        filtered_mask_class = np.where(np.array([(masks_class[np.newaxis, :, :] == t) for t in target_ids]).T.any(-1).T, masks_class, 0)[0]
        filtered_mask_class = filtered_mask_class.astype(int)
        color_masks = self.colors[filtered_mask_class].astype('uint8') # h,w,c

        # ---------------------------------------------------------#
        #   获取目标分类
        # ---------------------------------------------------------#

        BBOX_EDA = False

        
        if not BBOX_EDA:
            # cv2.imwrite(save_path + '/building_mask.png', color_masks)

            return None, color_masks, colors, rcnn_info

        else:
            scale = 0.6
            thickness = int(max((image.size[0] + image.size[1]) // self.IMAGE_MAX_DIM, 1))
            font = cv2.FONT_HERSHEY_DUPLEX

            # 원본에 영역 흐리게 표시한 image
            image_fused = cv2.addWeighted(color_masks, 0.4, image_origin, 0.6, gamma=0)

            for i in range(np.shape(class_ids)[0]):
                if not class_ids[i] in target_ids:
                    continue

                #     masks_class     = masks_sigmoid[i] * (class_ids[None, None, :] + 1) 
                #     masks_class     = np.reshape(masks_class, [-1, np.shape(masks_sigmoid[i])[-1]])
                #     masks_class     = np.reshape(masks_class[np.arange(np.shape(masks_class)[0]), np.reshape(masks_arg[i], [-1])], [image_shape[0], image_shape[1]])

                top, left, bottom, right = np.array(box_thres[i, :], np.int32)

                # ---------------------------------------------------------#
                #   Get color and draw prediction box
                # ---------------------------------------------------------#
                cv2.rectangle(image_fused, (left, top), (right, bottom), colors, thickness)

                # ---------------------------------------------------------#
                #   Get the type of this frame and write it on the picture
                # ---------------------------------------------------------#
                class_name = self.class_names[class_ids[i]]
                # print(class_name, top, left, bottom, right)
                text_str = f'{class_name}: {class_thres[i]:.2f}'
                text_w, text_h = cv2.getTextSize(text_str, font, scale, 1)[0]
                cv2.rectangle(image_fused, (left, top), (left + text_w, top + text_h + 5), colors, -1)
                cv2.putText(image_fused, text_str, (left, top + 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

            # cv2.imwrite(save_path + img_name + '_bbox.png', image_fused)

            return np.uint8(image_fused), color_masks, colors, rcnn_info
