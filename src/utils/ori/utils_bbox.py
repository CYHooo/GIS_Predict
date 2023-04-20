<<<<<<< HEAD
import numpy as np
from .utils import _resize
import datetime

# -----------------------------------------#
#   将语义分割结果映射到原图上
# -----------------------------------------#
def unmold_mask(mask, bbox, image_shape):
    y1, x1, y2, x2 = bbox
    mask = _resize(mask, (y2 - y1, x2 - x1))

    full_mask = np.zeros(image_shape[:2], dtype=np.float32)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


# -----------------------------------------#
#   将框的位置信息进行标准化限制在0-1之间
# -----------------------------------------#
def norm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


# -----------------------------------------#
#   将标准化后的框再调整回去
# -----------------------------------------#
def denorm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def postprocess(detections, mrcnn_mask, image_shape, input_shape, window):
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   消除预测结果中的padding部分
    # -----------------------------------------#
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    if N == 0:
        return None, None, None, None, None
    print(f"消除padding时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   框的坐标，物品种类，得分
    # -----------------------------------------#
    box_thres = detections[:N, :4]
    class_thres = detections[:N, 5]
    class_ids = detections[:N, 4].astype(np.int16)
    print(f"框坐标时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   取出分割结果
    # -----------------------------------------#
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]
    print(f"取出分割结果时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   删掉背景的部分
    # -----------------------------------------#
    class_ids = class_ids - 1

    # -----------------------------------------#
    #   获得window框的小数形式
    # -----------------------------------------#
    wy1, wx1, wy2, wx2 = norm_boxes(window, input_shape[:2])
    wh = wy2 - wy1
    ww = wx2 - wx1
    print(f"获得window框时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # --------------------------------------------#
    #   将框的坐标进行调整，调整为相对于原图的
    # --------------------------------------------#
    shift = np.array([wy1, wx1, wy1, wx1])
    scale = np.array([wh, ww, wh, ww])
    box_thres = np.divide(box_thres - shift, scale)
    box_thres = denorm_boxes(box_thres, image_shape[:2])
    print(f"框坐标调整时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # ----------------------#
    #   将错误框剔除掉
    # ----------------------#
    exclude_ix = np.where((box_thres[:, 2] - box_thres[:, 0]) * (box_thres[:, 3] - box_thres[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        box_thres = np.delete(box_thres, exclude_ix, axis=0)
        class_thres = np.delete(class_thres, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]
    print(f"框剔除错误框时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   将语义分割结果映射到原图上
    # -----------------------------------------#
    masks_sigmoid = []
    for i in range(N):
        full_mask = unmold_mask(masks[i], box_thres[i], image_shape)
        masks_sigmoid.append(full_mask)

    # TODO: 해당 부분이 시간 상당 부분 소요. -> RAM 부족으로 인한 시간 소요인 것으로 판단.
    masks_sigmoid = np.stack(masks_sigmoid, axis=-1) if masks_sigmoid else np.empty(image_shape[:2] + (0,))
    print(f"预测结果映射时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # ----------------------------------------------------------------------#
    #   masks_arg   [image_shape[0], image_shape[1]]    
    #   获得每个像素点所属的实例
    # ----------------------------------------------------------------------#
    masks_arg = np.argmax(masks_sigmoid, axis=-1)
    print(f"每个像素点时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # ----------------------------------------------------------------------#
    #   masks_arg   [image_shape[0], image_shape[1], num_of_kept_boxes]
    #   判断每个像素点是否满足门限需求
    # ----------------------------------------------------------------------#
    masks_sigmoid = masks_sigmoid > 0.5
    print(f"判断时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    return box_thres, class_thres, class_ids, masks_arg, masks_sigmoid
=======
import numpy as np
from .utils import _resize
import datetime

# -----------------------------------------#
#   将语义分割结果映射到原图上
# -----------------------------------------#
def unmold_mask(mask, bbox, image_shape):
    y1, x1, y2, x2 = bbox
    mask = _resize(mask, (y2 - y1, x2 - x1))

    full_mask = np.zeros(image_shape[:2], dtype=np.float32)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


# -----------------------------------------#
#   将框的位置信息进行标准化限制在0-1之间
# -----------------------------------------#
def norm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


# -----------------------------------------#
#   将标准化后的框再调整回去
# -----------------------------------------#
def denorm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def postprocess(detections, mrcnn_mask, image_shape, input_shape, window):
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   消除预测结果中的padding部分
    # -----------------------------------------#
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    if N == 0:
        return None, None, None, None, None
    print(f"消除padding时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   框的坐标，物品种类，得分
    # -----------------------------------------#
    box_thres = detections[:N, :4]
    class_thres = detections[:N, 5]
    class_ids = detections[:N, 4].astype(np.int16)
    print(f"框坐标时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   取出分割结果
    # -----------------------------------------#
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]
    print(f"取出分割结果时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   删掉背景的部分
    # -----------------------------------------#
    class_ids = class_ids - 1

    # -----------------------------------------#
    #   获得window框的小数形式
    # -----------------------------------------#
    wy1, wx1, wy2, wx2 = norm_boxes(window, input_shape[:2])
    wh = wy2 - wy1
    ww = wx2 - wx1
    print(f"获得window框时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # --------------------------------------------#
    #   将框的坐标进行调整，调整为相对于原图的
    # --------------------------------------------#
    shift = np.array([wy1, wx1, wy1, wx1])
    scale = np.array([wh, ww, wh, ww])
    box_thres = np.divide(box_thres - shift, scale)
    box_thres = denorm_boxes(box_thres, image_shape[:2])
    print(f"框坐标调整时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # ----------------------#
    #   将错误框剔除掉
    # ----------------------#
    exclude_ix = np.where((box_thres[:, 2] - box_thres[:, 0]) * (box_thres[:, 3] - box_thres[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        box_thres = np.delete(box_thres, exclude_ix, axis=0)
        class_thres = np.delete(class_thres, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]
    print(f"框剔除错误框时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # -----------------------------------------#
    #   将语义分割结果映射到原图上
    # -----------------------------------------#
    masks_sigmoid = []
    for i in range(N):
        full_mask = unmold_mask(masks[i], box_thres[i], image_shape)
        masks_sigmoid.append(full_mask)

    # TODO: 해당 부분이 시간 상당 부분 소요. -> RAM 부족으로 인한 시간 소요인 것으로 판단.
    masks_sigmoid = np.stack(masks_sigmoid, axis=-1) if masks_sigmoid else np.empty(image_shape[:2] + (0,))
    print(f"预测结果映射时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # ----------------------------------------------------------------------#
    #   masks_arg   [image_shape[0], image_shape[1]]    
    #   获得每个像素点所属的实例
    # ----------------------------------------------------------------------#
    masks_arg = np.argmax(masks_sigmoid, axis=-1)
    print(f"每个像素点时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    # ----------------------------------------------------------------------#
    #   masks_arg   [image_shape[0], image_shape[1], num_of_kept_boxes]
    #   判断每个像素点是否满足门限需求
    # ----------------------------------------------------------------------#
    masks_sigmoid = masks_sigmoid > 0.5
    print(f"判断时间 : {(datetime.datetime.now() - T).seconds} s")
    T = datetime.datetime.now()
    return box_thres, class_thres, class_ids, masks_arg, masks_sigmoid
>>>>>>> d59dd7db66de6d72c42a79238c248c5b82f81626
