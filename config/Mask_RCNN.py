############################################
cfg = {
    "weights": 'C:/Company/AI-Detection/Lastmile-AI/weight/ep200_building_best_weights.h5',
    "classes": ["building", "vinyl_house"],
    "conf": 0.5,
    "nms_iou": 0.3,
    "max_channels": 1024,
    "rpn_anchors": [32, 64, 128, 256, 512],
}


############################################
# COLORS
############################################
colors = [[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183],
          [100, 30, 60], [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212],
          [20, 55, 200], [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57],
          [70, 25, 100], [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34],
          [90, 155, 50], [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34],
          [98, 55, 20], [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134],
          [90, 125, 120], [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234],
          [8, 155, 220], [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64],
          [198, 75, 20], [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144],
          [78, 155, 120], [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134],
          [18, 185, 90], [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84],
          [130, 115, 170], [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234],
          [18, 25, 190], [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
          [155, 0, 0], [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155],
          [155, 0, 255], [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155],
          [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244],
          [128, 25, 70]]
