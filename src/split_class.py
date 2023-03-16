import cv2
import numpy as np

from config.common import area_threshold




def format_coor(contours, position, image_shape):
    '''
    :param contours: road contours
    :param position: crop input masks position (ie:[[0,0],[1,0]...])
    :param image_shape: input road mask's shape
    :return road predicat result xy coordinate from original image's coordinate
    '''
    position_x, position_y = position[1], position[0]
    coor = []
    position_coor = []
    for i in range(len(contours)):
        if contours[i] is None:
            i += 1 
        else:
            xy_coor = [[[x, y] for [x, y] in contour.reshape(-1, 2)] for contour in contours[i]]
            
            x = np.array(xy_coor)[:,0,0] + position_x * image_shape[0]
            y = np.array(xy_coor)[:,0,1] + position_y * image_shape[1]
            output = [val for pair in zip(x,y) for val in pair]

            # coor.append(output)
        
            # check_output = np.array(coor).reshape(len(coor[0])//2,1,2)
            check_output = np.array(output).reshape(len(output)//2,1,2)
            position_coor.append(check_output)

    return position_coor
    



def split_class(pred, label_idx, colors, position):
    """
        
    first, split each class in predicted image, and then split each object in the same class and save to dictionary for each.

    :param pred: image predicted by AI 
    :param label_idx: indexes for class name of object
    :param colors: colors for class name of object 
    :return: list of polygon for object split by each color 
    """
    

    divided_obj = []
    # should split the image for each color, so iterate using colors
    for i in range(len(colors)):
        color = np.array(colors[i])
        # in predicted image, search index same as the color
        mask = (pred == color).all(-1)

        # using searched index, remove incorrect color and extract only object with correct color
        leaf = np.where(mask[..., None], color, [0, 0, 0]).astype(np.uint8)

        # convert the BGR to grayscale because of extracting contour of object
        gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)

        # preprocessing
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # find contours in gray mask.
        # findcontours(image, 검색방법, method of approximation)
        # TREE hierarchy = [Next, Previous, First inner contour, First outer contour]
        # CCOMP hierarchy = [Next, Previous, inner contour, outer contour]
        contours, hierarchy = cv2.findContours(gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_L1)

        # format contours' coordinates, xy coordinate is from original image's coordiante
        contours = format_coor(contours, position, pred.shape)

        if len(contours) > 0 and hierarchy is not None:
            latent = np.zeros((1, len(hierarchy[0])), dtype=np.int8)
            latent[0][np.unique(hierarchy[:, :, 2])[1:]] = 1  # interior association

            # iteration with found contours
            for c in range(len(contours)):
                # too much small area is removed
                area = cv2.contourArea(contours[c])
                if area < area_threshold:
                    continue

                if latent[0, c] > 0:
                    continue

                if hierarchy[0, c, 3] != -1:
                    divided_obj[-1]["contour"].append(contours[c].tolist())
                    continue

                divided_obj.append({})
                # draw object filled. redraw in image because we should save to split object.
                # drawContours(image, contours, index for contours, color, thickness, line_type)
                # out_mask = np.zeros((h, w))
                # output = cv2.drawContours(out_mask, contours, c, 255, cv2.FILLED)

                output = contours[c].reshape(1, -1, 2).tolist()

                # inner
                if hierarchy[0, c, 2] != -1:
                    idx = hierarchy[0, c, 2]
                    output.append(contours[idx].tolist())
                    # output = cv2.drawContours(output, contours, idx, 0, cv2.FILLED)

                divided_obj[-1]["id"] = label_idx[i+2]
                divided_obj[-1]["contour"] = output

    return divided_obj

if __name__ == "__main__":
    pred = cv2.imread("../pred.png", cv2.IMREAD_COLOR)
    split_class(pred, label_idx=[1,2,3,4], colors=[[54, 67, 244], [99, 30, 233], [215, 64, 64], [124, 215, 30]])
