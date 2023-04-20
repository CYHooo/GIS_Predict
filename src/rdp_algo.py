from rdp import rdp
import numpy as np

def rdp_algo(polygon_points, epsilon=50):
    """
    rdp algorithm is also organization the surroundings of objects but, this is arrangement from the perspective of polygons.

    :param polygon_points: polygon to be optimized by rdp algorithm.
    :param epsilon: parameter for how much you want to optimize the polygon
    :return: polygon optimized by rdp
    """
    # input polygon shape is linear, so make the coordinate to a couple of that.
    return [rdp(np.array(polygon_points[i]), epsilon).ravel().tolist() for i in range(len(polygon_points))]
