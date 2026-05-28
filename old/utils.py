import numpy as np
import numpy as np


def find_correspondence_points(img1, img2):
    """
    Find correspondence points among two images, img1 and img2. Use SIFT method to
    find keypoints and then apply Lowe's SIFT matching ratio test to return pair of points
    that are homologs in the two images.
    :param img1: one of the two image to find correspondence
    :param img2: the other image to find correspondence
    :return: Tuple (pts1, pts2) of coordinates for the homologs points from the two images
    """
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # Find point matches
    # FLANN: Fast Library for Approximate Nearest Neighbors
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2) #Usando los k vecinos más cercanos k=2

    # Apply Lowe's SIFT matching ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T


def read_matrix(path, astype=np.float64):
    """ Reads a file containing a matrix where each line represents a point
        and each point is tab or space separated. * are replaced with -1.
    :param path: path to the file
    :parama astype: type to cast the numbers. Default: np.float64
    :returns: array of array of numbers
    """
    with open(path, 'r') as f:
        arr = []
        for line in f:
            arr.append([(token if token != '*' else -1)
                        for token in line.strip().split()])
        return np.asarray(arr).astype(astype)


def cart2hom(arr):
    """ Convert catesian to homogenous points by appending a row of 1s
    :param arr: array of shape (num_dimension x num_points)
    :returns: array of shape ((num_dimension+1) x num_points) 
    """
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))


def hom2cart(arr):
    """ Convert homogenous to catesian by dividing each row by the last row
    :param arr: array of shape (num_dimension x num_points)
    :returns: array of shape ((num_dimension-1) x num_points) iff d > 1 
    """
    # arr has shape: dimensions x num_points
    num_rows = len(arr)
    if num_rows == 1 or arr.ndim == 1:
        return arr

    return np.asarray(arr[:num_rows - 1] / arr[num_rows - 1])

def rotation_3d_from_angles(x_angle, y_angle=0, z_angle=0):
    """ Creates a 3D rotation matrix given angles in degrees.
        Positive angles rotates anti-clockwise.
    :params x_angle, y_angle, z_angle: x, y, z angles between 0 to 360
    :returns: 3x3 rotation matrix """
    ax = np.deg2rad(x_angle)
    ay = np.deg2rad(y_angle)
    az = np.deg2rad(z_angle)

    # Rotation matrix around x-axis
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)]
    ])
    # Rotation matrix around y-axis
    ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ])
    # Rotation matrix around z-axis
    rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1]
    ])

    return np.dot(np.dot(rx, ry), rz)