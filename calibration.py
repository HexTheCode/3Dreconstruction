import numpy as np
import cv2 as cv
import glob

# Chessboard inner corners
coners_size = (7,7)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print("Criterio Cargado...")
print(criteria)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((coners_size[0]*coners_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:coners_size[1],0:coners_size[0]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('dataset/calibratio_imgs/*.jpg')
print("Imagenes cargadas... "+str(len(images)))
img = None
corners2 = None
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, coners_size, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        break
       
    else:
        print("Error no se han podido identificar las esquinas de "+fname)


ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Guardando matriz K:")
print(f"dataset/matrices/K_matrix_{fname[0]}.npy")
np.save(f"dataset/matrices/K_matrix_{fname[0]}.npy",K)

print(K)
print(rvecs)
print(tvecs)
print(dist)

 # Draw and display the corners
cv.drawChessboardCorners(img, coners_size, corners2, True)
cv.imshow('img', img)
cv.waitKey(10000)
