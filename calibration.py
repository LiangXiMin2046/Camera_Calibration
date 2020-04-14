'''
camera calibration
'''

import glob
from cv2 import cv2 
import numpy as np 
import os
#设置寻找角点的参数 最大迭代30步 最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

#设置标定板角点位置 令z = 0
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

obj_points = [] #存储二维点
img_points = [] #存储三维点

#读取图片
images_path = 'images/*.jpg'
images = glob.glob(images_path)
i = 0

#开始标记
draw_path = 'draw/' #标记后的图片保存目录
if not os.path.exists(draw_path):
    os.mkdir(draw_path)
for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图片
    size = gray.shape[::-1]
    ret,corners = cv2.findChessboardCorners(gray,(9,6),None)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        cv2.drawChessboardCorners(img,(9,6),corners,ret)
        i += 1
        cv2.imwrite(draw_path+str(i)+'.jpg' ,img)
        cv2.waitKey(0)

#结果
print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数


