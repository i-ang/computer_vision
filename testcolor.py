# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# load the image, convert BGR to HSV
#img = cv2.VideoCapture(0)
#img = cv2.imread("bo.jpg")
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

#upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv,lower_red,upper_red)

#joining the masks
mask = mask0 + mask1

#setting output img to zero everywhere except mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or hsv image
output_hsv = hsv.copy()
output_hsv[np.where(mask==0)] = 0

cv2.imshow('output',output_img)
cv2.imshow('hsv',output_hsv)
