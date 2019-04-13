import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils.video import WebcamVideoStream


vs = WebcamVideoStream(0).start()
while True:
    frame = vs.read()
    cv2.imshow("title",frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    
    
    #Blur image
    blur = cv2.GaussianBlur(frame,(7,7),0)

    #HSV Threshoold
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #LOWER RED MASK
    lower_red = np.array([0,50,50])
    upper_red = np.array([16,255,255])
    mask0 = cv2.inRange(hsv,lower_red,upper_red)
    #UPPER RED MASK
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)
    mask_red = cv2.bitwise_or(mask0,mask1)
    #bitwise red
    eroded = cv2.erode(mask_red, None, iterations=3)
    dilated2 = cv2.dilate(eroded,None, iterations=3)
    masked_red = cv2.bitwise_and(blur,blur,mask=dilated2)

    #GREEN MASK
    lower_green = np.array([32,96,9])
    upper_green = np.array([89,237,250])
    mask_green = cv2.inRange(hsv,lower_green,upper_green)
    #bitwise green
    eroded = cv2.erode(mask_green, None, iterations=3)
    dilated3 = cv2.dilate(eroded,None, iterations=3)
    masked_green = cv2.bitwise_and(blur,blur,mask=dilated3)
    
    #bitwise blue
    #bitwise yellow

    
    #contours detect
    gray = cv2.cvtColor(masked_red, cv2.COLOR_BGR2GRAY) #switch for masked input
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts)>0:
        gray_Green = cv2.cvtColor(masked_green, cv2.COLOR_BGR2GRAY)
        edged_Green = cv2.Canny(gray_Green,50,100)
        edged_Green = cv2.dilate(edged_Green, None, iterations=1)
        edged_Green = cv2.erode(edged_Green,None, iterations=1)
        cnts_Green = cv2.findContours(edged_Green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_Green = imutils.grab_contours(cnts_Green)
    
        for c in cnts:
            #filter out the high up ball, take bottom 80%
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints (box)
            box = np.array(box, dtype = "int")
            dh = dist.euclidean((box[0, 0], box[0 ,1]), (box [1, 0], box [1, 1]))
            dw = dist.euclidean( (box[1, 0], box[ 1, 1]), (box[2, 0 ], box[2, 1]))
            acceptableY = dh
            acceptableX = dw
            if dw !=0:
                ratio = int(dh)/ int (dw)
            else:
                ratio = 0
            
            #filter out the non sq ones
            if abs(ratio - 1) < 0.2:
                    cX = np.average(box[:,0])
                    cY = np.average(box[:,1])
                    for g in cnts_Green:
                        box_Green = cv2.minAreaRect(g)
                        box_Green = cv2.boxPoints(box_Green)
                        cX_g = np.average(box_Green[:,0])
                        cY_g = np.average(box_Green[:,1])
                        ydist = cY - cY_g
                        xdist = cX - cX_g

                        if abs(ydist) <= acceptableY:
                            if abs(xdist) <= acceptableX:
                                acceptableX = abs(xdist)
                                acceptableY = abs(ydist)
                                Dh_Green = dist.euclidean((box_Green[0, 0], box_Green[0 ,1]), (box_Green [1, 0], box_Green [1, 1]))
                                if abs(Dh_Green) < 14:
                                    cv2.drawContours(frame, [box_Green.astype("int")],-1,[0,255,0],1)
                                    cv2.drawContours(frame, [box.astype("int")],-1,(0,255,0),2)
                        
                    
                    cv2.imshow("edgey",edged)

    cv2.imshow("maskedred",masked_red)
    cv2.imshow("maskedgreen",masked_green)
cv2.destroyAllWindows()
vs.stop()
