import numpy as np
import cv2
import edged

def text_detect(img,ele_size=(8,3)): #choose the restructuring element size as 8X3
    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# convert to grayscale if in RGB
    img_edged = cv2.Sobel(img,cv2.CV_8U,1,0)
    img_threshold = cv2.threshold(img_edged,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT,ele_size)
    img_threshold = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
    contours = cv2.findContours(img_threshold,0,1)
    Rect = [cv2.boundingRect(i) for i in contours[1] if i.shape[0]>100]
    RectP = [(int(i[0]-i[2]*0.08),int(i[1]-i[3]*0.08),int(i[0]+i[2]*1.1),int(i[1]+i[3]*1.1)) for i in Rect]
    return RectP
