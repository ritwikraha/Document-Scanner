import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank


def thresho(imgj):
	kernel = np.ones((2,2),np.uint8)
	gradient = cv2.morphologyEx(imgj, cv2.MORPH_GRADIENT, kernel)
	edges_inv = cv2.bitwise_not(gradient)
	th3 = cv2.adaptiveThreshold(edges_inv,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,9,6)
	return th3