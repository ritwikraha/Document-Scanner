import cv2
import numpy as np
import resizer
import edged
import masker


image = cv2.imread('test.jpg')
kernel = np.ones((1,1),np.uint8)

# resize image so it can be processed
# choose optimal dimensions such that important content is not lost
image = cv2.resize(image, (800, 1200))

# creating copy of original image
orig = image.copy()

# convert to grayscale and blur to smooth
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
#blurred = cv2.medianBlur(gray, 5)

# apply Edge Detection algorithm developed from morphological gradient filter
edgedetect = edged.thresho(cv2.bitwise_not(blurred))
orig_edged = edgedetect.copy()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
image1, contours, hierarchy = cv2.findContours(edgedetect, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)



# get approximate contour
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break


dst = resizer.rectify(orig,target)

dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


# using thresholding on warped image to get scanned effect (If Required)

th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,7,4)
ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)

th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,4)
ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
dst = cv2.bitwise_not((dst > th1).astype("uint8") * 255)


cv2.imwrite("mode_dark.jpg", th1)
cv2.imwrite("mode_mean.jpg", th2)
cv2.imwrite("mode_gaussian.jpg", th3)
cv2.imwrite("mode_magic.jpg", th4)
cv2.imwrite("mode_magic_cropped.jpg", dst)



