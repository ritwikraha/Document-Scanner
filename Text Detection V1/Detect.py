import cv2
import TD

img = cv2.imread('test.jpg')
rect = TD.text_detect(img)
for i in rect:
    cv2.rectangle(img,i[:2],i[2:],(0,255,0))
cv2.imwrite('img-out.png', img)