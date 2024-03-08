import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
#Matplot is RGB, opencv is BGR

img=cv.imread('Resources/Photos/cats.jpg')
cv.imshow('cats',img)
rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('RGB',rgb)
plt.imshow(rgb)
plt.show()

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray )

#BGR to HSV

hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('HSV',hsv)

#BGR to L*a*b
lab=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('LAB',lab)

#Covert HSV TO BGR
hsv_bgr=cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
cv.imshow("hsv_bgr",hsv_bgr)

lab_bgr=cv.cvtColor(hsv,cv.COLOR_LAB2BGR)
cv.imshow("LAB------BGR",lab_bgr)

cv.waitKey(0)
