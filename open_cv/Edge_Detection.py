import cv2 as cv
import numpy as np


img=cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats',img)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow('Laplacian',lap)#Like a pencil drawing

#Sobel Compute gradient into direction (X and Y)

sobelx=cv.Sobel(gray,cv.CV_64F, 1,0)
sobely=cv.Sobel(gray,cv.CV_64F, 0,1)
combinded_sobel=cv.bitwise_or(sobelx,sobely)


cv.imshow('Sobel X',sobelx)
cv.imshow('Sobel Y',sobely)
cv.imshow('SobelXY',combinded_sobel)

canny=cv.Canny(gray,150,175)
cv.imshow('Canny',canny)

cv.waitKey(0)



# Laplaction