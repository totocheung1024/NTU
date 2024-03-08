import cv2 as cv

img=cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('GRay',gray)
#Simple Thresholding

threshold,thresh=cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow('Simple Threshold',thresh)
#Value>150, become 255

threshold,thresh_inv=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
#Value <155, become 255
cv.imshow('Simple Threshold Inverse',thresh_inv)

#Adaptive Thresholding
adaptive_thresh=cv.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv.THRESH_BINARY_INV, blockSize=13, C=1)
cv.imshow('Adaptive Thresholding',adaptive_thresh)
#Compute the mean within a kernal, and select the appropriate threshold

cv.waitKey(0)