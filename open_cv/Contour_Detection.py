import cv2 as cv 
import numpy as np
img=cv.imread('Resources/Photos/cats.jpg')

#Draw edge have 2 methods, lik

# cv.imshow('Cats', img)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)

canny=cv.Canny(img,0,250)
#canny is used fo edge detection, (img, low_threshold, high_threshold)

cv.imshow('Cany Edges', canny)
#Contour=輪廓

contours, hierarchies= cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(hierarchies)} counters(s) founds')
# cv.findContours (canny, retrieval model=cv.RETR_LIST refers to try to retrieves all the contours without grouping them tgt
#, )Contour Approximation Method stands for storing all of the contour points
contours, hierarchies= cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.RETR_TREE refers to storing all the contours into  tree format, and  cv.CHAIN_APPROX_SIMPLE stands for only store
# the less sample points , like start and end point 
print(f'{len(hierarchies)} counters(s) founds')



blur=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)

ret,thresh= cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('Thresh',thresh)
#Binary = black or white
contours,hierarchies=cv.findContours(thresh,cv.RETR_LIST,cv.
                                     CHAIN_APPROX_SIMPLE)
print(f'{len(contours)}conouts(s) found')

blank=np.zeros(img.shape,dtype='uint8')
cv.imshow('Blank',blank)

cv.drawContours(blank,contours,-1,(0,0,255),2)
#Only highlight the contours in red
cv.imshow('Contours Drawn',blank)


cv.waitKey(0)