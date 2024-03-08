import cv2 as cv
import numpy as np

img= cv.imread('Resources/Photos/cats 2.jpg')
# cv.imshow('Cat', img)

blank=np.zeros(img.shape[:2],dtype='uint8')
# cv.imshow('Blank Image',blank)

mask=cv.circle(blank,(img.shape[1]//2+45,img.shape[0]//2),100,255,-1)
# cv.imshow('Masked',mask)

masked=cv.bitwise_and(img,img,mask=mask)
# cv.imshow('Masked Image',masked)

masked=cv.rectangle(blank,(img.shape[1]//2,img.shape[0]//2),(img.shape[1]//2
                                                             +100,img.shape[0]//2+100),255,-1)
# cv.imshow('Masked_Rectangle',masked)
blank=np.zeros(img.shape[:2],dtype='uint8')
circle=cv.circle(blank.copy(),(img.shape[1]//2 + 45,img.shape[0]//2),100,255,-1)
cv.imshow('circle',circle)
rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),1000,-1)
cv.imshow('rectangle',rectangle)
weird_shape=cv.bitwise_and(circle,rectangle)
cv.imshow("Weird_Shape",weird_shape)

masked=cv.bitwise_and(img,img,mask=weird_shape)
cv.imshow("Weird_masked",masked)

cv.waitKey(0)