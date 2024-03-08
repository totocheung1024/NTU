import cv2 as cv
import numpy as np

blank= np.zeros((500,500,3),dtype='uint8')
#(height,width,RGB)
cv.imshow('Blank',blank)

#1. Paint the image a certain coloour
blank[:]=0,255,0
"""or 
blank[:, :, 0] = 255
blank[:, :, 1] = 255
blank[:, :, 2] = 255
"""
cv.imshow('Green',blank)

blank[:]=0,0,255
cv.imshow('Green',blank)

# 1. Paint the image a certian colour
# blank[200:300,300:400]=0,255,0
# cv.imshow('Green',blank)

# 2. Draw a Rectangle
#cv.rectangle(blank, (0,0),(250,250),(0,255,0),thickness=2)
cv.rectangle(blank,(0,0),(250,500),(0,255,0),thickness=cv.FILLED)
# Coordinate of the top-left coner, bottom-right corner, color, thickness


cv.imshow('Rectangle',blank)
#cv.rectangle(blank,(0,0),(250,500),(0,255,0),thickness=-1)
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0),thickness=-1)
#circle: height, wid


# 3. Draw a circle
# cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40, (0,255,0),thickness=-1)

# object, (center of the circle, radius, (color), thickness)
cv.imshow('Circle',blank)

# 4. Draw a line
cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,255,255),thickness=3)
cv.imshow("line",blank)
#Starting coordinate of the line , end of the coordinate of the line, color, thinkness


# 5. Write text
cv.putText(blank,'Hello',(255,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(255,255,0),thickness=2)
cv.imshow("Text",blank)
#Text, (Coordinate of the top-left coner), Type of font, fontsize, thinkness of line

img= cv.imread("Resources/Photos/cat.jpg")
cv.imshow('Cat',img)

cv.waitKey(0)