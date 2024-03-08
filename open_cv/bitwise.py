import cv2 as cv
import numpy as np
#0 turn off, 1 turn on

blank=np.zeros((400,400),dtype="uint8")
cv.imshow('blank',blank)
rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle=cv.circle(blank.copy(),(200,200),200,255,-1)

cv.imshow('Rectangle',rectangle)
cv.imshow('Circle',circle)

# bitwise AND -->Intesecting regions
bitwise_and=cv.bitwise_and(rectangle, circle)
cv.imshow('bitwise_and',bitwise_and)

#bitwise or -->non intersecting and intersecting regions
bitwise_or=cv.bitwise_or(rectangle, circle)
cv.imshow('bitwise_or',bitwise_or)

#bitwise xor-->non intersecting regions
bitwise_xor=cv.bitwise_xor(rectangle,circle)
cv.imshow('Bitwise XOR',bitwise_xor)

cv.waitKey(0)