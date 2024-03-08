import cv2 as cv
import numpy as np

img=cv.imread('Resources/Photos/cats.jpg')
cv.imshow("Cat",img)

blank=np.zeros(img.shape[:2],dtype='uint8')

b,g,r=cv.split(img)

#Gray
cv.imshow('Blue',b)
cv.imshow('Green',g)
cv.imshow('Red',r)

print(img.shape,"img.shape")
print(b.shape,"b.shape")
print(g.shape)
print(r.shape)

blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])
print(blue.shape,"Blue.shape")
print(green.shape)
print(red.shape)

cv.imshow('Blue',blue)
cv.imshow('Green',green)
cv.imshow('Red',red)


print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged=cv.merge([b,g,r])
cv.imshow('Merged Image',merged)
cv.waitKey(0)