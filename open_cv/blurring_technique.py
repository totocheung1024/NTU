import cv2 as cv

img= cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats',img)

#Averaging

average=cv.blur(img, (3,3))
cv.imshow('Average Blur',average)

#Gaussian_Blur
gauss=cv.GaussianBlur(img,(3,3), 0)
cv.imshow('Gaussian Blur',gauss)

#Median Blur
#Kernal size default 33
median=cv.medianBlur(img,3)
cv.imshow('median',median)

#Bilateral Bluring, bluring but still see the edge
#Using the center of the pixel to affect the other pixel within a keral


bilateral=cv.bilateralFilter(img, 5, sigmaColor=35,sigmaSpace=25)
cv.imshow("bilateral",bilateral)

cv.waitKey(0)