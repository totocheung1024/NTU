import cv2 as cv

img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat',img)

# Converting to grayscale
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Blur 打格仔
blur= cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
cv.imshow('BLUR',blur)

# Edge Cascade 黑白得返d edge同線
canny=cv.Canny(img,125,175)
cv.imshow('Canny Edhes',canny)

# Dilating the image, iterations bigger, the thicker the outline
dilated= cv.dilate(canny,(7,7),iterations=4)
cv.imshow('Dilated', dilated)

#Eroding
eroded= cv.erode(dilated, (7,7),iterations=4)

cv.imshow('Eroded', eroded)

#Resized

resized=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)

#Croppng , Only keep the specific part of the image
cropped= img[50:200,200:400]
cv.imshow('Cropped',cropped)

cv.waitKey(0)