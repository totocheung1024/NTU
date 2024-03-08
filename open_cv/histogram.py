import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img= cv.imread('Resources/Photos/cats.jpg')

cv.imshow('Cats',img)

blank=np.zeros(img.shape[:2],dtype='uint8')


gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)


circle=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)

mask=cv.bitwise_and(img,img,mask=circle)
cv.imshow('Masked',mask)


# gray_hist=cv.calcHist([gray],channels=[0],mask=mask,histSize=[256],ranges=[0,256])

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('number of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

#Color histogram

colors=('b','g','r')
for i,col in enumerate(colors):
    hist=cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])

plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('number of pixels')
plt.show()

cv.waitKey(0)