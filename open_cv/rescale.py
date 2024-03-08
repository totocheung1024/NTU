import cv2 as cv

img=cv.imread('Resources/Photos/cat_large.jpg')

cv.imshow('Cat',img)

#rescaleFrame refers to the size of window


def rescaleFrame(frame,scale=0.75):
    #Work for Images, Videos, Live video
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    
    return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    #Live video
    capture.set(3,width)
    capture.set(4,height)
    

resized_images=rescaleFrame(img)
cv.imshow('Cat',resized_images)
capture=cv.VideoCapture('resources/Videos/dog.mp4')

while True:
    isTrue,frame=capture.read()
    
    frame_resized= rescaleFrame(frame)
    cv.imshow('Video Resized',frame_resized)
    cv.imshow('Video',frame)
    
    if cv.waitKey(20) and 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()