import cv2 as cv

#img= cv.imread('Resources/Photos/cat_large.jpg')

#cv.imshow('cat', img)

capture=cv.VideoCapture('Resources/videos/dog.mp4')

while True:
    isTrue, frame=capture.read()
    cv.imshow('Video', frame)
    
    if cv.waitKey(20) and 0XFF==ord('d'):
        #WaitKey(20) refers to delay of the show of video
        #If the letter D is pressed, end of this video
        break

capture.release()
cv.destroyAllWindows()

#cv.waitKey(0)