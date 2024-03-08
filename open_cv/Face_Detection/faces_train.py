import os
import cv2 as cv
import numpy as np

people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
DIR=r'C:\GitHub_Profile\opencv-course\Face_Detection\Faces\train'

haar_casacde=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]


def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        
        for image in os.listdir(path):
            image_path=os.path.join(path,image)
            
            image_array=cv.imread(image_path)
            gray=cv.cvtColor(image_array,cv.COLOR_BGR2GRAY)
            
            face_rect=haar_casacde.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            
            for (x,y,w,h) in face_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
create_train()
print('Training done')
features=np.array(features,dtype='object')
labels=np.array(labels)

print(f'Length of the features={len(features)}')
print(f'Length of the labels={len(labels)}')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
#Train the recognizer on the features list and the labels list
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
             
np.save('features.npy',features)
np.save('labels.npy',labels)