from genericpath import exists
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import os
import csv
import time
from datetime import datetime
import numpy as np
from win32com.client import Dispatch

def speak(str1):
     speak=Dispatch("SAPI.SpVoice")
     speak.Speak(str1)

video=cv2.VideoCapture(0)
faces_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('names.pkl','rb') as f:
        LABELS=pickle.load(f)
with open('faces_data.pkl','rb') as f:
        FACES=pickle.load(f)
face_original_shape=FACES.shape
FACES_RESHAPE=FACES.reshape(face_original_shape[0],-1)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES_RESHAPE,LABELS)

img_bag=cv2.imread('imagebag.jpeg')

col_names=['NAME', 'TIME']


while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faces_detect.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        crop_image=frame[y:y+h,x:x+w, :]
        resized_img=cv2.resize(crop_image,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        os.path.isfile("attendence/attendance_" + date + ".csv")

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),- 1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
    
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
        attendance=[str(output[0]),str(timestamp)]
    img_bag[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow('Frame',img_bag)
    k=cv2.waitKey(1)
    if k==ord('o'):
        speak("Attendance Taken...")
        time.sleep(5)
        if exists:
            with open("attendence/attendance_" + date + ".csv","+a") as csvfile:
                 writer=csv.writer(csvfile)
                 writer.writerow(attendance)
            csvfile.close()
        else:
            with open("attendence/attendance_" + date + ".csv","+a") as csvfile:
                 writer=csv.writer(csvfile)
                 writer.writerow(col_names)
                 writer.writerow(attendance)
            csvfile.close()
             
    if k==ord('q') :
        break

video.release()
cv2.destroyAllWindows()

