import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import ImageGrab

path = 'People_Folder'
list_people_images= []
image_names = []
myList = os.listdir(path)
print(myList)

for per in myList:
    curImg = cv2.imread(f'{path}/{per}')
    list_people_images.append(curImg)
    image_names.append(os.path.splitext(per)[0])
print(image_names)

def determine_encodings(images):
    encodings_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings_list.append(encode)
    return encodings_list 

def markAttendance(name):
    with open('Attendance_FINAL.csv','r+') as f:
        Data = f.readlines()
        list_of_names=[]
        for statement in Data:
            entry=statement.split(',')
            list_of_names.append(entry[0])

        if name not in list_of_names: 
            current_time=datetime.now()
            date_time_string=current_time.strftime('%H:%M:%S')
            month_year_string=current_time.strftime('%Y-%m-%d')
            f.writelines(f'\n{name},{date_time_string},{month_year_string}')



current_encoding_list=determine_encodings(list_people_images)
print("ENCODING COMPLETE")   

web_per = cv2.VideoCapture(0)
 
while True:
    success, img = web_per.read()
    resized_img = cv2.resize(img,(0,0),None,0.25,0.25)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
 
    faces_shown_web = face_recognition.face_locations(resized_img)
    encodes_shown_web = face_recognition.face_encodings(resized_img,faces_shown_web)

    for encodeFace,faceLoc in zip(encodes_shown_web ,faces_shown_web ):
        matches = face_recognition.compare_faces(current_encoding_list,encodeFace)
        faceDis = face_recognition.face_distance(current_encoding_list,encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = image_names[matchIndex].upper()
    
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(10,255,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(10,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            markAttendance(name)

        cv2.imshow('Webcam',img)
        cv2.waitKey(1)

           
 
       
    