import numpy as np 
import cv2
import pickle
from PIL import Image
import os
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import base64
from io import BytesIO
import json
import random
from keras.preprocessing import image

model = load_model('face_recognition_model.h5')

#To access the frontalface cascade from Haarcascade
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

#To access the live feed from the webcam
cap = cv2.VideoCapture(0) 

while(True): 

  #To capture frame by frame
  isTrue, frame = cap.read()

  #Detecting face
  faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.3, minNeighbors = 5)

  for (x, y, w, h) in faces: 

    #This is the region of interest for us (the face) 
    roi = frame[y:y+h, x:x+w]
    face = roi

    #Resizing the array for model 
    face = cv2.resize(face, (224, 224))
    im = Image.fromarray(face)
      
    img_array = np.array(im)
                
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    print(pred)

    if(pred[0][1] > 0.5): 
      name = "Diljot"

    else: 
      name = "Unknown"

    #Draw rectangle 
    color = (0, 255, 0) 
    stroke = 2
    width = x+w
    height = y+h
    font = cv2.FONT_HERSHEY_SIMPLEX

    #starting and ending coordinates 
    cv2.rectangle(frame, (x, y), (width, height), color, stroke) 
    cv2.putText(frame, name, (x,y), font, 1 , color, stroke, cv2.LINE_AA)
    
    #Display the live feed 
    cv2.imshow('Frame', frame)

  if cv2.waitKey(20) & 0xFF == ord('d'): 
    img_id = 0
    break

#This is to release the capture
cap.release()
cv2.destroyAllWindows()
