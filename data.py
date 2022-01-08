import numpy as np 
import cv2
import os

def get_faces(directory, name): 

  #To access the frontalface cascade from Haarcascade
  face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

  #To access the live feed from the webcam
  cap = cv2.VideoCapture(0) 

  #Keeps track of the img index 
  img_id = 0

  #Accessing images folder 
  os.chdir(directory)
  os.mkdir(name)

  while(True): 

    #To capture frame by frame
    isTrue, frame = cap.read()

    #Detecting face
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.3, minNeighbors = 5)

    #Increase img id as a snapshot is taken
    img_id += 1

    for (x, y, w, h) in faces: 

      #This is the region of interest for us (the face) we are saving this as a png image
      roi = frame[y:y+h, x:x+w]

      #To get the image with region of interest
      img_item = str(img_id) + ".jpg"
      save = directory + name 
      cv2.imwrite(os.path.join(save,img_item), roi)

      #Draw rectangle 
      color = (0, 255, 0) #BGR
      stroke = 2
      width = x+w
      height = y+h

      #starting and ending coordinates 
      cv2.rectangle(frame, (x, y), (width, height), color, stroke) 
    
    #Display the live feed 
    cv2.imshow('Frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'): 
      img_id = 0
      break

  #This is to release the capture
  cap.release()
  cv2.destroyAllWindows()


def test_images(img_direc, save_path, name): 

  face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

  #Make saving directory
  os.chdir(save_path)
  os.mkdir(name)
  img_id = 0

  #Going through given images 
  for image in os.listdir(img_direc): 
    
    #reading image
    frame = cv2.imread(img_direc + image)

    #To detect face
    face = face_cascade.detectMultiScale(frame, scaleFactor = 1.3, minNeighbors = 2)
    img_id += 1

    #Getting ragion of interest and saving it
    for (x, y, w, h) in face:
      roi = frame[y:y+h, x:x+w]
      img_item = str(img_id) + ".jpg"
      save = save_path + name 
      cv2.imwrite(os.path.join(save,img_item), roi)

direc = '/users/diljot/Desktop/valid/'
save = '/users/diljot/Desktop/'
test_images(direc, save, 'Diljot')