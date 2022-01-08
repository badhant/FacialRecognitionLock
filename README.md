# Facial-Recognition-Door-Lock

In this project we designed an embedded system using a RaspberryPi to create a facial recognition door lock. The way this system work is it uses a keypad code (there is one for each designated user) to identify the the set users. With the basis of the passcode, we then used to a facial recognition model to recognise if the right user is opening the lock. If these parameters are passed correctly, the lock opens. 

# Face Recognition Model 

The model was made using convolutional neural networks (CNN) with VGG16 in the Keras machine learning library. 
