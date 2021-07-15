################## IMPORTS #####################

from textwrap import indent
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL
#load the model and pre-trained weights
from tensorflow.keras.models import model_from_json
from inception_blocks_v2 import faceRecoModel
from tensorflow import keras


#################### CALCULATING ENCODINGS ##################

from FaceRecognition import img_to_encoding, who_is_it

model = keras.models.load_model('/home/charchit/Desktop/DLS/face_recog/files/model_data/')

#model alias
faceRecogModel = model

#path to images
aman_image_path = '/home/charchit/Desktop/DLS/face_recog/dataset/aman/aman8.jpg'
charchit_image_path = '/home/charchit/Desktop/DLS/face_recog/dataset/charchit/charchit0.png'
mummy_image_path = '/home/charchit/Desktop/DLS/face_recog/dataset/mummy/mummy6.jpg'
papa_image_path = '/home/charchit/Desktop/DLS/face_recog/dataset/papa/papa0.jpg'


#create the database of the users! key is the name and value is the encoding value
database = {}
database['charchit'] = img_to_encoding(charchit_image_path,faceRecogModel)
database['aman'] = img_to_encoding(aman_image_path,faceRecogModel)
database['papa'] = img_to_encoding(papa_image_path,faceRecogModel)
database['mummy'] = img_to_encoding(mummy_image_path,faceRecogModel)

##########################################################

detector = MTCNN()

cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, color = cap.read()
    

    #Use MTCNN to detect faces
    # color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(color)
    if result != []:
        for person in result:
            x,y,w,h = person['box']

            _, identity = who_is_it(color,database,model)

            cv2.rectangle(color,
                          (x, y),
                          (x+w, y + h),
                          (0,155,255),
                          2)
            cv2.putText(color,identity,(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)

    #display resulting frame
    cv2.imshow('frame',color)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()
