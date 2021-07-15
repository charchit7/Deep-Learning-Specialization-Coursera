from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv2
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


#create the encoding of the images!
def img_to_encoding(image_path, model):
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)

    return embedding / np.linalg.norm(embedding, ord=2)

#create the encoding of the images!
def img_to_encoding_for_who(image_name, model):
    target_size = (160,160)
    img = Image.fromarray(image_name, 'RGB')
    aaa = img.resize(target_size)
    img = np.around(np.array(aaa) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)

    return embedding / np.linalg.norm(embedding, ord=2)




def who_is_it(image_name, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras
    
    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """
  
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding =  img_to_encoding_for_who(image_name,model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name
    
    # if min_dist > 0.7:
    #     print("Not in the database.")
    # else:
    #     print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity