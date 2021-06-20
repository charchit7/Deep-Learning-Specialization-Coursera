import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras_preprocessing.image import smart_resize

path = '/media/charchit/New Volume/DataSet/Cat_Dog_data/train/cat'
cat_images = os.listdir(path)

n = len(cat_images)
