#simple script to change the images names as per your requirement!


import os 
path = "/home/charchit/Desktop/DLS/face_recog/dataset/"
name_path = os.path.join(path,'aman')
name_images = os.listdir(name_path)

name = 'papa'
counter = 0
ext = '.jpg'

for image in name_images:
    a=os.path.join(name_path,image)
    os.rename(a,name+str(counter)+ext)
    counter+=1

