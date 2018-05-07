# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:40:14 2018

@author: TWang

"""
from keras.models import load_model

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
import cv2
import shutil

# dimensions of our images.
img_width, img_height = 100, 250

input_shape = (img_width, img_height, 3)

test_model = load_model('final_model.h5')

def predict(basedir, model):
    
    male_counter = 0
    female_counter = 0
    
    N = 7687
    
    dst_male = r"C:\Users\twang\Desktop\New Folder"
    dst_female = r"Y:\data_argumentation\pedestrian_data\Gender\female"
    
    
    for i in range(0,N):
        
        try:
            image_path = basedir + str(i) + '.jpg'
            
            orig = cv2.imread(image_path)
        
            #print("[INFO] loading and preprocessing image...")
            image = load_img(image_path, target_size=(img_width, img_height))
            image = img_to_array(image)
        
            # important! otherwise the predictions will be '0'
            image = image / 255
        
            image = np.expand_dims(image, axis=0)
            
            # use the bottleneck prediction on the top model to get the final
            # classification
            
            probabilities = model.predict(image)
            
            if probabilities > 0.1 :
                label = 'male'
                male_counter +=1
                shutil.copy2(image_path,dst_male)
                
            else :
                pass
#==============================================================================
#                 label = 'female'
#                 female_counter +=1
#                 shutil.copy2(image_path,dst_female)
#==============================================================================
            
            '''
            make correct list when testing
            '''
            print("male: ", male_counter, male_counter/N)
            print("female: ",female_counter, female_counter/N)
            
        except:
            continue
        
# =============================================================================
# basedir = r"D:\WANG Tao\gender\data\test\female\female"
# predict(basedir, test_model)
# 
# =============================================================================
basedir = r"D:\TF_Try\gender\data\train\female"

predict(basedir, test_model)

print('done')
