# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:31:49 2017

@author: TWang

after traing , test 

"""

from keras.models import load_model

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
import cv2


# dimensions of our images.
img_width, img_height = 100, 250

input_shape = (img_width, img_height, 3)

test_model = load_model('final_model.h5')

def predict(basedir, model):
    
    
    male_counter = 0
    female_counter = 0
    
    N = 7687
    
    for i in range(0,N):
        
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
        
        if probabilities > 0.5 :
            label = 'male'
            male_counter +=1
        else :
            label = 'female'
            female_counter +=1
            
        # get the prediction label
        #print("Image ID: {}, Label: {}, Probability:{}".format(inID, label, probabilities))
        
# =============================================================================
#         # display the predictions with the image
#         cv2.putText(orig, "{}".format(label), (10, 30),
#                     cv2.FONT_HERSHEY_PLAIN, 1, (43, 99, 255), 2)
#     
#         cv2.imshow("Classification", orig)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
# =============================================================================
        
        '''
        make correct list when testing
        '''
    print("male: ", male_counter, male_counter/N)
    print("female: ",female_counter, female_counter/N)
        
# =============================================================================
# basedir = r"D:\WANG Tao\gender\data\test\female\female"
# predict(basedir, test_model)
# 
# =============================================================================
basedir = r"Y:\data_argumentation\pedestrian_data\airport4\person"
predict(basedir, test_model)



print('done')