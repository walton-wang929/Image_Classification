# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:56:33 2018

@author: TWang

using our trained model to add more data 

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import cv2
import shutil

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
    
  img = cv2.imread(file_name) 
  resized = cv2.resize(img,(input_height, input_width))
  float_img= np.asfarray(resized,dtype='float32')
  image_np_expanded = np.expand_dims(float_img, axis=0)
  normalized = np.divide(np.subtract(image_np_expanded,[input_mean]),[input_std])
  
  return normalized

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  
  model_file = r"D:\TF_Try\gender\tensorflow for poets\tf_files\retrained_graph_5000.pb"
  label_file = r"D:\TF_Try\gender\tensorflow for poets\tf_files\retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  graph = load_graph(model_file)
  
  test_imgs_dir = r'D:\TF_Try\gender\data\train\male'
  
  test_img_path = os.listdir(test_imgs_dir)

  dst = r'D:\TF_Try\gender\data\train\other2'

  for img in test_img_path:
      
      file_name = os.path.join(test_imgs_dir,img)
      
      t = read_tensor_from_image_file(file_name,
                                      input_height=input_height,
                                      input_width=input_width,
                                      input_mean=input_mean,
                                      input_std=input_std)
    
      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      input_operation = graph.get_operation_by_name(input_name);
      output_operation = graph.get_operation_by_name(output_name);
      
      with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
      results1 = np.squeeze(results)
      
      if results1[0]>0.5:
          print(img,results1[0])
          shutil.move(file_name,dst)
      else:
          pass

