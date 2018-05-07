# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:24:27 2018

@author: TWang

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import cv2

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
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  print("result", result)
  return result

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
  
  test_imgs_dir = r'D:\TF_Try\gender\tensorflow for poets\tf_files\test'
  
  test_img_path = [ os.path.join(test_imgs_dir, '{}.jpg'.format(i)) for i in range(7,17) ]

  for img in test_img_path:
      
      frame = cv2.imread(img)
      file_name = img
      
      t = read_tensor_from_image_file(file_name,
                                      input_height=input_height,
                                      input_width=input_width,
                                      input_mean=input_mean,
                                      input_std=input_std)
    
      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      input_operation = graph.get_operation_by_name(input_name);
      output_operation = graph.get_operation_by_name(output_name);
      print("output",output_operation)
      
      with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
      results1 = np.squeeze(results)
    
      top_k = results1.argsort()[-5:][::-1]
      labels = load_labels(label_file) 
  
      print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    
      for i in top_k:
        print(labels[i], results1[i])
        
      cv2.putText(frame,"F:%.5f"%results1[0],(5,20),2, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.putText(frame,"M:%.5f"%results1[1],(5,40),2, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.imshow('detect',frame)
      if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            
cv2.destroyAllWindows()        
