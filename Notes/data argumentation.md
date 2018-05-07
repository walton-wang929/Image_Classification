# common data argumentation methods

### 1. calculate training img mean value
```
#using opencv, numpy
import os
import cv2
import numpy as np 

# define list elements add 
def list_add(int_list):
   sum_num = 0
   for i in int_list:
      sum_num = sum_num+i
   ave_num = sum_num/len(int_list)
   return int(ave_num)
  
file_dir = r'D:\TF_Try\gender\data\train\female'

img_list = os.listdir(file_dir)

blue = []
green = []
red = []

for img in img_list:
    image = cv2.imread(os.path.join(file_dir,img))
    b,g,r = cv2.split(image)
    
    b = np.concatenate(b)
    g = np.concatenate(g)
    r = np.concatenate(r)
    
    average_b = int(np.divide(np.sum(b),np.size(b)))
    average_g = int(np.divide(np.sum(g),np.size(g)))
    average_r = int(np.divide(np.sum(r),np.size(r)))
    
    blue.append(average_b)
    green.append(average_g)
    red.append(average_r)
    
ave_blue = list_add(blue)
ave_green = list_add(green)
ave_red = list_add(red)

ave = int((ave_blue + ave_green + ave_red) / 3 )

print('mean img value : ', ave)

```
