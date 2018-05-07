# Data processing 

**multiple files rename**
```
import os
import numpy as np 

def file_rename(file_directory, renamed, random_shuffle = True):
    
    file_list  = os.listdir(file_directory)
    
    
    if random_shuffle == True:
        
        np.random.shuffle(file_list)
        
        print("the file list has been random shuffled.\n")
        
        i = 0
        for item in file_list:
        
            os.rename(os.path.join(file_directory,item),os.path.join(file_directory,(renamed + str(i)+'.jpg')))
            i+=1    
        print("all files has been renamed.\n")
    
    else:
        print("don't choose random shuffled.\n")
        i = 0
        for item in file_list:
        
            os.rename(os.path.join(file_directory,item),os.path.join(file_directory,(renamed + str(i)+'.jpg')))
            i+=1    
        print("all files has been renamed.\n")
         
file_directory = r'Y:\WangTao\Person_attributes_detection\airport_gender_data\data\female'

renamed = 'female'

file_rename(file_directory=file_directory, renamed=renamed)
```

**random sample file and move into revelant folder**
```
import os
import shutil
import random

def file_sample(source_directory, sample_num, dst_directory):
    
    file_list  = os.listdir(source_directory)
    
    sample_list = random.sample(file_list, sample_num)
    
    for item in sample_list:
        
        shutil.move(os.path.join(source_directory,item),os.path.join(dst_directory,item))
        
    print("all files has been moved.\n")
    
```
