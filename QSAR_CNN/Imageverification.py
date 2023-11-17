import pandas as pd
import numpy as np
import os
import cv2
import imghdr
import tensorflow as tf

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

class Image_verification():
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def data_dir_list(self, data_dir):
        data_dir_train = data_dir+'train' 
        data_dir_val = data_dir+'val'
        data_dir_test = data_dir+'test'
        dir_list = [data_dir_train, data_dir_val, data_dir_test ]
        
        return dir_list

    def Image_verify(self, dir_list):
        for data in dir_list:
            image_exts = ['png','jpg']
            error = 0
            for image_class in os.listdir(data): 
                for image in os.listdir(os.path.join(data, image_class)):
                    image_path = os.path.join(data, image_class, image)
                    try: 
                        img = cv2.imread(image_path)
                        tip = imghdr.what(image_path)
                        if tip not in image_exts: 
                            error += 1
                            print('Image not in ext list {}'.format(image_path))
                            # os.remove(image_path)
                    except:
                        error += 1 
                        print('Issue with image {}'.format(image_path))
                        # os.remove(image_path)
        if error == 0:
            print('All images are readable')
    
    def fit(self):
        self.dir_list = self.data_dir_list(data_dir=self.data_dir)
        self.Image_verify(dir_list=self.dir_list)
        