import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class Data_loader():
    def __init__(self, data_train_dir, data_val_dir, data_test_dir, batch_size, random_seed):
        self.data_train_dir = data_train_dir
        self.data_val_dir = data_val_dir
        self.data_test_dir = data_test_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
    
    def load_dataset(self, data):
        
        #Load dataset
        data_load = tf.keras.utils.image_dataset_from_directory(data, batch_size=self.batch_size, 
                                                                 label_mode='binary', labels ="inferred", seed=self.random_seed)
        return data_load
        
    #Visualize
    def visualize(self):
        print('Data train samples')
        data = self.load_dataset(data=self.data_train_dir)
        data_iterator = data.as_numpy_iterator()
        batch = data_iterator.next()
        
        sns.set()
        fig, ax = plt.subplots(ncols=4, figsize=(20,20))
        for idx, img in enumerate(batch[0][:4]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(batch[1][idx]) 
        
    
    def fit(self):
        self.data_train = self.load_dataset(data=self.data_train_dir)
        self.data_val = self.load_dataset(data=self.data_val_dir)
        self.data_test = self.load_dataset(data=self.data_test_dir)
        self.visualize()
        
    