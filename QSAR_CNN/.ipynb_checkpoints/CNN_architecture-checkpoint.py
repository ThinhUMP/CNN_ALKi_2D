import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa
from keras_core.constraints import max_norm
from keras.callbacks import LearningRateScheduler
import warnings
import math
warnings.filterwarnings("ignore")
def step_decay(epochs):
  initial_lrate = .1
  drop = .5
  epochs_drop = 10.0
  lrate = initial_lrate * math.pow(drop, math.floor((1+epochs)/(epochs_drop)))
  return lrate
class CNN_network():

    def __init__(self, node_layer1, node_layer2, input_dim, optim = 'Adam', loss=tf.losses.BinaryCrossentropy(),
                 learning_rate = 0.001, weight_decay=0.1, regu_l2=0.001, kernel_initial = 'uniform', node_out=1,
                 out_activation='sigmoid', drop_out=0.0, batch_norm=False, max_nor = 2.0, 
                 noise = False, noise_values=0.1, metrics=['accuracy']):
        self.node_layer1 = node_layer1
        self.node_layer2 = node_layer2
        self.input_dim = input_dim
        self.optim = optim
        self.loss = loss
        self.learning_rate =learning_rate
        self.weight_decay = weight_decay
        self.kernel_initial = kernel_initial
        self.regu_l2 = regu_l2
        self.node_out = node_out
        self.out_activation = out_activation
        self.drop_out = drop_out
        self.max_nor = max_nor
        self.batch_norm = batch_norm
        self.noise = noise
        self.noise_values = noise_values
        self.metrics = metrics

    def architecture(self):
        #CNN layer
        model = Sequential(name='CNN_Network')
        model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=self.input_dim))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(16, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())

        
        #NLP layer
        model.add(Flatten())
        model.add(Dense(input_dim=self.input_dim, units=self.node_layer1, kernel_regularizer=l2(self.regu_l2), 
                        kernel_initializer=self.kernel_initial, activation='relu', name='Layer_1', 
                        # kernel_constraint = max_norm(max_value = self.max_nor)
                                                    ))
        if self.noise == True:
            model.add(GaussianNoise(self.noise_values))
        if self.batch_norm == True:
            model.add(BatchNormalization())
        model.add(Dropout(self.drop_out, name='Dropout_1'))
        model.add(Dense(units=self.node_layer2, kernel_regularizer=l2(self.regu_l2), 
                        kernel_initializer=self.kernel_initial, activation='relu', name='Layer_2', 
                        # kernel_constraint = max_norm(max_value = self.max_nor)
                        ))
        if self.noise == True:
            model.add(GaussianNoise(self.noise_values))
        if self.batch_norm == True:
            model.add(BatchNormalization())
        model.add(Dropout(self.drop_out, name='Dropout_2'))

        #Out layer
        model.add(Dense(units = self.node_out, activation = self.out_activation, kernel_initializer = self.kernel_initial, name = 'Out_put'))
        return model

    def generate_model(self):
        model = self.architecture()
        #Optimize
        if self.optim == 'Adam':
            opt = Adam(learning_rate=self.learning_rate, decay=self.weight_decay)
        else:
            opt = SGD(learning_rate=self.learning_rate, decay=self.weight_decay)
        model.compile(loss=self.loss, optimizer = opt, metrics = self.metrics)
        return model