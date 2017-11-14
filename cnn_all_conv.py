import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import InputLayer
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.utils import shuffle
import keras
import csv
import tensorflow as tf
import math
from keras.layers import Lambda
from keras.callbacks import TensorBoard

seed = 7
np.random.seed(seed)

train = pd.read_csv('./data/train_x_cleaned.csv',header=None)
train_x = train.values

def normalize(array):
    # Normalize the data
    array = array.astype(np.float32) / 255.0 
    a = array-array.mean(axis=1,keepdims=True)
    a = a / array.std(axis = 1,keepdims = True)
    return a

train_x = normalize(train_x)
train_x = train_x.reshape(train_x.shape[0],1,64,64)

result = pd.read_csv('./data/train_y.csv',header=None)
train_y = np_utils.to_categorical(result.values.ravel())
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

def fmp(x):
    return tf.nn.fractional_max_pool(x, p_ratio, pseudo_random = rand, overlapping = True)[0]

def larger_model():
    model = Sequential()
    
    model.add(Convolution2D(64, (7, 7), border_mode='valid', input_shape=(1, 64, 64)))
    model.add(LeakyReLU(alpha=.5)) 
    model.add(Dropout(0.1))
    model.add(Convolution2D(64, (7, 7)))    
    model.add(LeakyReLU(alpha=.5)) 
    model.add(Convolution2D(64, (7, 7)))
    model.add(LeakyReLU(alpha=.5)) 
    model.add(Dropout(0.1))
    model.add(Convolution2D(64, (7, 7)))
    model.add(LeakyReLU(alpha=.5))
    model.add(Convolution2D(64, (7, 7)))
    model.add(LeakyReLU(alpha=.5))     
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, (7, 7)))
    model.add(LeakyReLU(alpha=.5))
    model.add(Convolution2D(64, (7, 7)))
    model.add(LeakyReLU(alpha=.5))     
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, (7, 7)))
    model.add(LeakyReLU(alpha=.5))     
    model.add(Convolution2D(64, (7, 7)))
    model.add(LeakyReLU(alpha=.5))     
    model.add(Dropout(0.4))
    model.add(Convolution2D(64, (8, 8)))
    model.add(LeakyReLU(alpha=.5))  
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization(epsilon=1e-05,momentum=0.99,weights=None,beta_init='zero',gamma_init='one',gamma_regularizer=None,beta_regularizer=None))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

test = pd.read_csv('./data/test_x_cleaned.csv',header=None)
test_x = normalize(test.values)
test_x = test_x.reshape(test_x.shape[0],1,64,64)

num_classes = train_y.shape[1]

# build the model
model = larger_model()
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# fit the model
model.fit(train_x, train_y, nb_epoch=300, batch_size=200, verbose=1, callbacks = [tensorboard])

# predict on the test dadaset
classes = model.predict_classes(test_x, batch_size=100)
proba = model.predict_proba(test_x, batch_size=100)
slnum=5
# save the classes and probability for each prediction in the test dataset
np.savetxt("./output/classes_sample_kaggle_arr_10{0}.csv".format(slnum),classes, delimiter =",")
np.savetxt("./output/prob_sample_kaggle_arr_10{0}.csv".format(slnum),proba, delimiter =",")

writer = open('./output/CNN_epoch20_normalized features_arr_10{0}.csv'.format(slnum), 'w')
writer.write('Id,Label\n')
for i,row in enumerate(classes):
    writer.write(str(i+1)+","+str(row)+"\n")
writer.close()

classes = model.predict_classes(val_x, batch_size=100)
classes = np_utils.to_categorical(classes.ravel())
from sklearn.metrics import accuracy_score
acc_pred = np.asarray(classes)
print (accuracy_score(val_y, classes))
