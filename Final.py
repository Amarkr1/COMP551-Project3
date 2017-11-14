
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D
from keras.regularizers import l2
from sklearn.cross_validation import StratifiedKFold
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.utils import shuffle
import keras
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard


# In[2]:


seed = 7
np.random.seed(seed)


# In[3]:


train = pd.read_csv('./input/train_x.csv',header=None)
train_x = train.values


# In[4]:


train.sample(2)


# In[5]:


def normalize1(array):
    # Normalize the data
    array = array.astype(np.float32) / 255.0 
    a = array-array.mean(axis=1,keepdims=True)
    a = a / array.std(axis = 1,keepdims = True)
    return a


# In[6]:


def normalize2(array):
    # Normalize the data
    array = array.astype(np.float32) / 255.0 
    return array


# In[7]:


train_x = normalize2(train_x)


# In[8]:


train_x = train_x.reshape(train_x.shape[0],1,64,64)


# In[9]:


result = pd.read_csv('./input/train_y.csv',header=None)
train_y = np_utils.to_categorical(result.values.ravel())


# In[10]:


train_xs, test_xs, train_ys, test_ys= train_test_split(train_x, train_y, test_size=0.05)
print(train_xs.shape)
print(test_xs.shape)


# In[11]:


num_classes = train_y.shape[1]


# In[12]:


def larger_model():
    model = Sequential()
    model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 64, 64), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[13]:


def larger_model2():
    # create model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 64, 64), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(BatchNormalization(epsilon=1e-05,momentum=0.99,weights=None,beta_init='zero',gamma_init='one',gamma_regularizer=None,beta_regularizer=None))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[12]:


def larger_model3():
    # create model
    model = Sequential()
    model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 64, 64), activation='relu'))
    model.add(Convolution2D(64, 7, 7, activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(BatchNormalization(epsilon=1e-05,momentum=0.99,weights=None,beta_init='zero',gamma_init='one',gamma_regularizer=None,beta_regularizer=None))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[15]:


def larger_model4():
    # create model
    model = Sequential()
    model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 64, 64), activation='relu'))
    model.add(Convolution2D(64, 7, 7, activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(BatchNormalization(epsilon=1e-05,momentum=0.99,weights=None,beta_init='zero',gamma_init='one',gamma_regularizer=None,beta_regularizer=None))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(300, activation='relu'))
    #model.add(Dense(200, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[17]:


# build the model
model = larger_model()
# fit the model
#tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(train_xs, train_ys, nb_epoch=100, batch_size=200, verbose=1,validation_data=(test_xs, test_ys))#,callbacks = [tensorboard])
#history = model.fit(train_x, train_y, nb_epoch=120, batch_size=200, verbose=1)


# In[18]:


# build the model
model = larger_model4()
# fit the model
# fit the model
history = model.fit(train_xs, train_ys, nb_epoch=200, batch_size=200, verbose=1,validation_data=(test_xs, test_ys))
#history = model.fit(train_x, train_y, nb_epoch=120, batch_size=200, verbose=1)


# In[13]:


# build the model
model = larger_model3()
# fit the model
# fit the model
history = model.fit(train_xs, train_ys, nb_epoch=100, batch_size=200, verbose=1,validation_data=(test_xs, test_ys))
#history = model.fit(train_x, train_y, nb_epoch=120, batch_size=200, verbose=1)


# In[15]:


score = model.evaluate(test_xs, test_ys, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[16]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.xlabel('epoch', fontsize=20)
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[17]:


#saving the trained model
# serialize model to JSON
model_json = model.to_json()
slnum=26
with open("model{0}.json".format(slnum), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model{0}.h5".format(slnum))
print("Saved model to disk")


# In[18]:


test = pd.read_csv('./input/test_x.csv',header=None)
test_x = normalize2(test.values)
test_x = test_x.reshape(test_x.shape[0],1,64,64)

test_x.shape


# In[19]:


# predict on the test dadaset
classes = model.predict_classes(test_x, batch_size=200)
proba = model.predict_proba(test_x, batch_size=200)

# save the classes and probability for each prediction in the test dataset
#np.savetxt("./output/classes_sample_kaggle_{0}.csv".format(slnum),classes, delimiter =",")
#np.savetxt("./output/probabilities_model_{0}.csv".format(slnum),proba, delimiter =",")


# In[20]:


writer = open('./output/Final_{0}.csv'.format(slnum), 'w')
writer.write('Id,Label\n')
for i,row in enumerate(classes):
    writer.write(str(i+1)+","+str(row)+"\n")
writer.close()


# In[21]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[27]:


cm = confusion_matrix(np.argmax(test_ys,axis=1), model.predict_classes(test_xs))
plt.matshow(cm)
plt.title('Confusion matrix', fontsize=20,y=1.08)
plt.colorbar()
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.savefig("cMatrix_{0}.png".format(slnum), dpi=300)
plt.show()


# In[26]:


np.argmax(test_ys,axis=1)


# In[28]:


cm


# In[30]:


import numpy, scipy.io

scipy.io.savemat('cm.mat', mdict={'cm': cm})

