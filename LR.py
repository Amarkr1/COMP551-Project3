
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[2]:


# read training data from CSV file 
data = pd.read_csv('./input/train_x_clean.csv',header=None)


# In[3]:


print (data.sample(2))


# In[4]:


images = data.iloc[:,0:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))


# In[5]:


result = pd.read_csv('./input/train_y.csv',header=None)
labels_flat = result.values.ravel()


# In[6]:


train_x,val_x,train_y,val_y = train_test_split(images,result,test_size=0.20, random_state=42)


# In[7]:


print(train_x.shape)
print(val_x.shape)
print(train_y.shape)
print(val_y.shape)


# In[8]:


model = LogisticRegression()


# In[9]:


model.fit(train_x,train_y.values[:,0])


# In[10]:


print("Training Accuracy : {0}".format(model.score(train_x,train_y)))
print("Validation Accuracy : {0}".format(model.score(val_x,val_y)))


# In[11]:


val_x[val_x<(230.0/255.0)]=0 #thresholding the pixels and setting everything below 230 to zero
val_x[val_x>=(230.0/255.0)]=1


# In[12]:


print("Training Accuracy : {0}".format(model.score(train_x,train_y)))
print("New Validation Accuracy : {0}".format(model.score(val_x,val_y)))


# In[27]:


labelencoder_X = LabelEncoder()
train_y.values[:,0] = labelencoder_X.fit_transform(train_y.values[:, 0])


# In[40]:


test_x = pd.read_csv('./input/test_x_clean.csv',header=None)


# In[41]:


test_images = test_x.iloc[:,0:].values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(test_images.shape))


# In[42]:


test_y = model.predict(test_images)


# In[43]:


file = open('predict_LR.csv','w')
file.write('Id,Label\n')
for i in range(len(test_y)):
    file.write('{0},{1}\n'.format((i+1),test_y[i]))
file.close()  


# In[13]:


#the bottom code for trying out 2nd variation of the project


# In[14]:


train_x,val_x,train_y,val_y = train_test_split(images,result,test_size=0.20, random_state=42)


# In[15]:


train_x[train_x<(230.0/255.0)] = 0
train_x[train_x>=(230.0/255.0)] = 1


# In[16]:


print(train_x.shape)
print(val_x.shape)
print(train_y.shape)
print(val_y.shape)


# In[17]:


model2 = LogisticRegression()


# In[18]:


model2.fit(train_x,train_y.values[:,0])


# In[19]:


print("Training Accuracy : {0}".format(model2.score(train_x,train_y)))
print("Validation Accuracy : {0}".format(model2.score(val_x,val_y)))


# In[20]:


val_x[val_x<230]=0 #thresholding the pixels and setting everything below 230 to zero
val_x[val_x>=(230.0/255.0)]=1


# In[21]:


print("Training Accuracy : {0}".format(model2.score(train_x,train_y)))
print("New Validation Accuracy : {0}".format(model2.score(val_x,val_y)))

