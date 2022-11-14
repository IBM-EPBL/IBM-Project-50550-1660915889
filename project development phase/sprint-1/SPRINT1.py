#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[10]:


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


# In[11]:


x_train=train_datagen.flow_from_directory(r'/content/drive/MyDrive/TRAIN_SET',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=train_datagen.flow_from_directory(r'/content/drive/MyDrive/TEST_SET',target_size=(64,64),batch_size=32,class_mode='categorical')


# In[16]:


print(x_train.class_indices)


# In[17]:


print(x_test.class_indices)


# In[18]:


from collections import Counter as c
c(x_train .labels)

