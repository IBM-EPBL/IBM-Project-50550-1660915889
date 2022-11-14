#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


ls


# In[ ]:


cd//content/drive/MyDrive/Colab Notebooks/Dataset-20221108T081455Z-001/Dataset


# In[ ]:


ls


# **Import Neccessary Library**

# In[ ]:


import numpy as np#used for numerical analysis
import tensorflow #open source used for both ML and DL for computation
from tensorflow.keras.models import Sequential #it is a plain stack of layers
from tensorflow.keras import layers #A layer consists of a tensor-in tensor-out computation function
#Dense layer is the regular deeply connected neural network layer
from tensorflow.keras.layers import Dense,Flatten
#Faltten-used fot flattening the input or change the dimension
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout #Convolutional layer
#MaxPooling2D-for downsampling the image
from keras.preprocessing.image import ImageDataGenerator


# **Image Data Agumentation**

# In[ ]:


#setting parameter for Image Data agumentation to the training data
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
#Image Data agumentation to the testing data
test_datagen=ImageDataGenerator(rescale=1./255)


# **Loading our data and performing data agumentation**

# In[ ]:


#performing data agumentation to train data
x_train = train_datagen.flow_from_directory(
    r'/content/drive/MyDrive/TRAIN_SET',
    target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='sparse')
#performing data agumentation to test data
x_test = test_datagen.flow_from_directory(
    r'/content/drive/MyDrive/TRAIN_SET',
    target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='sparse') 


# In[ ]:


print(x_train.class_indices)#checking the number of classes


# In[ ]:


print(x_test.class_indices)#checking the number of classes


# In[ ]:


from collections import Counter as c
c(x_train .labels)


# **Creating the Model**

# In[ ]:


# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))

# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=5, activation='softmax')) # softmax for more than 2


# In[ ]:


classifier.summary()#summary of our model


# **Compiling the Model**

# In[ ]:


# Compiling the CNN
# categorical_crossentropy for more than 2
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 


# **Fitting the Model**

# In[ ]:


classifier.fit_generator(
        generator=x_train,steps_per_epoch = len(x_train),
        epochs=10, validation_data=x_test,validation_steps = len(x_test))# No of images in test set


# **Saving our Model**

# In[ ]:


# Save the model
classifier.save('nutrition.h5')


# **Nutrition Image Analysis using CNN**
# ---
# **Predicting our results**

# In[ ]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# In[ ]:


img = image.load_img("/content/drive/MyDrive/TEST_SET/APPLES/n07740461_10080.jpg",target_size= (64,64))#loading of the imageimg


# In[ ]:


x=image.img_to_array(img)#conversion image into array


# In[ ]:


x


# In[ ]:


x.ndim


# In[ ]:


x=np.expand_dims(x,axis=0) #expand the dimension


# In[ ]:


x.ndim


# In[ ]:


pred = classifier.predict(x)


# In[ ]:


pred


# In[ ]:


labels=['APPLES', 'BANANA', 'ORANGE','PINEAPPLE','WATERMELON']
labels[np.argmax(pred)]

