#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('conda install --yes keras')


# In[ ]:


get_ipython().system('conda install tensorflow')


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


# In[ ]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[ ]:


x_train=train_datagen.flow_from_directory(r'C:\Users\dell\Downloads\Dataset-20221027T163526Z-001.zip/TRAIN_SET',target_size=(64, 64),batch_size=5,class_mode='sparse')


# In[ ]:


x_test=test_datagen.flow_from_directory(r'"C:\Users\dell\Downloads\Dataset-20221027T163526Z-001\Dataset\TEST_SET"',target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='sparse')


# In[ ]:


print(x_train.class_indices)


# In[ ]:


print(x_test.class_indices)


# In[ ]:


from collections import Counter as c
c(x_train.labels)


# In[ ]:


import numpy as np


# In[ ]:


import tensorflow


# In[ ]:


from tensorflow.keras.models import Sequential


# In[ ]:


from tensorflow.keras import layers


# In[ ]:


from tensorflow.keras.layers import Dense,Flatten


# In[ ]:


from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


model=Sequential()


# In[ ]:


classifier=Sequential()


# In[ ]:


classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64 ,3), activation='relu'))


# In[ ]:


classifier.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


classifier.add(Conv2D(32, (3, 3), activation='relu'))


# In[ ]:





# In[ ]:


classifier.add(Flatten())


# In[ ]:


classifier.add(Dense(units=128, activation='relu'))


# In[ ]:


classifier.add(Dense(units=5, activation='softmax'))


# In[ ]:


classifier.summary()


# In[ ]:


classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


classifier.save('nutrition.h5')


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


from keras.preprocessing import image


# model = load_model("nutrition.h5")

# In[ ]:


from flask import Flask,render_template,request


# In[ ]:


import os


# In[ ]:


import numpy as np


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


from tensorflow.keras.preprocessing import image


# In[ ]:


import requests


# In[ ]:


app= Flask(__name__,template_folder="templates")


# In[ ]:


model=load_model('nutrition.h5')


# In[ ]:


print("Loaded model from disk")


# In[ ]:


@app.route('/')
def home():
    return render_template('homepage.html')


# In[ ]:


@app.route('/image1',methods=['GET','POST'])
def image1():
    return render_template("image.html")


# In[ ]:


@app.route('/predict',methods=['GET','POST'])
def launch():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname('__file__')
        filepath=os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)
        
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        
        pred=np.argmax(model.predict(x), axis=1)
        print("prediction",pred)
        index=['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON']
        result=str(index[pred[0]])
        x=result
        print(x)
        result=nutrition(result)
        print(result)
        return render_template("0.html",showcase=(result),showcase1=(x))


# In[ ]:


def nutrition(index):
    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"
    querystring = {"query":index}
    headers = {
        'X-RapidAPI-Key': 'Ffla3txogP9-H3DjCu7Z7XJPnb4Xms6WbWu49q6Wj2VE',
    'X-RapidAPI-Host': 'calorieninjas.p.rapidapi.com'
    }
    response = requests.request("GET",url,headers=headers, params=querystring)
    print(response.text)
    return response.json()['items']


# In[ ]:


if __name__ == "__main__":
    app.run(debug=False)


# In[ ]:





# In[ ]:





# In[ ]:




