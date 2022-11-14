#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.models import load_model


# In[3]:


from keras.preprocessing import image


# In[4]:


from flask import Flask,render_template,request


# In[5]:


import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests


# In[6]:


from google.colab import drive
drive.mount('/content/drive')


# In[7]:


app= Flask(__name__,template_folder="templates")


# In[8]:


from tensorflow import keras
model = keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/Dataset-20221108T081455Z-001/Dataset/nutrition.h5')


# In[9]:


print("Loaded model from disk")


# In[10]:


@app.route('/')
def home():
    return render_template('homepage.html')


# In[11]:


@app.route('/image1',methods=['GET','POST'])
def image1():
    return render_template("image.html")


# In[12]:


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


# In[13]:


def nutrition(index):
    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"
    querystring = {"query":index}
    headers = {
        'X-RapidAPI-Key': 'daaf576556msh5fcbc747e5cb27cp14bd10jsn07d05ab509ae',
    'X-RapidAPI-Host': 'calorieninjas.p.rapidapi.com'
    }
    response = requests.request("GET",url,headers=headers, params=querystring)
    print(response.text)
    return response.json()['items']


# In[ ]:


if __name__ == "__main__":
    app.run(debug=False)

