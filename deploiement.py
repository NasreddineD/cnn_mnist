#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:

import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

# Data manip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential # initialize neural network library
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten # build our layers library
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import save_model, load_model


@st.cache(allow_output_mutation=True)
def load_data():
   data = pd.read_csv("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/data/test.csv")
   data = data/255
   data = np.array(data)
   data = data.reshape((data.shape[0], 28, 28, 1)) # Réduction des données et reshape 
   return data

data = load_data()

@st.cache(allow_output_mutation=True)
def load_pred():
    model = load_model("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/best_model.h5")
    prediction = model.predict(data)
    prediction = np.argmax(prediction, axis=1)
    return prediction

prediction = load_pred()

# # load dataset
# data = pd.read_csv("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/data/test.csv")

# # Transform into 
# data = np.array(data)

# # Normalize the data
# data = data/255

# # Reshape dataset to have a single channel
# data = data.reshape((data.shape[0], 28, 28, 1))

# # Load best model
# model = load_model("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/best_model.h5")

# # Make the prediction
# prediction = model.predict(data)
# prediction = np.argmax(prediction, axis=1)

## Number visualization
def viz_num(num):
    #Reshape the 768 values to a 28x28 image
    image = data[num].reshape([28,28])
    fig = plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis("off")
    fig.show()
    return fig
#########################################################################################################################
# Streamlit Canvas drawing

st.title("Reconnaissance d'image dessinée")

MODEL_DIR = os.path.join("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST", 'best_model.h5')
model = load_model("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/notebooks/model.h5")
SIZE = 192

col1, col2, col3 = st.columns(3)

with col1:
    mode = st.checkbox("Draw (or Delete)?", True)
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key='canvas')

with col2 :
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)

with col3:
    if st.button('Predict'):
        test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val = model.predict(test.reshape(1,28,28,1))
        st.write(f'result: {np.argmax(val[0])}')

#########################################################################################################################
# Streamlit Dataset test

st.title("Reconnaissance d'image aléatoire")

if st.button('Predict a random image from our dataframe'):
    index = np.random.choice(data.shape[0])
    st.write('Picture number ' + str(index))
    st.write('Predicted number : ' + str(prediction[index]))
    viz = viz_num(index)
    st.pyplot(viz)