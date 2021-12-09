import os
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'best_model.h5')
reconstructed_model = tf.keras.models.load_model(MODEL_DIR)

data_test = os.path.join(os.path.dirname(__file__), 'test.csv')
csv_downloaded = pd.read_csv(data_test)

# reconstructed_model = tf.keras.models.load_model("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/git/best_model.h5")

#Importer un csv
# csv_downloaded = pd.read_csv('C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/git/test.csv',delimiter=',', decimal='.')

# preprocessing des inputs
def preprocess(input):
    input_preprocess = input / 255
    input_preprocess = np.array(input).reshape((-1, 28, 28, 1))
    return input_preprocess

# afficher l'image en input
def see_img(input):
    image = np.array(input).reshape([28,28])
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.get_cmap('gray'))
    st.pyplot(fig)

def my_prediction():
    # choisir une ligne au hasard dans le dataframe test
    my_input = csv_downloaded.sample(n=1)

    # afficher l'image en input
    see_img(my_input)

    # afficher la sortie cad le numero predit : print(argmax)
    my_predict = np.argmax(reconstructed_model.predict(preprocess(my_input)), axis=1)
    return str(my_predict)[1]

if st.button('Prédis moi un chiffre'):
    st.write("Je vois un "+my_prediction())