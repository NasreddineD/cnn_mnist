import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

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
def viz_num(num):
    #Reshape the 768 values to a 28x28 image
    image = X_raw_final[num].reshape([28,28])
    fig = plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis("off")
    fig.show()
    return fig

st.markdown(""" <style> img {
width:200px !important; height:200px;} 
</style> """, unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'best_model.h5')
if not os.path.isdir(MODEL_DIR):
    os.system('runipy train.ipynb')

model = load_model("C:/Users/Simplon/Google Drive/Nasreddine/Arturo/14 - DEEP LEARNING/03 - MNIST/git/best_model.h5")

st.title('Reconnaissance de Chiffre Dessiné')


SIZE = 192
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

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    image = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    image = image.convert('L')
    image = (tf.keras.utils.img_to_array(image)/255)
    image = image.reshape(1,28,28,1)
    test_x = tf.convert_to_tensor(image)

if st.button('Predict'):
    val = model.predict(test_x)
    st.write(f'result: {np.argmax(val[0])}')



st.title('Reconnaissance de Chiffre Aléatoire')


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data_test = pd.read_csv(uploaded_file)
    X_raw_final = data_test.values
    X_test_final = data_test.values.reshape(data_test.shape[0], 28, 28, 1)
    pred_testing = model.predict(X_test_final)
    pred_testing = np.argmax(pred_testing, axis=1)

    if st.button('Predict a random image from our dataframe'):
        random_number = np.random.choice(28000)
        st.write('Picture number ' + str(random_number))
        st.write('Predicted number : ' + str(pred_testing[random_number]))
        viz = viz_num(random_number)
        st.pyplot(viz)
