import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

import base64
import streamlit as st

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:background/background.avif;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background("C:/Users/Hema/Desktop/Major Project(Brain Stroke)/Sourcecode/background/1.png")
import streamlit as st
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
from numpy import argmax
import matplotlib.pyplot as plt

# Create an empty container
placeholder = st.empty()

actual_email = "admin"
actual_password = "123"

# Insert a form in the container
with placeholder.form("login"):
    st.markdown("#### Enter your credentials")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if email == actual_email and password == actual_password:

    placeholder.empty()
    st.success("Login successful")
    # def import_and_predict(image_data, model):
        
    #         size = (65,65)    
    #         image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    #         image = image.convert('RGB')
    #         image = np.asarray(image)
    #         image = (image.astype(np.float32) / 255.0)

    #         img_reshape = image[np.newaxis,...]

    #         prediction = model.predict(img_reshape)
            
    #         return prediction

    model = tf.keras.models.load_model('C:/Users/Hema/Desktop/Major Project(Brain Stroke)/Sourcecode/model.h5')

    st.write("""
             # Brain stroke stages detection using deep learning models
             """
             )

    st.write("Brain stroke ")

    file = st.file_uploader("upload an image file", type=["jpg", "png"])
    #
    if file is None:
        st.text("please select a image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        HEIGHT = 65
        WIDTH = 65
        N_CHANNELS = 3
        size=(65,65)
        test_image_o = ImageOps.fit(image, size, Image.ANTIALIAS)
        categories = ['Hemorrhagic','Ischemic','no_stroke','TIA'  ]


        model = tf.keras.models.load_model('C:/Users/Hema/Desktop/Major Project(Brain Stroke)/Sourcecode/model.h5')
        #test_data.append(test_image)
        # scale the raw pixel intensities to the range [0, 1]
        test_data = np.array(test_image_o, dtype="float") / 255.0
        test_data=test_data.reshape([-1,65, 65, 3])
        pred = model.predict(test_data)
        predictions = argmax(pred, axis=1) # return to label
        print ('Prediction : '+categories[predictions[0]])

        st.write ('Prediction : '+categories[predictions[0]])
        #Imersing into the plot
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:white')
        plt.title(categories[predictions[0]])
        plt.imshow(test_image_o)
        prediction = categories[predictions[0]]

    
else:
    print("login failed ")



#cd "C:\Users\Hema\.streamlit"
# streamlit run "C:/Users/Hema/Desktop/Major Project(Brain Stroke)/Sourcecode/app.py"
    
    