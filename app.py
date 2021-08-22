# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 18:06:23 2021

@author: sachin h s
"""

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()



st.title('IMAGE CLASSIFIER')

st.markdown("Welcome to this simple web application that classifies images. The images are classified into six different classes namely: buildings, forest, glacier, mounatain,sea and street.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "my_model.hdf5"
    IMAGE_SHAPE = ((128, 128,3))
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((128,128))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names=['buildings',
                 'forest',
                 'glacier',
                 'mountain',
                 'sea',
                 'street']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'buildings': 0,
          'forest': 0,
          'glacier': 0, 
          'mountain': 0, 
          'sea': 0,
          'street':0
}
    

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result



if __name__ == "__main__":
    main()