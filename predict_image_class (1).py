import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py


st.header('IMAGE CLASS PREDICTOR')

def main():
  file_uploaded=st.file_uploader("choose the file",type=['jpg','png','jpeg'])
  if file_uploaded is not None:
    image=Image.open(file_uploaded)
    figure=plt.figure()
    plt.imshow(image)
    plt.axis('off')
    result =predict_class(image)
    st.write(result)
    st.pyplot(figure)

    
 def prdict_class(image):
  classifier_model=tf.keras.models.load_model(r"/content/gdrive/MyDrive/intel_image_classification_data/my_model.hdf5"),
  shape=((128,128,3))
  model=tf.keras.Sequential(hub[hub.KerasLayer(classifier_model, input_shape=shape)])
  test_image=image.resize((128,128))
  test_image=test_image/255.0
  test_image=np.expand_dims(test_image,axis=0)
  class_name=['buildings',
              'forest',
              'glacier',
              'mountain',
              'sea',
              'street']
  prediction=model.predict(test_image)
  scores=tf.nn.softmax(prediction[0])
  scores=scores.numpy()
  image_class=class_names[np.argmax(scores)]
  result="The IMAGE uploaded is:{}".format(image_class)
  return result

if __name__== "__main__":
  main()
