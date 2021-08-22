# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 01:12:28 2021

@author: sachin h s
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pathlib


import tensorflow as tf
tf.test.gpu_device_name()


from tensorflow.keras.preprocessing.image import ImageDataGenerator
rescaled=rescalesd=ImageDataGenerator(1/255)
train_fed= rescaled.flow_from_directory(r"C:\Users\sachin h s\Downloads\intel_image_classification_data\seg_train\seg_train",target_size=(128,128),batch_size=12,class_mode='categorical')
test_fed= rescaled.flow_from_directory(r"C:\Users\sachin h s\Downloads\intel_image_classification_data\seg_test\seg_test",target_size=(128,128),batch_size=12,class_mode='categorical')

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)),
                                  tf.keras.layers.MaxPool2D(2,2),

                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),

                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),

                                  
                                  tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                  tf.keras.layers.MaxPool2D(2,2),

                                  tf.keras.layers.Flatten(),


                                  tf.keras.layers.Dense(128,activation='relu'),
                                  tf.keras.layers.Dropout(0.5),
                                  tf.keras.layers.Dense(6,activation='softmax')
                                  


])

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import h5py

erl_stop=EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True)
mod_chk=ModelCheckpoint(filepath=r"C:\Users\sachin h s\Downloads\intel_image\models\my_model.h5",monitor='val_loss',save_best_only=True)
lr_rate=ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.1)


hist = model.fit_generator(train_fed,shuffle=True,epochs=15,validation_data=test_fed,
                           callbacks=[erl_stop,mod_chk,lr_rate],verbose=2)

acc=model.evaluate(test_fed,steps=len(test_fed),verbose=2)
print("%.2f"%(acc[1]*100))

