# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:12:53 2018

@author: BEP
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot
import os
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from keras.utils import np_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pa
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
K.set_image_dim_ordering('th')


#fixrandom seed for reproducibility
seed = 7
np.random.seed(seed)

#readimag


Train_dir='trainF'
Img_size=28
num_pixel=Img_size*Img_size
batch_size=100
num_epoch=150
#bach 200 epochc 10  
print("batch size ",batch_size)
print ("num epochs",num_epoch)

def label_img(img):
    img_label=img.split('.')[-3]
    
    if img_label=='cat':
        return 0
    elif  img_label=='dog':
        return 1

#create train data 
def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(Train_dir)):
        label=label_img(img)
        path=os.path.join(Train_dir,img)
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(Img_size,Img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    return training_data
  
train_data=create_train_data()
images=np.array([i[0] for i in train_data])
labels=[i[1] for i in train_data]


images = images.reshape(images.shape[0], 1,Img_size,Img_size)
#function randomly flip the image for geting more data 
def Randomflip(images):
    #create randrom flip
    randflip= ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    # fit parameters from data
    randflip.fit(images)
    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in randflip.flow(images,labels, batch_size=5000):
        break
    return X_batch,y_batch
#function geting more data by rotation
def Rotation(images):
    #create rotation to 90 degree
    datagen_Rotation2= ImageDataGenerator(rotation_range=90)
    # fit parameters from data
    datagen_Rotation2.fit(images)
    for X_batch_rot2, y_batch_rot2 in datagen_Rotation2.flow(images,labels, batch_size=6000):
        break
    return X_batch_rot2, y_batch_rot2 
#function whitining the redendency of the image
def Zca_Whitening(images):    
    #zca whitiening the redencdency of the image
    datagen_zca = ImageDataGenerator(zca_whitening=True)
    # fit parameters from data
    datagen_zca.fit(images)
    for X_batch_zca, y_batch_zca in datagen_zca.flow(images,labels, batch_size=7000):
        break
    return X_batch_zca, y_batch_zca
#function shiting the image to the center with 0.2
def Shift_Img(images):
    #shift the image with 0.2
    shift = 0.2
    datagen_shift = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
    # fit parameters from data
    datagen_shift.fit(images)
    for X_batch_shift, y_batch_shift in datagen_shift.flow(images,labels , batch_size=6000):
        break
    return X_batch_shift, y_batch_shift

X_batch,y_batch=Randomflip(images)
X_batch_rot2, y_batch_rot2 =Rotation(images)
X_batch_zca, y_batch_zca=Zca_Whitening(images)
X_batch_shift, y_batch_shift=Shift_Img(images)

conimgnew=np.concatenate((images,X_batch,X_batch_rot2,X_batch_zca,X_batch_shift))
conlab=np.concatenate((labels,y_batch,y_batch_rot2,y_batch_zca,y_batch_shift))



X_train, X_test, y_train, y_test = train_test_split(conimgnew,conlab, random_state=1)
# flatten 28*28 images to a 784 vector for each image
X_train = X_train.reshape(X_train.shape[0],1,Img_size,Img_size).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,Img_size,Img_size).astype('float32')


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the  model
def catvsdog_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),  input_shape=(1, Img_size, Img_size), activation='relu'))
    model.add(Conv2D(32, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes,activation='sigmoid'))
    rms=optimizers.RMSprop(lr=1e-4)

    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model
       

# build the model
model = catvsdog_model()
# Fit the model
hist=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=batch_size)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
print("Accuracy: %.2f%%" % (scores[1]*100))

 
# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(5,3))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(5,3))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)
  
Y_predt = model.predict(X_train)
#print(Y_pred)
y_predt = np.argmax(Y_predt, axis=1)
#                       (or)

#y_pred = model.predict_classes(X_test)
#print(y_pred)

p=model.predict_proba(X_test) # to predict probability

target_names = ['class 1', 'class 2']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

print(classification_report(np.argmax(y_train,axis=1), y_predt,target_names=target_names))
print(confusion_matrix(np.argmax(y_train,axis=1), y_predt))






	








