#Importing lib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#Getting necessary data
train_data = pd.read_csv('train.csv')
X = train_data.iloc[:,1:].values
y = train_data.iloc[:,0].values

X[X>0]=255

X_reshape = X.reshape(-1,28,28,1)
#plt.imshow(X_reshape[498][:,:,0])

#OneHotEncoding y-label
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()

#Test-cross-val split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.model_selection import train_test_split
X_train_reshape, X_test_reshape, y_train_reshape, y_test_reshape = train_test_split(X_reshape, y, test_size=0.25, random_state=0)

#Normalization
X_train = X_train/255
X_test = X_test/255

X_train_reshape = X_train_reshape/255
X_test_reshape = X_test_reshape/255
X_aug = X_train_reshape.copy()

print(X_train_reshape.shape)
#=====================================================================#
#ImageDataGen
train_datagen = ImageDataGenerator(
        rotation_range=10,
        shear_range=0.2,
        zoom_range = 0.10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False)

train_datagen.fit(X_aug)
X_train_reshape = np.concatenate((X_train_reshape, X_aug),0)
y_train = np.concatenate((y_train, y_train),0)
print(X_train_reshape.shape)
#Convolution NN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=[28,28,1]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))  
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=200, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))
# cnn.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
cnn.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_reshape, y_train, validation_data=(X_test_reshape,y_test), batch_size=64,epochs = 5)

cnn.save('digit_recog_cnn_1.h5')
#========================================================================#

# =============================================================================
# 
# #X_train
# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=784, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
# ann.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
# 
# print("Starting...")
# ann.fit(X_train,y_train, validation_data=(X_test,y_test) ,epochs = 10)
# print("Done")
# 
# ann.save('digit_recog_ann_1.h5')
# =============================================================================
