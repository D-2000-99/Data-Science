import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#Importing test data
test_data_X = np.array(pd.read_csv('test.csv'))

#Visualizing
for i in range(0,9):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data_X[i].reshape(28,28))
    plt.axis('off')
plt.show()

#Model loading
model = tf.keras.models.load_model('digit_recog_cnn.h5')
pred = model.predict(test_data_X[0].reshape(-1,28,28,1))
result = np.where(pred[0]==1)[0][0]

plt.imshow(test_data_X[0].reshape(28,28))
plt.title('Prediction: '+str(result))
print('Prediction: ', result)