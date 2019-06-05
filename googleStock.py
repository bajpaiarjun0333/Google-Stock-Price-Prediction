#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:26:08 2019

@author: bajpaiarjun0333
"""
#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd

#IMPORTING THE TRAINING SET OF THE OPEN STOCK PRICE
dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values

#Feature Scaling 
#Applying the normalizition of the data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
#Creating the timesteps datastructure for the keras to accept
#For each timestep we consider the data for the previous 60 timesteps
X_train=[]
Y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])

#conveting them into the numpy arrays
X_train=np.array(X_train)
Y_train=np.array(Y_train)

#Reshaping the array to the keras standards
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#importing the keras layers regularization and lstm
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential

#initializing the model
regressor=Sequential()
#The very first layer of the lstm
#Input shape is required only for first layer
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#Building it deep
#Second layer of the lstm
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the third  layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the fourth lstm layer 
#return_sequences is kept false in the last layer of the model
regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))

#Adding the output layer
#output is only one dimensional
regressor.add(Dense(1))

#compiling the model
regressor.compile('adam',loss='mean_squared_error')
regressor.fit(X_train,Y_train,epochs=100,batch_size=32)

#Importing the data to test the trained model on 
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_price=dataset_test.iloc[:,1:2].values

#Getting the predicted price for january 2017
#since to predict the price we may provide  the information about last 60 days
#We need to do some concation of the datasets

#We must always preserve the actual test values
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_price=regressor.predict(X_test)
predicted_price=sc.inverse_transform(predicted_price)

#Visualizing the results obtained
plt.plot(real_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_price,color='blue',label='Predicted Google Stoct Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



        




    

