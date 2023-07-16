import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error

import pickle


def preProcess(df):
    df1=df.reset_index()['close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    return df1,train_data, test_data,scaler


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0] 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

def stacked_lstm(X_train,y_train,X_test,y_test,m):
    # model=Sequential()
    # model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    # model.add(LSTM(50,return_sequences=True))
    # model.add(LSTM(50))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error',optimizer='adam')
    # model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)
    with open(m,'rb') as f:
        model=pickle.load(f)
    return model


def training(model,X_train,X_test,scaler):
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    return train_predict,test_predict


def plot(train_predict, test_predict,df1,scaler):
    look_back=100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig('static/images/graph1.png')
    plt.close()


def predict(model, test_data, df1, scaler):
    x_input=test_data[340:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    df2=df1.tolist()
    df2.extend(lst_output)
   
    df2=scaler.inverse_transform(df2).tolist()
    plt.plot(df2[len(df2)-30:])
    plt.savefig('static/images/graph2.png')
    plt.close()
    plt.plot(df2[len(df2)-100:])
    plt.savefig('static/images/graph3.png')
    plt.close()