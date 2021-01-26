import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
#from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnplateau
import datetime

#Load Dataset
data=pd.read_csv('C:/Users/neado/Downloads/005930.KS.csv')
data.head() #데이터의 앞부분 5개를 로드해옴

#Compute Mid Price
high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2

#Create Windows
seq_len = 50        #예측시 사용할 일수
sequence_length = seq_len +1

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index:index + sequence_length])

#Normalize Data
normalized_data = []
for window in result:
    normailzed_window=[((float(p)/float(window[0]))-1)for p in window]
    normalized_data.append(normailzed_window)

result = np.array(normalized_data)

#split train and test data(아래에서는 학습용90%, 테스트용10%)
row = int(round(result.shape[0] * 0.9))
train=result[:row,:]
np.random.shuffle(train)

x_train = train[:,:-1]
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))
y_train = train[:,-1]

x_test = result[row:,:-1]
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:,1]

x_train.shape, x_test.shape

#Build a Model
model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(50,1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse',optimizer='rmsprop')
model.summary()

#Training
model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=10,epochs=20)

#Prediction
pred=model.predict(x_test)
fig=plt.figure(facecolor='white')
ax=fig.add_subplot(111)
ax.plot(y_test,label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()