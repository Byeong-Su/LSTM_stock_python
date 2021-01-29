import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

stock_df = pd.read_csv('C:/Users/neado/Downloads/005930.KS.csv')
stock_df.head()

stock_df.tail()

stock_df.info()

stock_df = stock_df.dropna()
stock_df.info()

stock_df['average'] = (stock_df['High'] + stock_df['Low'])/2
stock_df.head(2)

plt.plot(stock_df['Volume'])
plt.plot(stock_df['average'])

#거래량과 평균값으로만 데이터 구성
stock = stock_df[['Volume','average']].values
stock.shape

#정규화
stock = (stock - stock.min(axis=0)) / (stock.max(axis=0) - stock.min(axis=0))
#정규화 결과 출력
stock[:2]

#학습용 데이터 생성
lookback = 50
X = []
Y = []

#첫번째는 stock[0:50](stock의 0~50), 두번째는 stock[1:50](stock의 1~50)...순으로 반복해서 X에 붙임
for i in range(len(stock)-lookback):
    X.append(stock[i:i+lookback])
    Y.append(stock[i+lookback,[1]])
   
X = np.array(X)
Y = np.array(Y)

X.shape, Y.shape

stock[50]
stock[50,[1]]

X_train = X[:-50-lookback]
X_test = X[-50:]
Y_train = Y[:-50-lookback]
Y_test = Y[-50:]

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


model = Sequential()

model.add(BatchNormalization(axis=1, input_shape=X_train.shape[1:]))

model.add(LSTM(32, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=True, activation='relu'))
model.add(LSTM(32))

model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

h=model.fit(X_train, Y_train, batch_size=5, epochs=10, validation_data=[X_test,Y_test])

pred_y = model.predict(X_test)

plt.figure(figsize=[12,6])
plt.plot(pred_y.ravel(), 'r-', label='pred_y')
plt.plot(Y_test.ravel(), 'b-', label='Y_test')
plt.plot((pred_y-Y_test).ravel(), 'g-', label='diff*10')
plt.legend()
plt.title('Samsung - stock price')

plt.plot(h.history['loss'], label='loss')
plt.plot(h.history['val_loss'],label='val_loss')
plt.legend()
plt.title('Loss')
