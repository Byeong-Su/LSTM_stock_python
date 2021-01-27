import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

#Load Dataset
data = pd.read_csv('C:/Users/neado/Downloads/005930.KS.csv')
data.head() #데이터 프레임의 맨앞 5개 출력

#결측치 제거(제거하지 않으면 loss값이 nan으로 나옴)
dataset = data.dropna()

#Compute Mid Price
high_prices = dataset['High'].values
low_prices = dataset['Low'].values
mid_prices = (high_prices + low_prices) / 2

#Create Windows
seq_len = 50    #Window 크기
sequence_length = seq_len + 1    #50개를 만들고 1개를 예측 => 총 51개

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index: index + sequence_length])


#Normalize Data
normalized_data = []
#윈도우의 젤 첫번째 값을 0으로 잡고 나머지를 첫번째값과 상대적으로 측정
for window in result:
    normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)    #정규화된 결과값

# split train and test data
#아래는 학습용 90% 테스트용 10%로 나눔
row = int(round(result.shape[0] * 0.9))    #학습용 데이터셋
train = result[:row, :]
np.random.shuffle(train)    #학습용 데이터셋 랜덤으로 섞어줌

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape


#Build a Model
model = Sequential()

#첫번째 LSTM (유니스(?)가 50)
model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))

#두번째 LSTM (유니스(?)가 64)
model.add(LSTM(64, return_sequences=False))

#다음날 하루의 데이터를 예측
model.add(Dense(1, activation='linear'))

#손실함수는 Mean Squared Error 옵티마이저는 rsmprop
model.compile(loss='mse', optimizer='rmsprop')

model.summary()

#Training
#batch size : 한번에 묶어서 학습시킬 양, epochs : 반복학습 시킬 횟수
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=20)

#Prediction
pred = model.predict(x_test)    #테스트데이터를 예측한 결과값

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()
