import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import adam_v2
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model

ticker = "MSFT"
df = pd.read_csv("../MSFT.csv", na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True);
# print(df.head())

# Print the shape of Dataframe  and Check for Null Values
# print("Dataframe Shape:", df.shape)
# print("Null Value Present: ", df.isnull().values.any())

# df['Adj Close'].plot()
# Keeps graph window from closing upon program termination
# plt.show()

output = pd.DataFrame(df['Adj Close'])
# Selecting Training Variables
training_vars = ['Open', 'High', 'Low', 'Volume']

# Scales initial data down for memory consumption
scale = MinMaxScaler()
feature_transform = scale.fit_transform(df[training_vars])
feature_transform = pd.DataFrame(columns=training_vars, data=feature_transform, index=df.index)
feature_transform.head()

# Programming the training model to price predict
# Splitting to Training set and Test set
timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[
                                                            len(train_index): (len(train_index) + len(test_index))]
    y_train, y_test = output[:len(train_index)].values.ravel(), output[len(train_index): (
                len(train_index) + len(test_index))].values.ravel()

trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes=True, show_layer_names=True)

#LSTM Prediction
y_pred= lstm.predict(X_test)

plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title(ticker,"Predicition")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()