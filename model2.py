import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yh as web
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def prediction_model(company):
    # Load Data via Ticker Symbol
    company = company

    start = dt.datetime(2017, 1, 1)
    end = dt.datetime.now()
    # Pulls Data from Yahoo
    data = web.DataReader(company, 'yahoo', start, end)

    # Prepare Data - Scales down the numbers for better memory consumption
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Number of past days used to predict future prices
    p_days = 60

    # Prepare training data
    x_train, y_train = [], []

    for x in range(p_days, len(scaled_data)):
        x_train.append(scaled_data[x-p_days:x, 0])
        y_train.append(scaled_data[x,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Building Prediction Model using LSTM (Long Term Short Term Memory)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Provide prediction of the next closing price
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    ''' Performance/Accuracy Check based on Pre-Existing Past Data '''
    # Loading Test Data
    t_start = dt.datetime(2020, 1, 1)
    t_end = dt.datetime.now()

    t_data = web.DataReader(company, 'yahoo', t_start, t_end)
    actual_prices = t_data['Close'].values

    # Training set combines training data with test data
    total_data = pd.concat((data['Close'], t_data['Close']),axis=0)

    # Used by model to predict prices
    model_input = total_data[len(total_data) - len(t_data) - p_days:].values
    model_input = model_input.reshape(-1, 1)
    model_input = scaler.transform(model_input)

    # Checks Model Performance on Test Data
    x_test = []

    for i in range(p_days, len(model_input)):
        x_test.append(model_input[i-p_days:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Given the initial scaling process, the prices must be inverted
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plotting Tested Predictions based on past data
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Prices")
    plt.title(f"{company} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.show()

# Predict the following days close price
    rdata = [model_input[len(model_input) - p_days:len(model_input + 1), 0]]
    rdata = np.array(rdata)
    rdata = np.reshape(rdata, (rdata.shape[0], rdata.shape[1], 1))
    p_price = model.predict(rdata)
    p_price = scaler.inverse_transform(p_price)
    print(f"{company} Prediction: ${p_price}")




