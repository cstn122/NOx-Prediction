# Merging RNN model and communication with ThingsBoard

# Core Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN

# For data conditioning
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

# Make results reproducible
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.compat.v1.set_random_seed(1)

# Other essential libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import array

# MQTT communication
import paho.mqtt.client as paho             #mqtt library
import os
import json
import time
from datetime import datetime
# =================================================================================

def prediction():
    # Set input number of timestamps and training days
    n_timestamp = 5
    train_days = 1500  # number of data to train from
    testing_days = 500  # number of data to be predicted
    n_epochs = 10
    filter_on = 0

    # import data
    file = "nox.csv"
    dataset = pd.read_csv(file)
    print(dataset)
    if filter_on == 1:
        dataset['NOx'] = medfilt(dataset['NOx'], 3)
        dataset['NOx'] = gaussian_filter1d(dataset['NOx'], 1.2)

    # Set number of training and testing data
    train_set = dataset[0:train_days].reset_index(drop=True)
    test_set = dataset[train_days: train_days + testing_days].reset_index(drop=True)
    training_set = train_set.iloc[:, 1:2].values
    print(training_set.shape)
    testing_set = test_set.iloc[:, 1:2].values

    # Normalize data
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    testing_set_scaled = sc.fit_transform(testing_set)

    # Split data into n_timestamp
    def data_split(sequence, n_timestamp):
        X = []
        y = []
        for i in range(len(sequence)):
            end_ix = i + n_timestamp
            if end_ix > len(sequence) - 1:
                break
            # i to end_ix as input
            # end_ix as target output
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    X_train, y_train = data_split(training_set_scaled, n_timestamp)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test, y_test = data_split(testing_set_scaled, n_timestamp)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Model type: SimpleRNN
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(SimpleRNN(50, activation='relu'))
    model.add(Dense(1))

    # Start training
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=32)
    loss = history.history['loss']
    epochs = range(len(loss))

    # Get predicted data
    y_predicted = model.predict(X_test)

    # 'De-normalize' the data
    y_predicted_descaled = sc.inverse_transform(y_predicted)
    y_train_descaled = sc.inverse_transform(y_train)
    y_test_descaled = sc.inverse_transform(y_test)
    y_pred = y_predicted.ravel()
    y_pred = [round(yx, 2) for yx in y_pred]
    y_tested = y_test.ravel()

    mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
    r2 = r2_score(y_test_descaled, y_predicted_descaled)
    print("mse= " + str(round(mse, 2)))
    print("r2= " + str(round(r2, 2)))

    # Show results
    plt.figure(figsize=(8, 7))

    # print(dataset.shape)
    plt.subplot(3, 1, 1)  # 311
    plt.plot(dataset['NOx'], color='black', linewidth=1, label='True value')
    plt.ylabel("Heating Value")
    plt.xlabel("Time")
    plt.title("All data")

    plt.subplot(3, 2, 3)  # 323
    plt.plot(y_test_descaled, color='black', linewidth=1, label='True value')
    plt.plot(y_predicted_descaled, color='red', linewidth=1, label='Predicted')
    plt.legend(frameon=False)
    plt.ylabel("Heating Value")
    plt.xlabel("Time")
    plt.title("Predicted data (n minumtes)")

    plt.subplot(3, 2, 4)  # 324
    plt.plot(y_test_descaled[0:75], color='black', linewidth=1, label='True value')
    plt.plot(y_predicted_descaled[0:75], color='red', label='Predicted')
    plt.legend(frameon=False)
    plt.ylabel("Heating Value")
    plt.xlabel("Time")
    plt.title("Predicted data (first 75 minumtes)")

    plt.subplot(3, 3, 7)  # 337
    plt.plot(epochs, loss, color='black')
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.title("Training curve")

    plt.subplot(3, 3, 8)  # 338
    plt.plot(y_test_descaled - y_predicted_descaled, color='black')
    plt.ylabel("Residual")
    plt.xlabel("Time")
    plt.title("Residual plot")

    plt.subplot(3, 3, 9)  # 339
    plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
    plt.ylabel("Y true")
    plt.xlabel("Y predicted")
    plt.title("Scatter plot")

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()

    return y_predicted_descaled

# ========================================================================
def comm(pred_result):
    for i in (pred_result):
        ACCESS_TOKEN = 'XYEkR5gMNEETUOpnxx6B'                 #Token of your device
        broker = "demo.thingsboard.io"                        #host name
        port = 1883                                           #data listening port

        def on_publish(client, userdata, result):             #create function for callback
            # print("data published to thingsboard \n")
            pass
        client1 = paho.Client("control1")                    #create client object
        client1.on_publish = on_publish                      #assign function to callback
        client1.username_pw_set(ACCESS_TOKEN)                #access token from thingsboard device
        client1.connect(broker, port, keepalive=60)          #establish connection
        
        payload = "{"
        payload += "\"prediction\":%f" %i[0]
        payload += "}"
        ret = client1.publish("v1/devices/me/telemetry", payload)   #topic:v1/devices/me/telemetry
        print("Please check LATEST TELEMETRY field of your device")
        print(payload)
        time.sleep(5)
# ================================================================================================

if __name__ == "__main__":
    pred_result = prediction()
    to_thingsboard = comm(pred_result)
