import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
# Download the stock price data from Yahoo Finance
df = pd.read_csv('AAPL.csv', index_col='Date')

# Extract the closing price
closing_price = df['Close']

# Split the data into training and testing sets
train_size = int(len(closing_price) * 0.8)
train_data = closing_price[:train_size]
test_data = closing_price[train_size:]
# Scale the data
scaler = tf.keras.preprocessing.MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Create the sequence data for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return sequences

# Create the sequences for training and testing sets
sequence_length = 50
train_sequences = create_sequences(scaled_train_data, sequence_length)
test_sequences = create_sequences(scaled_test_data, sequence_length)

# Convert the sequences into NumPy arrays
train_sequences = np.array(train_sequences)
test_sequences = np.array(test_sequences)

# Reshape the data for LSTM
train_sequences = train_sequences.reshape((train_sequences.shape[0], train_sequences.shape[1], 1))
test_sequences = test_sequences.reshape((test_sequences.shape[0], test_sequences.shape[1], 1))
# Create the sequential model
model = Sequential()

# Add the LSTM layer
model.add(LSTM(128, input_shape=(sequence_length, 1)))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')
# Train the model on the training data
model.fit(train_sequences, train_data[sequence_length:], epochs=100)
# Evaluate the model on the test data
test_predictions = model.predict(test_sequences)

# Calculate the mean squared error (MSE)
mse = np.mean((test_predictions - test_data[sequence_length:])**2)

# Print the MSE
print('MSE:', mse)
# Make a prediction for the next day
next_day_prediction = model.predict(test_sequences[-1:])

# Inverse scale the prediction
inverse_scaled_prediction = scaler.inverse_transform(next_day_prediction)

# Print the prediction
print('Prediction for next day:', inverse_scaled_prediction)
