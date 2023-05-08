import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def load_dataset():
    df = pd.read_csv('house_price.csv', index_col='Date')
    return df

dataset = load_dataset()

input_num = 1
output_num = 1
context_unit = 3 # Hidden Layer in RNN

# How many previous datas needed in predicting the next prediction
time_seq = 3

test_size = 30

minMaxScaler = MinMaxScaler()
norm_data = minMaxScaler.fit_transform(dataset)

train_data, test_data = train_test_split(norm_data, test_size=0.3, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)

def create_dataset(data, time_seq):
    x, y = [], []
    for i in range(len(data) - time_seq):
        x.append(data[i:i + time_seq, 0])
        y.append(data[i + time_seq, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(train_data, time_seq)
x_val, y_val = create_dataset(val_data, time_seq)

x_train = x_train.reshape((-1, time_seq, input_num))
y_train = y_train.reshape((-1, output_num))
x_val = x_val.reshape((-1, time_seq, input_num))
y_val = y_val.reshape((-1, output_num))

# Input to Hidden
cell = tf.keras.layers.SimpleRNN(units=context_unit, activation='relu', return_sequences=False)

model = tf.keras.Sequential([
    cell,
    tf.keras.layers.Dense(units=output_num, activation='relu')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss=tf.keras.losses.MeanSquaredError())

epoch = 1000
batch_size = 3

model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(x_val, y_val))

seed_data = list(test_data[:-time_seq])

for i in range(len(test_data) - time_seq):
    x_batch = np.array(seed_data[-time_seq:]).reshape([1, time_seq, input_num])

    predict = model.predict(x_batch)
    seed_data.append(predict[0, 0])

# From MinMaxScaler, inverse to real value
result = minMaxScaler.inverse_transform(np.array(seed_data[-(len(test_data) - time_seq):]).reshape([len(test_data) - time_seq, 1]))

test_data = dataset.iloc[len(train_data) + len(val_data):, :]
test_data['prediction'] = np.nan
test_data['prediction'][-(len(test_data) - time_seq):] = result.flatten()
test_data.plot()
plt.show()
