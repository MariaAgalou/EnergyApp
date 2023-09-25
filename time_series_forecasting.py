import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from keras import Model
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, LayerNormalization, MultiHeadAttention, Conv1D, Input, GlobalAveragePooling1D
from keras.optimizers import Adam, SGD


# Load dataset
#s_time_chunk = time.time()    # time taken to read data
dataset = pd.read_csv("C:/Users/ΜΑΡΙΑ/Desktop/Partitioned LCL Data/Small LCL Data/LCL-June2015v2_0.csv", na_values="Null")
#e_time_chunk = time.time()
#print(e_time_chunk-s_time_chunk)

#print(dataset.shape)
#print(dataset.columns)
#print(dataset.dtypes)
#print(dataset.isnull().sum())

# Cast
dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
dataset['LCLid'] = dataset['LCLid'].astype('string')
dataset['stdorToU'] = dataset['stdorToU'].astype('string')
print(dataset.dtypes)

# Plot dataset
dataset.plot(x = "DateTime", y = "KWH/hh (per half hour) ")
plt.show()

# EDA
data = dataset.head(100000)
data.drop(['LCLid'], axis=1, inplace=True)                     # Drop 'LCLid' column
data.dropna(axis=0, inplace=True)                              # Drop missing values
data = pd.get_dummies(data, columns=['stdorToU'], dtype=int)   # One hot encoding of 'stdorToU' column
#data.set_index('DateTime', inplace=True)                      # Set 'DateTime' column as index (only for ARIMA model)



# steps = 48 for the next day
# steps = 336 for the next week
# steps = 1440 for the next month


###########################################################################################################################################
##################################################                ARIMA MODEL            ##################################################


# Split dataset to training and test set
train_size = int(len(data) * 0.70) 
train, test = data[:train_size], data[train_size:]

# ARIMA model
model = ARIMA(endog = train['KWH/hh (per half hour) '], exog = train['stdorToU_Std'], order = (1, 1, 1))
model_fit = model.fit()

# Summary of fit model
print(model_fit.summary())

# Forecast the same number of steps as test data
forecast_steps = len(test)  
forecast = model_fit.forecast(steps = forecast_steps, exog = test['stdorToU_Std'])

# Performance Metrics
mse = mean_squared_error(test['KWH/hh (per half hour) '], forecast)
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(test['KWH/hh (per half hour) '], forecast)
print("MSE: %f, RMSE: %f, MAPE: %f" % (mse, rmse, mape))

# Plot actual vs predicted values
plt.plot(test.index, test['KWH/hh (per half hour) '], label='Actual Data')
plt.plot(test.index, forecast, label='Predicted Data', color='red')
plt.show()



###########################################################################################################################################
##################################################                LSTM MODEL            ###################################################


# Scale 'KWH' column
scaler = MinMaxScaler()
data['KWH/hh (per half hour) '] = scaler.fit_transform(data['KWH/hh (per half hour) '].values.reshape(-1, 1))
data = data['KWH/hh (per half hour) '].values

# Sequence length = Time Steps
sequence_length = 336

# X at time t, Y at time t+1
X = []
y = []

for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Create training and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(None, 1), return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=False))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Summary of fit model
#print(model.summary())

# Fit model
model.fit(x_train, y_train, epochs=10)

#model.save('weekly_model.keras')

# Forecast
y_hat = model.predict(x_test)

# Performance Metrics
mse = mean_squared_error(y_test, y_hat)
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_hat)
print("MSE: %f, RMSE: %f, MAPE: %f" % (mse, rmse, mape))

# Plot actual vs predicted values
plt.plot(y_test, label='Actual Data')
plt.plot(y_hat, label='Predicted Data', color='red')
plt.show()



###########################################################################################################################################
###################################################                GRU MODEL            ###################################################


# Scale 'KWH' column
scaler = MinMaxScaler()
data['KWH/hh (per half hour) '] = scaler.fit_transform(data['KWH/hh (per half hour) '].values.reshape(-1, 1))
data = data['KWH/hh (per half hour) '].values

# Sequence length = Time Steps
sequence_length = 1

# Time series data into sequences of fixed length
# X at time t, Y at time t+1
X = []
y = []

for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Create training and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# GRU model
model = Sequential()
model.add(GRU(100, activation='relu', input_shape=(None, 1), return_sequences=True))
model.add(GRU(100, activation='relu', return_sequences=True))
model.add(GRU(100, activation='relu', return_sequences=False))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

# Summary of fit model
#print(model.summary())

# Fit model
model.fit(x_train, y_train, epochs=10)

# Forecast
y_hat = model.predict(x_test)

# Performance Metrics
mse = mean_squared_error(y_test, y_hat)
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_hat)
print("MSE: %f, RMSE: %f, MAPE: %f" % (mse, rmse, mape))

# Plot actual vs predicted values
plt.plot(y_test, label='Actual Data')
plt.plot(y_hat, label='Predicted Data', color='red')
plt.show()



###########################################################################################################################################
#####################################################            TRANSFORMERS         #####################################################


# Split dataset to training and test set
train_size = int(len(data) * 0.70) 
train, test = data[:train_size], data[train_size:]

train_data = train['KWH/hh (per half hour) '].tolist()
test_data = test['KWH/hh (per half hour) '].tolist()


# Function that creates x and y arrays
def to_sequences(data, sequence_length):
    x = []
    y = []

    for i in range(len(data) - sequence_length):
        window = data[i:(i+sequence_length)]          # X at time t, Y at time t+1
        after_window = data[i+sequence_length]
        window = [[x] for x in window]

        x.append(window)
        y.append(after_window)

    return np.array(x), np.array(y)


# Sequence length = Time Steps
sequence_length = 1 

# Create training and test sets
x_train, y_train = to_sequences(train_data, sequence_length)
x_test, y_test = to_sequences(test_data, sequence_length)

print("Shape of training set: {}".format(x_train.shape))
print("Shape of test set: {}".format(x_test.shape))


# Function that creates the transformer encoder
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    # Normalization and Attention
    x = LayerNormalization(epsilon = 1e-6)(inputs)
    x = MultiHeadAttention(key_dim = head_size, num_heads = num_heads, dropout = dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon = 1e-6)(res)
    x = Conv1D(filters = ff_dim, kernel_size = 1, activation = "relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters = inputs.shape[-1], kernel_size=1)(x)
    return x + res


# Function that builds the transformer model
def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    outputs = Dense(1)(x)

    return Model(inputs, outputs)



input_shape = x_train.shape[1:]

# Transformer Model
model = build_model(input_shape, head_size = 256, num_heads = 4, ff_dim = 4, num_transformer_blocks = 5, mlp_units =[128], 
                    dropout=0.1, mlp_dropout=0.1)

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Fit model
model.fit(x_train, y_train, epochs=10)

# Forecast
y_hat = model.predict(x_test)

# Performance Metrics
mse = mean_squared_error(y_test, y_hat)
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_hat)
print("MSE: %f, RMSE: %f, MAPE: %f" % (mse, rmse, mape))

# Plot actual vs predicted values
plt.plot(y_test, label='Actual Data')
plt.plot(y_hat, label='Predicted Data', color='red')
plt.show()



