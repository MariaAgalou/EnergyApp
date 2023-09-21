import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import plotly.express as px
from datetime import date
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import base64
from io import BytesIO


# Function that executes the forecasting process

# steps = 48 for the next day
# steps = 336 for the next week
# steps = 1440 for the next month

def forecastingprocess(steps, flag, day=True, week=False, month=False):

    # Load dataset
    dataset = pd.read_csv("C:/Users/ΜΑΡΙΑ/Desktop/Partitioned LCL Data/Small LCL Data/LCL-June2015v2_0.csv", na_values="Null")

    # Cast features of dataset
    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
    dataset['LCLid'] = dataset['LCLid'].astype('string')
    dataset['stdorToU'] = dataset['stdorToU'].astype('string')

    # Get the current consumption for "Future Consumption"
    if flag == True:

        # Split 'DateTime' column to 2 columns 'Dates' & 'Time'
        dataset['Dates'] = pd.to_datetime(dataset['DateTime']).dt.date
        dataset['Time'] = pd.to_datetime(dataset['DateTime']).dt.time

        # Get current consumption, i.e. data of '27-02-2014'
        current_day_data = dataset.loc[dataset['Dates'] == date(2014, 2, 27)]

        # Drop unnecessary columns
        current_day_data.drop(['LCLid', 'stdorToU', 'Dates', 'Time'], axis=1, inplace=True)

        # Group by hour and calculate mean consumption of each hour
        curr = current_day_data.groupby(current_day_data['DateTime'].dt.hour)['KWH/hh (per half hour) '].mean()
        

    # Preprocessing
    data = dataset.head(100000)
    data.drop(['LCLid'], axis=1, inplace=True)                     # Drop 'LCLid' column
    data.dropna(axis=0, inplace=True)                              # Drop missing values
    data = pd.get_dummies(data, columns=['stdorToU'], dtype=int)   # One hot encoding of 'stdorToU' column

    # Scale 'KWH/hh (per half hour) ' column
    scaler = MinMaxScaler()
    data['KWH/hh (per half hour) '] = scaler.fit_transform(data['KWH/hh (per half hour) '].values.reshape(-1, 1))
    data = data['KWH/hh (per half hour) '].values

    # Sequence length = Time Steps
    sequence_length = steps

    # X at time t, Y at time t+1
    X = []
    y = []
    for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
    X = np.array(X)
    y = np.array(y)

    # Create training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # Load the appropriate model
    if day == True:
        model = keras.models.load_model('energy\daily_model.keras')         #copy relative path
    elif week == True:
        # model = keras.models.load_model('energy\weekly_model.keras')
        model = keras.models.load_model('energy\daily_model.keras')
    else:
        # model = keras.models.load_model('energy\monthly_model.keras')
        model = keras.models.load_model('energy\daily_model.keras')

    # Forecast
    y_hat = model.predict(x_test)

    
    # If flag == True, it is the "Future Consumption" case, else it is the "Records" case
    if flag == True:

        # Get list of current values
        curr = curr.values
        currlist = curr.tolist()

        y_hat = np.ravel(y_hat.tolist())
        
        # Add "y_hat" values after "currlist" values
        currlist.extend(y_hat)

        # Plot current consumption vs predicted fututre consumption
        currdf = pd.DataFrame({'value': currlist})

        fig = px.line(currdf, x = currdf.index, y = 'value')

        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='KWH')
        fig.update_layout(width=700, height=470)

        graph_json = fig.to_json()
        
        
        return y_hat, graph_json

    else:

        # Performance Metrics
        mse = mean_squared_error(y_test, y_hat)
        rmse = sqrt(mse)

        y_hat = np.ravel(y_hat.tolist())

        # Plot actual vs predicted values
        y_df = pd.DataFrame({'Y_test': y_test, 'Y_hat': y_hat})

        fig = px.line(y_df, x = y_df.index, y = ['Y_test', 'Y_hat'])

        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='KWH')
        fig.update_layout(width=700, height=470)

        fig.update_traces(name="True consumption", selector=dict(name="Y_test"))
        fig.update_traces(name="Predicted consumption", selector=dict(name="Y_hat"))

        graph_json = fig.to_json()


        #return rmse, y_test, string.decode('utf-8')
        return rmse, y_test, graph_json

