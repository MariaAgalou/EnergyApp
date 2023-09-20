import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, mode
from datetime import date
import datetime
import base64
from io import BytesIO


# Load dataset
dataset = pd.read_csv("C:/Users/ΜΑΡΙΑ/Desktop/Partitioned LCL Data/Small LCL Data/LCL-June2015v2_0.csv", na_values="Null")

# Cast features of dataset
dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
dataset['LCLid'] = dataset['LCLid'].astype('string')
dataset['stdorToU'] = dataset['stdorToU'].astype('string')

# Drop missing values
dataset.dropna(axis=0, inplace=True)

dataset = dataset.head(400000)


# Check if an ID exists in the dataset
def id_exists(id, dataset = dataset):

    for rec in dataset.index:
        # If it exists, return True
        if dataset['LCLid'][rec] == id:
            return True
    
    # Else, return False
    return False



# Consumption of past day / past 24 hours
def last_day_consumption(dataset = dataset):

    # Split 'DateTime' column to 2 columns 'Dates' & 'Time'
    dataset['Dates'] = pd.to_datetime(dataset['DateTime']).dt.date
    dataset['Time'] = pd.to_datetime(dataset['DateTime']).dt.time

    # Get data of '26-02-2014' (which i have defined as the last day or last 24 hours)
    data = dataset.loc[dataset['Dates'] == date(2014, 2, 26)]

    # Get statistics
    min_value = data['KWH/hh (per half hour) '].min()
    max_value = data['KWH/hh (per half hour) '].max()
    mean_value = data['KWH/hh (per half hour) '].mean()

    id_min = mode(data[data['KWH/hh (per half hour) '] == min_value]['LCLid'])
    id_max = (data[data['KWH/hh (per half hour) '] == max_value]['LCLid'].values)[0]
    id_mean = mode(data['LCLid'])

    time_min = mode(data[data['KWH/hh (per half hour) '] == min_value]['Time'])
    time_max = (data[data['KWH/hh (per half hour) '] == max_value]['Time'].values)[0]
    time_mean = mode(data['Time'])

    # Save statistics to list
    l = []
    l.extend((min_value, mean_value, max_value, id_min, id_mean, id_max, time_min, time_mean, time_max))


    # Drop unnecessary columns
    data.drop(['LCLid', 'stdorToU', 'Dates', 'Time'], axis=1, inplace=True)

    # Group by hour and calculate mean consumption of each hour
    mean_data = data.groupby(data['DateTime'].dt.hour)['KWH/hh (per half hour) '].mean()

    # Create dataframe with 24 hours
    time_intervals = [datetime.time(hour=i).strftime('%H:%M:%S') for i in range(24)]
    df = pd.DataFrame({'time_intervals': time_intervals})

    # Join consumption with hours
    dt = df.join(mean_data)

    # Plot data
    fig = dt.plot(x = 'time_intervals', y = 'KWH/hh (per half hour) ')
    plt.xlabel("Hour") 
    plt.ylabel("KWH")

    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    string = base64.b64encode(buf.read())
    

    return string.decode('utf-8'), l



# Statistics about all the customers
def customers_statistics(dataset = dataset):

    # Split 'DateTime' column to 2 columns 'Dates' & 'Time'
    dataset['Dates'] = pd.to_datetime(dataset['DateTime']).dt.date
    dataset['Time'] = pd.to_datetime(dataset['DateTime']).dt.time

    # Get data of last month (i defined February 2014 as the last month)
    dataset = dataset.loc[dataset['Dates'] >= date(2014, 2, 1)]

    # Get unique values / IDs of customers
    data = dataset['LCLid'].unique()
    x = len(data)    # number of unique values / customers

    # Array containing the current consumption
    curr = []

    # Get data of current day (i defined '27 - 02 - 2014' as the current day)
    current_day_data = dataset.loc[dataset['Dates'] == date(2014, 2, 27)]

    # Get data of current hour (i defined '23:00:00' as the current hour)
    current_hour_data = current_day_data[current_day_data['Time'] == pd.to_datetime('23:00:00').time()]

    # Get current consumption
    curr = current_hour_data['KWH/hh (per half hour) '].values


    # Array with all consumptions for each customer (list of list)
    cons = []
    for i in range(x):
        cons.append([])

    # Calculate all consumptions of each customer
    z = 0
    for k in data:
        for y in dataset.index:
            if dataset['LCLid'][y] == k:
                cons[z].append(dataset['KWH/hh (per half hour) '][y])
        z += 1


    # Dataframe with ID 
    df = pd.DataFrame(data)
    df.columns = ['ID']

    # Arrays
    mins, means, maxs = [], [], []
    mins_time, maxs_time = [], []

    # Calculate min, mean, max and current consumption for each customer
    for i in range(x):
        mins.append(min(cons[i]))
        means.append(round(mean(cons[i]), 3))
        maxs.append(max(cons[i]))


    # Create columns on dataframe and save values
    df['Min_Consumption'] = mins
    df['Mean_Consumption'] = means
    df['Max_Consumption'] = maxs
    df['Current_Consumption'] = curr


    # Get time of min and max consumption for each customer
    for i in df.index:
        for j in dataset.index:
            if dataset['LCLid'][j] == df['ID'][i]: 
                if dataset['KWH/hh (per half hour) '][j] == df['Min_Consumption'][i]:
                    mins_time.append(dataset['DateTime'][j])
                    break
                

    for i in df.index:
        for j in dataset.index:
            if dataset['LCLid'][j] == df['ID'][i]: 
                if dataset['KWH/hh (per half hour) '][j] == df['Max_Consumption'][i]:
                    maxs_time.append(dataset['DateTime'][j])
                    break

    # Create columns on dataframe and save values
    df['Min_Consumption_Time'] = mins_time
    df['Max_Consumption_Time'] = maxs_time

    df['Min_Consumption_Time'] = df['Min_Consumption_Time'].dt.strftime('%H:%M:%S')
    df['Max_Consumption_Time'] = df['Max_Consumption_Time'].dt.strftime('%H:%M:%S')
    
    
    # DATAFRAME = ( ID - Min Cons. - Mean Cons. - Max Cons. - Current Cons. - Time of Min Cons. - Time of Max Cons.)
    return df



# Statistics about one single customer
def customer_statistics(id, dataset = dataset):

    df = customers_statistics(dataset)
    customer_data = df.loc[df['ID'] == id]

    # Statistics
    min = customer_data['Min_Consumption'].values
    max = customer_data['Max_Consumption'].values
    avg = customer_data['Mean_Consumption'].values
    lowtime = customer_data['Min_Consumption_Time'].values
    hightime = customer_data['Max_Consumption_Time'].values


    # Split 'DateTime' column to 2 columns 'Dates' & 'Time'
    dataset['Dates'] = pd.to_datetime(dataset['DateTime']).dt.date
    dataset['Time'] = pd.to_datetime(dataset['DateTime']).dt.time

    # Get data of last month (i defined February 2014 as the last month)
    data = dataset.loc[dataset['Dates'] >= date(2014, 2, 1)]

    # Get data of the customer that matches the given ID
    data = data.loc[data['LCLid'] == id]

    # Group data by day and calculate sum of 'KWH' for each day
    grouped_data = data.groupby('Dates')['KWH/hh (per half hour) '].sum()
    print("GROUPED DATA", grouped_data.to_string())

    # Graphs
    fig1 = grouped_data.plot(kind='line')           # Linechart
    plt.title('Linechart of Daily KWH Usage')
    plt.xlabel("Date") 
    plt.ylabel("KWH")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    string1 = base64.b64encode(buf.read())

    fig2 = plt.boxplot(grouped_data)              # Boxplot
    plt.title('Boxplot of Daily kWh Usage')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    string2 = base64.b64encode(buf.read())

    

    return string1.decode('utf-8'), string2.decode('utf-8'), min[0], max[0], avg[0], lowtime[0], hightime[0]
    


