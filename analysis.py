import os
from datetime import datetime
from multiprocessing import Process

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


#################
# Preprocessing #
#################

"""
    Apple Data Informal Documentation

    Rows: Various Categorical Information

    Breaks geographical data into
        - Countries
        - Sub Regions: Such as states
        - Counties
        - Cities
"""
# Read in Apple Mobility Data
apple_data = pd.read_csv('./data/applemobilitytrends-2020-09-21.csv',
                         low_memory=False)


# Extract column names to be renamed
apple_date_columns = apple_data.loc[:, '1/13/2020':]
column_names = apple_date_columns.columns
updated_column_names = []

# Convert column names to have matching date format
for name in column_names:
    date = datetime.strptime(name, '%m/%d/%Y').strftime('%Y-%m-%d')
    updated_column_names.append(date)

# Update names and reform original DataFrame
apple_date_columns.columns = updated_column_names
apple_data = pd.concat([apple_data.loc[:, :'country'], apple_date_columns],
                       axis=1)

# Forcibly clean up duplicate date columns to preserve memory
del apple_date_columns
del column_names
del updated_column_names

# Break the data into more specific subsets.
# The data has the following structure (from broad to specific):
# Country/Region -> Sub-Region(States in the US) -> County -> City
apple_countries = apple_data.loc[apple_data['geo_type'] == 'country/region']
apple_sub_regions = apple_data.loc[apple_data['geo_type'] == 'sub-region']
apple_counties = apple_data.loc[apple_data['geo_type'] == 'county']
apple_cities = apple_data.loc[apple_data['geo_type'] == 'city']

"""
    Google Data Informal Documentation
"""

# Read in Google Mobility Data
# google_data = pd.read_csv('./data/Google_Global_Mobility_Report.csv',
#                           low_memory=False)

"""
    John Hopkins Data Informal Documentation
"""

# Read in JHU time series data
# Date format mismatches with Apple/Google datasets
# Perform preprocessing to ensure compatibility
jhu_path = './data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
jhu_data = pd.read_csv(jhu_path + 'time_series_covid19_confirmed_global.csv')

# Rename US to United States in JHU Time Series df
jhu_data.loc[jhu_data["Country/Region"] == "US", "Country/Region"] = "United States"

# Extract column names to be renamed
jhu_date_columns = jhu_data.loc[:, '1/22/20':]
column_names = jhu_date_columns.columns
updated_column_names = []

# Convert column names to have matching date format
for name in column_names:
    date = datetime.strptime(name, '%m/%d/%y').strftime('%Y-%m-%d')
    updated_column_names.append(date)

# Update names and reform original DataFrame
jhu_date_columns.columns = updated_column_names
jhu_data = pd.concat([jhu_data.loc[:, :'Long'],
                     jhu_date_columns],
                     axis=1)

# Forcibly clean up duplicate date columns to preserve memory
del jhu_date_columns
del column_names
del updated_column_names

# List of DataFrames for each country
country_df_list = []

for index, row in apple_countries.iterrows():
    country_name = row['region'].strip()

    # Flag for determining if matching country dataframe was found
    found = False

    for index, df in enumerate(country_df_list):
        if df['region'].iloc[0].strip() == country_name:
            modified_df = country_df_list[index].append(row,
                                                        ignore_index=True)
            country_df_list[index] = modified_df
            found = True

    if not found:
        country_df_list.append(row.to_frame().T)


# Converts the "direction type" index label to be a more general "datatype"
# This now indicates whether it was walking, driving, transit, or covid
# Where covid data is JHU time series data, and all other data is apple maps
# mobility statistics.
for index, df in enumerate(country_df_list):
    df.columns = ['datatype' if x == 'transportation_type'
                  else x for x in df.columns]


# Adds time series data for each country into each country's dataframe
for index, row in jhu_data.iterrows():
    country_name = row['Country/Region'].strip()
    subregion_name = str(row['Province/State']).strip()

    # This step gets each row into a labeled format that is compatible
    # with the dataframes in the country_df_list. This does not mean that the
    # element counts will be compatible. Apple/Google are missing some days and
    # JHU has more data available to it. The synchronization will
    # need to be done in an additional for loop.
    row = pd.concat([pd.Series(['country/region',
                                row[1],
                                'covid',
                                subregion_name,
                                country_name]),
                    row['2020-01-22':'2020-09-21']],
                    axis=0)

    for index, df in enumerate(country_df_list):
        if df['region'].iloc[0].strip() == country_name:
            new_index = ['geo_type',
                         'region',
                         'datatype',
                         'sub-region',
                         'country']
            new_index.extend(list(row.index.values[5:]))
            row.index = new_index
            modified_df = country_df_list[index].append(row,
                                                        ignore_index=True)
            country_df_list[index] = modified_df


for index, df in enumerate(country_df_list):
    try:
        covid_data = df.loc[df['datatype'] == 'covid'].iloc[0].tolist()[5:]
    except:
        country_df_list.pop(index)


############
# Analysis #
############

# Ensures that figures directory exists
# If figure regeneration is needed,
# it ensures that the storage directory is regenerated
figures_path = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_path):
    os.makedirs(os.path.join(os.getcwd(),'figures'))

def country_analysis(df):
    df = df.fillna(0)

    country_name = df['region'][0]

    country_path = os.path.join(figures_path, country_name)
    if not os.path.exists(country_path):
        os.makedirs(country_path)

    date_list = df.columns.values.tolist()[5:]
    covid_data = df.loc[df['datatype'] == 'covid'].iloc[0].tolist()[5:]
    driving_data = df.loc[df['datatype'] == 'driving'].iloc[0].tolist()[5:]
    walking_data = df.loc[df['datatype'] == 'walking'].iloc[0].tolist()[5:]

    walking_means = []
    driving_means = []
    labels = []
    covid = []
    for x in range(0, int(len(covid_data)/7)):
        walking_mean = 0
        driving_mean = 0
        for i in range(0, 6):
            walking_mean += walking_data[(x*7)+i]
            driving_mean += driving_data[(x*7)+i]
            if i == 4:
                covid.append(covid_data[(x*7)+i])
        walking_mean /= 7
        driving_mean /= 7
        walking_means.append(walking_mean)
        driving_means.append(driving_mean)
        labels.append((x*7)+4)


    fig, ax0 = plt.subplots()
    ax0.set_xscale('linear')
    ax0.ticklabel_format(useOffset=False, style='plain')
    ax0.scatter(labels, covid, s=walking_means)
    #ax0.scatter(date_list, covid_data, s=walking_data)
    fig.suptitle(country_name + ": Correlation of Walking Directions and Confirmed COVID Cases")
    ax0.set_xlabel("Time Passed In Days Since Jan 22nd")
    ax0.set_ylabel("Confirmed Covid Cases")
    file_name = country_name + '_covid_walking.png'
    fig.savefig(os.path.join(country_path, file_name))
    plt.clf()
    plt.close(fig)

    fig, ax0 = plt.subplots()
    ax0.set_xscale('linear')
    ax0.ticklabel_format(useOffset=False, style='plain')
    ax0.scatter(labels, covid, s=driving_means)
    #ax0.scatter(date_list, covid_data, s=driving_data)
    fig.suptitle(country_name + ": Correlation of Driving Directions and Confirmed COVID Cases")
    ax0.set_xlabel("Time Passed In Days Since Jan 22nd")
    ax0.set_ylabel("Confirmed Covid Cases")
    file_name = country_name + '_covid_driving.png'
    fig.savefig(os.path.join(country_path, file_name))
    plt.clf()
    plt.close()

# Parallelism to ensure that analysis and graph
# generation isn't prohibitively time consuming.

# processes = []
# for df in country_df_list:
#     p = Process(target=country_analysis, args=(df,))
#     p.start()
#     processes.append(p)
# for p in processes:
#     p.join()



####################
# Machine Learning #
####################

n_input_size = 100

def preprocess_ml_data(df):
    df = df.fillna(0)

    country_name = df['region'][0]
    date_list = df.columns.values.tolist()[5:]
    covid_data = df.loc[df['datatype'] == 'covid'].iloc[0].tolist()[5:]
    driving_data = df.loc[df['datatype'] == 'driving'].iloc[0].tolist()[5:]
    walking_data = df.loc[df['datatype'] == 'walking'].iloc[0].tolist()[5:]



    # Array of arrays. Each nested array has 28 days of case data
    X_train = []
    Y_train = []
    for index, datapoint in enumerate(covid_data):
        data = []

        # Ensures that bounds exception isn't raised
        if index < len(driving_data) - (n_input_size + 1):
            # Gets datapoint and 28 days following
            for i in range(0, n_input_size):
                data.append(driving_data[index+i])

        X_train.append(data)

    X_train = [x for x in X_train if len(x) == n_input_size]
    X_len = len(X_train)
    X_train = np.array(X_train)

    return X_train.reshape((X_len, n_input_size, 1)), np.array(covid_data[:X_len])

x_train, y_train = preprocess_ml_data(country_df_list[0])

#############
# RNN Model #
#############
model = Sequential()
model.add(layers.LSTM(32, input_shape=(n_input_size,1), return_sequences=True))
# model.add(layers.LSTM(32, return_sequences=True))
# model.add(layers.LSTM(32, return_sequences=True))
# model.add(layers.LSTM(32, return_sequences=True))
# model.add(layers.Activation('softmax'))
model.add(layers.Dense(1))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=2,
                                                  mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit(x_train, y_train, epochs=30,
                    validation_split=0.1,
                    callbacks=[early_stopping])

# history = model.fit(x_train, y_train,
#                     epochs=30, batch_size=16,
#                     validation_split=0.1,
#                     verbose=1, shuffle=False)
