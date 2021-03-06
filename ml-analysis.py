# Influenced heavily by
# https://www.tensorflow.org/tutorials/structured_data/time_series#normalize_the_data

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
from tensorflow.keras.layers import Dense, LSTM, GRU, TimeDistributed


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
jhu_path = './data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/'
jhu_data = pd.read_csv(jhu_path + 'time_series_covid19_confirmed_global.csv')

# Rename US to United States in JHU Time Series df
jhu_data.loc[jhu_data["Country/Region"] == "US", "Country/Region"] = "United States"

# Extract column names to be renamed
# Date format does not match other data, so do conversion
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






"""
    Mobility data and COVID data is desired on a per country basis.
    This will create a a list of dataframes where each dataframe
    holds the data from Apple, Google, and JHU for a single country.
    Country's missing one or more datasources are not included.
"""

# List of DataFrames for each country
country_df_list = []

# Gets all countries in Apple's dataset
for index, row in apple_countries.iterrows():
    country_name = row['region'].strip()

    # Flag for determining if matching country dataframe was found
    found = False

    # Iterates through list of country dataframes
    for index, df in enumerate(country_df_list):
        # Checks to determine if country is already present
        if df['region'].iloc[0].strip() == country_name:
            modified_df = country_df_list[index].append(row,
                                                        ignore_index=True)
            country_df_list[index] = modified_df
            found = True

    # Ensures that countries that were not already found are added
    if not found:
        country_df_list.append(row.to_frame().T)


# Converts the "direction type" index label to be a more general "datatype"
# This now indicates whether it was walking, driving, transit, or covid
# Where covid data is JHU time series data, and all other data is apple maps
# mobility statistics.
for index, df in enumerate(country_df_list):
    df.columns = ['datatype' if x == 'transportation_type'
                  else x for x in df.columns]


# Adds JHU data for each country into each country's dataframe
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

    # Searches for matching country dataframe
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

# Filter out countries that are lacking covid data
for index, df in enumerate(country_df_list):
    try:
        covid_data = df.loc[df['datatype'] == 'covid'].iloc[0].tolist()[5:]
    except:
        country_df_list.pop(index)



####################
# Machine Learning #
####################


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):

        # Training, validation, and testing dataframes
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
               enumerate(train_df.columns)}

        # Parameters for window
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
          labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=1,
              shuffle=True,
              batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


def preprocess_data(df):
    """
    Generates Windows from data
    """
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    # Perform Data Normalization
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std


    num_features = df.shape[1]

    return WindowGenerator(input_width = 28,
                           label_width = 1,
                           shift = 28,
                           train_df=train_df,
                           val_df=val_df,
                           test_df=test_df)


def compile_and_fit(model, window, patience=2, MAX_EPOCHS=30):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


df = country_df_list[0].transpose().fillna(0)[5:]
print(df)
w = preprocess_data(df)

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

gru_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.Dense(units=1000),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

#compile_and_fit(lstm_model, w)
compile_and_fit(gru_model, w)

