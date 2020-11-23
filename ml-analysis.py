# Influenced heavily by
# https://www.tensorflow.org/tutorials/structured_data/time_series#normalize_the_data

import os
import copy
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
google_data = pd.read_csv('./data/Google_Global_Mobility_Report.csv',
                          low_memory=False).fillna(0)



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


# Extract Google country data into dataframe
for country in set(google_data['country_region'].to_list()):

    # Second line ensures that no duplicate city data from countries is picked up
    country_data = google_data.loc[google_data['country_region'] == country]
    country_data = country_data.loc[country_data['sub_region_1'] == 0]

    # Seperates description information from mobility data
    # temp: stores mobility data
    # country_data: stores description information
    temp = country_data.transpose().iloc[7:]
    country_data = country_data.transpose().iloc[:7]
    country_data = country_data.iloc[:,:6]
    country_data = country_data.transpose()

    # creates a single column dataframe
    # will be used to label dataframe within country dataframe
    datatypes = temp.index.values.tolist()
    datatypes = pd.DataFrame(datatypes, columns=['datatype'])

    # renames column index in temp to use date format
    # renames row indices to be numeric
    # this makes concatenation work later
    temp.columns = temp.iloc[0]
    temp = temp.drop(temp.index[0])
    temp.index = list(range(6))
    datatypes = datatypes.drop(datatypes.index[0])
    datatypes.index = list(range(6))

    # creates country dataframe with all information
    # additional logic is needed to match overall column index format
    google_country_df = pd.concat([country_data, datatypes, temp], axis=1)
    google_country_df.rename(columns={'country_region_code':'geo_type',
                                      'country_region':'region',
                                      'sub_region_1':'sub-region',
                                      'sub_region_2':'country'}, inplace=True)

    # reorder columns to match country_df
    cols = list(google_country_df.columns.values)
    cols_reorder = ['geo_type',
                    'region',
                    'datatype',
                    'sub-region',
                    'country']
    cols = cols_reorder + cols[8:]

    google_country_df = google_country_df[cols].iloc[0:6]

    # Fill NaN with 0
    google_country_df = google_country_df.fillna(0)


    df = google_country_df.iloc[:, 5:]

    df = pd.concat([google_country_df.iloc[:, 0:6],
                                   df.groupby(df.columns, axis=1).mean()],
                                  axis=1)

    df = df.loc[:,~df.columns.duplicated()]

    # Normalize data to match apple dataset
    numeric_cols = [col for col in df if df[col].dtype.kind != 'O']
    df[numeric_cols] += 100
    df['geo_type'] = "country/region"
    df['region'] = country
    google_country_df = df

    # find matching country in country_df_list
    # append google data to matching dataframe
    for index, country_df in enumerate(country_df_list):
        if country_df['region'].iloc[0].strip() == country.strip():
            df = pd.concat([country_df, google_country_df], axis=0)
            country_df_list[index] = df.fillna(0)


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
    train_df = df[0:int(n*0.8)]
    val_df = df[int(n*0.5):int(n*0.7)]
    test_df = df[int(n*0.7):]

    # Perform Data Normalization
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std


    num_features = df.shape[1]

    return WindowGenerator(input_width = 18,
                           label_width = 1,
                           shift = 7,
                           train_df=train_df,
                           val_df=val_df,
                           test_df=test_df)


def compile_and_fit(model, window, patience=2, MAX_EPOCHS=30):

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        verbose=0)
    return history

# Ensures that figures directory exists
# If figure regeneration is needed,
# it ensures that the storage directory is regenerated
results_path = os.path.join(os.getcwd(), 'results')
if not os.path.exists(results_path):
    os.makedirs(os.path.join(os.getcwd(),'results'))

for df in country_df_list:

    # Store country name for labeling
    country_name = df['region'][0]

    print("Training on country: " + country_name)

    # Ensures that NaN are set to 0
    df = df.fillna(0)

    w = preprocess_data(df.transpose().fillna(0)[5:])

    rnn_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.SimpleRNN(128),
        #tf.keras.layers.GRU(32, return_sequences=True),
        #tf.keras.layers.GRU(32, return_sequences=True),
        #tf.keras.layers.Dense(units=1000),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    compile_and_fit(rnn_model, w, MAX_EPOCHS=500)
    rnn_val_performance = str(rnn_model.evaluate(w.train, verbose=0))
    rnn_performance = str(rnn_model.evaluate(w.test, verbose=0))

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        #tf.keras.layers.GRU(32, return_sequences=True),
        #tf.keras.layers.GRU(32, return_sequences=True),
        #tf.keras.layers.Dense(units=1000),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    compile_and_fit(lstm_model, w, MAX_EPOCHS=500)
    lstm_val_performance = str(lstm_model.evaluate(w.train, verbose=0))
    lstm_performance = str(lstm_model.evaluate(w.test, verbose=0))

    gru_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.GRU(32, return_sequences=True),
        #tf.keras.layers.GRU(32, return_sequences=True),
        #tf.keras.layers.GRU(32, return_sequences=True),
        #tf.keras.layers.Dense(units=1000),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    compile_and_fit(gru_model, w, MAX_EPOCHS=500)
    gru_val_performance = str(gru_model.evaluate(w.train, verbose=0))
    gru_performance = str(gru_model.evaluate(w.test, verbose=0))

    # Ensures that there is a path for figures to be stored (per country)
    country_path = os.path.join(results_path, country_name)
    if not os.path.exists(country_path):
        os.makedirs(country_path)

    model_performance_file = open(country_path + "/" + country_name + "_model_performance.txt", "w+")

    model_performance.write("RNN_MODEL:\n")
    model_performance_file.write("Val Performance: " + rnn_val_performance + "\n" )
    model_performance_file.write("Performance: " + rnn_performance + "\n\n\n")

    model_performance.write("LSTM_MODEL:\n")
    model_performance_file.write("Val Performance: " + lstm_val_performance + "\n" )
    model_performance_file.write("Performance: " + lstm_performance + "\n\n\n")

    model_performance.write("GRU_MODEL:\n")
    model_performance_file.write("Val Performance: " + gru_val_performance + "\n" )
    model_performance_file.write("Performance: " + gru_performance + "\n\n\n")

    model_performance_file.close()

