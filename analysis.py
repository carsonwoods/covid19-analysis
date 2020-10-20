import os
from datetime import datetime
from multiprocessing import Process

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


############
# Analysis #
############

# Ensures that figures directory exists
# If figure regeneration is needed,
# it ensures that the storage directory is regenerated
figures_path = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_path):
    os.makedirs(os.path.join(os.getcwd(),'figures'))

# Placed in a function for ease of multiprocessing
def country_analysis(df):
    # Ensures that NaN are set to 0
    df = df.fillna(0)

    # Store country name for labeling
    country_name = df['region'][0]

    # Ensures that there is a path for figures to be stored (per country)
    country_path = os.path.join(figures_path, country_name)
    if not os.path.exists(country_path):
        os.makedirs(country_path)

    # Converts df rows to lists for easier operations
    date_list = df.columns.values.tolist()[5:]
    covid_data = df.loc[df['datatype'] == 'covid'].iloc[0].tolist()[5:]
    driving_data = df.loc[df['datatype'] == 'driving'].iloc[0].tolist()[5:]
    walking_data = df.loc[df['datatype'] == 'walking'].iloc[0].tolist()[5:]

    # The data was initially too messy to interpret, and without
    # normalization it was useless. This takes average values over 7 day
    # intervals to make the data significantly more readable
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

        # Labels are now number of days since start of data
        labels.append((x*7)+4)

    # Draw Plots for each country's respective walking and driving data
    # Plots are scatter plots with, the size of each data point on the graph
    # corresponding to the amount of directions requested that day
    # (larger dots == more directions, smaller == less)
    fig, ax0 = plt.subplots()
    ax0.set_xscale('linear')
    ax0.ticklabel_format(useOffset=False, style='plain')
    ax0.scatter(labels, covid, s=walking_means)
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
processes = []
for df in country_df_list:
    p = Process(target=country_analysis, args=(df,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
