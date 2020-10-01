from datetime import datetime
import pandas as pd


#################
# Preprocessing #
#################

"""
    Apple Data Informal Documentation

    Rows: Various Categorical Information, followed by dates since Jan 13th, 2020

    Breaks geographical data into
        - Countries
        - Sub Regions: Such as states
        - Counties
        - Cities
"""
# Read in Apple Mobility Data
apple_data = pd.read_csv('./data/applemobilitytrends-2020-09-21.csv', low_memory=False)


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
apple_data = pd.concat([apple_data.loc[:,:'country'], apple_date_columns], axis=1)

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
# google_data = pd.read_csv('./data/Google_Global_Mobility_Report.csv', low_memory=False)



"""
    John Hopkins Data Informal Documentation
"""

# Read in JHU time series data
# Date format mismatches with Apple/Google datasets
# Perform preprocessing to ensure compatibility
jhu_data = pd.read_csv('./data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

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
jhu_data = pd.concat([jhu_data.loc[:,:'Long'], jhu_date_columns], axis=1)

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
            country_df_list[index] = country_df_list[index].append(row, ignore_index=True)
            found = True

    if not found:
        country_df_list.append(row.to_frame().T)


# Converts the "direction type" index label to be a more general "datatype"
# This now indicates whether it was walking, driving, transit, or covid
# Where covid data is JHU time series data, and all other data is apple maps
# mobility statistics. 
for index, df in enumerate(country_df_list):
    df.columns = ['datatype' if x=='transportation_type' else x for x in df.columns]


# Adds time series data for each country into each country's dataframe 
for index, row in jhu_data.iterrows():
    country_name = row['Country/Region'].strip()
    subregion_name = str(row['Province/State']).strip()
    
    # This step gets each row into a labeled format that is compatible
    # with the dataframes in the country_df_list. This does not mean that the 
    # element counts will be compatible. Apple/Google are missing some days and 
    # JHU has more data available to it. The synchronization will need to be done 
    # in an additional for loop.
    row = pd.concat([pd.Series(['country/region', 
                                row[1],
                                'covid',
                                subregion_name,
                                country_name]), row['2020-01-22':'2020-09-21']], axis=0)
    
    for index, df in enumerate(country_df_list):
        if df['region'].iloc[0].strip() == country_name:
            country_df_list[index] = country_df_list[index].append(row, ignore_index=True)
    



####################
# Machine Learning #
####################
