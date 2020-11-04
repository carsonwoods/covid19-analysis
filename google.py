import os
from datetime import datetime
from multiprocessing import Process

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Google Data Informal Documentation
"""

# Read in Google Mobility Data
google_data = pd.read_csv('./data/Google_Global_Mobility_Report.csv',
                          low_memory=False)

google_data = google_data.fillna(0)


# Extensive Preprocessing Required

# Get a single country (this step will be done by loop eventually)
# Second line ensures that no duplicate city data from countries is picked up
country_data = google_data.loc[google_data['country_region_code'] == 'AE']
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
country_df = pd.concat([country_data, datatypes, temp], axis=1)