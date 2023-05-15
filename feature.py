import numpy as np 
import pandas as pd
import fastparquet 
import os

# import dataset
df = pd.read_parquet('df.parquet.gzip', engine = 'fastparquet')
print(df.info())

# Calculate 30 day rolling average and output to new column
df['vol_moving_avg'] = df.groupby('Symbol')['Volume'].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate 30 day rolling median and output to new column
df['adj_close_rolling_med'] = df.groupby('Symbol')['Volume'].rolling(window=30,min_periods=1).median().reset_index(level=0,drop=True)

# print sample to ensure data integrity 
print(df[['vol_moving_avg','adj_close_rolling_med']].sample(10))

# define directory to save parquet file
directory = 'features'

# Create directory is it doesn't exist
if not os.path.exists(directory):
	os.makedirs(directory)

# Save df to parquet format in directory 
df.to_parquet(os.path.join(directory, 'df_feature.parquet.gzip'))


