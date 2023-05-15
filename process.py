# Import Libraries 
import pandas as pd 
import numpy as np
import os
import glob 
import sys
import fastparquet

# Access all CSV Files in ETF folder and Stocks Folder
etf_path = "/Users/Taylor/Desktop/Code/Python/riskthink_analysis/venv/etfs"
stock_path = "/Users/Taylor/Desktop/Code/Python/riskthink_analysis/venv/stocks"

# create object for csv files
csv_etf = glob.glob(etf_path+"/*.csv")
csv_stocks = glob.glob(stock_path+"/*.csv")

# create for loop that reads all csv's in folder and creates dataframe of data 
data = []

for csv in csv_etf and csv_stocks:
    frame = pd.read_csv(csv)
    frame['filename'] = os.path.basename(csv)
    data.append(frame)

df = pd.concat(data, ignore_index = True)

# Change "filename" to "Symbol", re-order, and get rid of .csv string at end 
df.rename(columns={'filename':'Symbol'}, inplace = True)
df = df.reindex(columns = ['Symbol','Security Name','Date','Open','High','Low','Close','Adj Close','Volume'])
df['Symbol'] = df['Symbol'].str.replace('.csv','')

# create df for security names / metadata 
path = "/Users/Taylor/Desktop/Code/Python/riskthink_analysis/venv/"
names = pd.read_csv('symbols_valid_meta.csv')

df['Security Name'] = df['Symbol'].map(names.set_index('Symbol')['Security Name'])

# Convert Symbol, Date, Name to strings and Volume to Int
df[['Symbol','Security Name','Date']] = df[['Symbol','Security Name','Date']].astype('string')

# Look into all columns, look into na values and remove
print(df.isna().sum().sum()) #3894 values
df = df.dropna()
print(df.isna().sum().sum()) # check

# Convert Volume from float to int
df['Volume'] = df['Volume'].astype(int)
print(df['Volume'].dtype)

print(df.dtypes) # Check types 

#check out sample rows in df
print(df.sample(20))

# Convert dataset into Parquet 
df.to_parquet('df.parquet.gzip')


