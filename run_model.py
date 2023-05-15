# import libraries 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import joblib


# import dataset from 'Features folder'
path = 'features/df_feature.parquet.gzip'
df = pd.read_parquet(path)
print(df.sample(10))

# We have many col's in the dataset, and I want to find the best predictor of the target variable.
# Since Open, Close, High, Low..ect are all in $ and Volume variable are in number of stocks, we need to scale the data in order to compare and find a corelation coefficient.
# we will be using a standard scalar Source: https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/ 
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df[['Volume','Open','Close','High','Low','Adj Close','adj_close_rolling_med','vol_moving_avg']])
# df_scaled = pd.DataFrame(df_scaled)

# Calculate correlation matrix
# corr_matrix = df_scaled.corr(method='pearson')
# print(corr_matrix)
# coeff_vol = corr_matrix[]
# print(coeff_vol)

# Looking at the correlations, we see that the rolling median and rolling average are 0.855 and 0.859 
# respectively. All other values statistically insignificant with regards to volume. 
# Therefore, we will only be using rolling median and rolling average in our predictive model.

# Now that we have the data features we want, lets look at how much data we have per stock.
counts = df["Symbol"].value_counts()
# print(counts)

# Since some stocks have limited data, we need to ensure that each stock
# has more than 30 days of data as we are using that 30 day rolling mean and median
symbols_to_keep = counts[counts > 30].index.tolist()
df_filtered = df[df['Symbol'].isin(symbols_to_keep)]

print(df_filtered["Symbol"].value_counts()) 
print(df_filtered.info())

# Sample half of the dataset due to memory constraints on local machine
df_sample = df_filtered.sample(frac=0.5,random_state=7)

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# define training and testing 
target = df_sample['Volume']
features = df_sample[['adj_close_rolling_med','vol_moving_avg']]

# split training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.2, random_state=7)

# Define RF model
rf = RandomForestRegressor(random_state=7, verbose=1)

# Fit the model with training data
rf.fit(features_train, target_train)

# Predict on test data
rf_pred = rf.predict(features_test)

# Calculate evaluation metrics
r2 = r2_score(target_test, rf_pred)
mse = mean_squared_error(target_test, rf_pred)
mae = mean_absolute_error(target_test, rf_pred)
rmse = np.sqrt(mse)

# Print evaluation metrics
print("R2 Score:", r2)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# Save training metrics as log files
logging.basicConfig(filename='training_metrics.log', level=logging.INFO)
logging.info(f"R2 Score: {r2}")
logging.info(f"Mean Squared Error: {mse}")
logging.info(f"Mean Absolute Error: {mae}")
logging.info(f"Root Mean Squared Error: {rmse}")

# Save model to disk
joblib.dump(rf, 'rf_model.joblib')
