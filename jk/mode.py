# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import gzip

# Load the dataset
df = pd.read_csv(r"C:\Users\govin\Downloads\credit.csv")

# Data preprocessing
df.drop('MINIMUM_PAYMENTS', inplace=True, axis=1)
df['CUST_ID'] = df['CUST_ID'].astype('category').cat.codes
df.drop('CREDIT_LIMIT', inplace=True, axis=1)

# Define features and target
X = df.drop(columns=['BALANCE']).values
Y = df['BALANCE'].values

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Train the RandomForestRegressor
rg = RandomForestRegressor(n_estimators=100, random_state=42)
rg.fit(X_train, Y_train)

# Evaluate the model
Y_pred = rg.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)

print("r2_score:", r2)
print("Mean Squared Error:", mse)

# Compress and save the model as a .gz file
with gzip.open('credit_model_compressed.pkl.gz', 'wb') as file:
    pickle.dump(rg, file)

print("Model compressed and saved as credit_model_compressed.pkl.gz")
