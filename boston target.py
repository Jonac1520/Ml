# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load Dataset
from sklearn.datasets import load_boston
boston = load_boston()

# Convert to pandas DataFrame
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target  # Target variable

# Features and target
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 5: Save the Model
joblib.dump(model, "linear_regression_model.pkl")
print("Model saved as 'linear_regression_model.pkl'")

# Step 6: Load the Model
loaded_model = joblib.load("linear_regression_model.pkl")
print("Loaded model from file.")

# Step 7: Make Predictions with the Loaded Model
new_data = X_test.iloc[:5]  # Select first 5 samples from test data
predictions = loaded_model.predict(new_data)
print("Predictions for new data:")
print(predictions)