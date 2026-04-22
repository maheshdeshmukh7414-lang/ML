import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("data.csv")

# Display first few rows
print("Dataset:\n", data.head())

# Features and target
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Output results
print("\nPredictions:", predictions)
print("\nActual:", list(y_test))

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error:", mae)
