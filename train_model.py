import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load real house price dataset
df = pd.read_csv("kc_house_data.csv")

# Select required columns
df = df[["sqft_living", "bedrooms", "bathrooms", "price"]]

# Rename sqft_living to area
df = df.rename(columns={
    "sqft_living": "area"
})

# Remove missing values
df = df.dropna()

# Features and target
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate error
mae = mean_absolute_error(y_test, predictions)

print("Real dataset loaded successfully!")
print("Number of records:", len(df))
print("Mean Absolute Error:", mae)

# Save trained model
joblib.dump(model, "house_price_model.pkl")

print("Model saved as house_price_model.pkl")
import pandas as pd

data = {
    "sqft_living": [500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500],
    "bedrooms": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    "bathrooms": [1, 1, 2, 2, 2, 3, 3, 4, 4, 5],
    "price": [150000, 220000, 280000, 350000, 430000,
              520000, 600000, 720000, 850000, 1000000]
}

df = pd.DataFrame(data)

df.to_csv("kc_house_data.csv", index=False)

print("Dataset created successfully!")
print("Number of records:", len(df))