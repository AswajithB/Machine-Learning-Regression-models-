import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("car data (2).csv")

# ------------------------------
# Outlier Treatment using IQR
# ------------------------------
outlier_cols = ['Selling_Price', 'Present_Price', 'Kms_Driven']

def remove_outliers_iqr(data, column):
    q1, q2, q3 = np.percentile(data[column], [25, 50, 75])
    IQR = q3 - q1
    lower_limit = q1 - (1.5 * IQR)
    upper_limit = q3 + (1.5 * IQR)

    # Capping and flooring
    data[column] = np.where(data[column] > upper_limit, upper_limit, data[column])
    data[column] = np.where(data[column] < lower_limit, lower_limit, data[column])

for column in outlier_cols:
    remove_outliers_iqr(df, column)

# ------------------------------
# Feature Engineering
# ------------------------------
df.drop(columns=["Car_Name"], inplace=True)

current_year = datetime.now().year
df["Car_Age"] = current_year - df["Year"]
df.drop(columns=["Year"], inplace=True)

# ------------------------------
# Label Encoding and Dummies
# ------------------------------
le_fuel = LabelEncoder()
le_trans = LabelEncoder()

df["Fuel_Type"] = le_fuel.fit_transform(df["Fuel_Type"])
df["Transmission"] = le_trans.fit_transform(df["Transmission"])
df = pd.get_dummies(df, columns=["Seller_Type"], drop_first=True)

# Save encoders
with open("Fuel_Type.pkl", "wb") as f:
    pickle.dump(le_fuel, f)
with open("Transmission.pkl", "wb") as f:
    pickle.dump(le_trans, f)

# ------------------------------
# Feature Scaling
# ------------------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)  # ✅ Keep feature names

# Save scaler
with open("scaling.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

# ------------------------------
# Model Training
# ------------------------------
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model R² Score: {accuracy:.4f}")

# ------------------------------
# Save Model
# ------------------------------
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model training complete and saved successfully!")
