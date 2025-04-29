import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# === Load the Excel file ===
file_path = os.path.join(os.getcwd(), 'battery_data.xlsx')
xls = pd.ExcelFile(file_path)

# === Read the two sheets ===
df_raw = xls.parse("raw")
df_stats = xls.parse("stats")

# === Clean column names (remove whitespace) ===
df_stats.columns = df_stats.columns.str.strip()

# === Create the RUL target ===
max_cycle = df_stats['c'].max()  # 'c' is assumed to be Cycle_Index
df_stats['RUL'] = max_cycle - df_stats['c']

# === Select useful features ===
features = [
    'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
    'Charge_Energy(Wh)', 'Discharge_Energy(Wh)', 'Internal_Resistance(Ohm)',
    'AC_Impedance(Ohm)', 'ACI_Phase_Angle(Deg)', 'Charge_Time(s)',
    'DisCharge_Time(s)', 'Vmax_On_Cycle(V)'
]

# === Check for missing columns ===
for col in features:
    if col not in df_stats.columns:
        raise ValueError(f"Missing expected column in stats sheet: {col}")

X = df_stats[features].fillna(0)
y = df_stats['RUL']

# === Split the data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train the model ===
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# === Evaluate the model ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Evaluation:")
print(f"MAE (Mean Absolute Error): {mae:.2f} cycles")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} cycles")

# === Predict RUL for the latest entry ===
latest_input = X.iloc[[-1]]
predicted_rul = model.predict(latest_input)[0]
print(f"\nPredicted RUL for the most recent cycle: {predicted_rul:.2f} cycles")
