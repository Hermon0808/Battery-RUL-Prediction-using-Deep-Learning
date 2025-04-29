import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv("processed_stats.csv")

features = [
    'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
    'Charge_Energy(Wh)', 'Discharge_Energy(Wh)', 'Internal_Resistance(Ohm)',
    'AC_Impedance(Ohm)', 'ACI_Phase_Angle(Deg)', 'Charge_Time(s)',
    'DisCharge_Time(s)', 'Vmax_On_Cycle(V)', 'Charge_Efficiency', 'Energy_Efficiency'
]

X = df[features].fillna(0)
y = df['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save model
import joblib
joblib.dump(model, "rul_ml_model.pkl")

y_pred = model.predict(X_test)
print("ML Model MAE:", mean_absolute_error(y_test, y_pred))
print("ML Model RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
