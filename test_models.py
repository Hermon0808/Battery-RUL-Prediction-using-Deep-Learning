import joblib
import pandas as pd
import torch
from train_dl_model import RULModel
import numpy as np

# === ML Model Prediction ===
df = pd.read_csv("processed_stats.csv")
features = [
    'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
    'Charge_Energy(Wh)', 'Discharge_Energy(Wh)', 'Internal_Resistance(Ohm)',
    'AC_Impedance(Ohm)', 'ACI_Phase_Angle(Deg)', 'Charge_Time(s)',
    'DisCharge_Time(s)', 'Vmax_On_Cycle(V)', 'Charge_Efficiency', 'Energy_Efficiency'
]
X = df[features].fillna(0)
model = joblib.load("rul_ml_model.pkl")
pred_ml = model.predict(X.iloc[[-1]])
print(f"ML Model RUL Prediction: {pred_ml[0]:.2f} cycles")

# === DL Model Prediction ===
model_dl = RULModel()
model_dl.load_state_dict(torch.load("rul_dl_model.pth"))
model_dl.eval()

df_seq = pd.read_pickle("sequence_data.pkl")
last_seq = torch.tensor(np.expand_dims(df_seq['Voltage_Seq'].iloc[-1], axis=(0, -1)), dtype=torch.float32)
pred_dl = model_dl(last_seq).item()
print(f"DL Model RUL Prediction: {pred_dl:.2f} cycles")
