import pandas as pd
import numpy as np
import os

file_path = os.path.join(os.getcwd(), 'battery_data.xlsx')
xls = pd.ExcelFile(file_path)

# Load sheets
df_raw = xls.parse("raw")
df_stats = xls.parse("stats")
df_stats.columns = df_stats.columns.str.strip()

# Add RUL to stats sheet
max_cycle = df_stats['c'].max()
df_stats['RUL'] = max_cycle - df_stats['c']

# Add engineered features
df_stats['Charge_Efficiency'] = df_stats['Discharge_Capacity(Ah)'] / df_stats['Charge_Capacity(Ah)']
df_stats['Energy_Efficiency'] = df_stats['Discharge_Energy(Wh)'] / df_stats['Charge_Energy(Wh)']
df_stats = df_stats.replace([np.inf, -np.inf], np.nan).fillna(0)

# Save processed stats for ML
df_stats.to_csv("processed_stats.csv", index=False)

# For deep learning: save time-series of Voltage over steps per cycle
grouped = df_raw.groupby('Cycle_Index')
sequences = []

for cycle, group in grouped:
    seq = group.sort_values('Step_Index')['Voltage(V)'].values
    if len(seq) >= 50:  # Pad or truncate to fixed length
        seq = seq[:50]
    else:
        seq = np.pad(seq, (0, 50 - len(seq)), mode='constant')
    sequences.append((cycle, seq))

df_seq = pd.DataFrame(sequences, columns=['Cycle_Index', 'Voltage_Seq'])
df_seq['Voltage_Seq'] = df_seq['Voltage_Seq'].apply(lambda x: np.array(x))
df_seq = df_seq.merge(df_stats[['c', 'RUL']], left_on='Cycle_Index', right_on='c')
df_seq[['Voltage_Seq', 'RUL']].to_pickle("sequence_data.pkl")
