import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the trained ML model
model = joblib.load("rul_ml_model.pkl")

# Create the main window
root = tk.Tk()
root.title("EV Battery RUL Prediction")

# Set the window size and background color
root.geometry("500x650")
root.config(bg="#f0f0f0")

# Add a title label
title_label = tk.Label(root, text="EV Battery Remaining Useful Life (RUL) Prediction", 
                       font=("Helvetica", 16), bg="#f0f0f0")
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# Create input labels and fields with data type hints (int or float)
labels = [
    ('Current (A)', 'float'), ('Voltage (V)', 'float'), ('Charge Capacity (Ah)', 'float'),
    ('Discharge Capacity (Ah)', 'float'), ('Charge Energy (Wh)', 'float'), 
    ('Discharge Energy (Wh)', 'float'), ('Internal Resistance (Ohm)', 'float'), 
    ('AC Impedance (Ohm)', 'float'), ('ACI Phase Angle (Deg)', 'float'), 
    ('Charge Time (s)', 'int'), ('Discharge Time (s)', 'int'), ('Vmax On Cycle (V)', 'float'), 
    ('Charge Efficiency', 'float'), ('Energy Efficiency', 'float')
]

inputs = {}

# Loop to create entry fields dynamically for each feature
for i, (label, data_type) in enumerate(labels):
    # Add data type hint in the label
    label_text = f"{label} ({data_type})"
    tk.Label(root, text=label_text, font=("Helvetica", 10), bg="#f0f0f0").grid(row=i+1, column=0, pady=8, padx=10, sticky="w")
    entry = tk.Entry(root, font=("Helvetica", 10), width=25)
    entry.grid(row=i+1, column=1, pady=8, padx=10)
    inputs[label] = entry

# Function to handle prediction
def predict_rul():
    try:
        # Gather inputs and convert to float
        input_values = [float(inputs[label].get()) if data_type == 'float' else int(inputs[label].get()) 
                        for (label, data_type) in labels]

        # Convert inputs to the shape the model expects
        X_user = np.array(input_values).reshape(1, -1)

        # Predict RUL
        predicted_rul = model.predict(X_user)[0]

        # Display the result in a message box
        messagebox.showinfo("Prediction", f"Predicted Remaining Useful Life (RUL): {predicted_rul:.2f} cycles")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# Add the prediction button with styling
predict_button = tk.Button(root, text="Predict RUL", command=predict_rul, font=("Helvetica", 12), 
                           bg="#4CAF50", fg="white", relief="raised", width=20, height=2)
predict_button.grid(row=len(labels)+1, column=0, columnspan=2, pady=20)

# Start the GUI event loop
root.mainloop()
