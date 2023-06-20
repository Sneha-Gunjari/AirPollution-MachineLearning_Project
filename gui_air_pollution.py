import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

def predict():
    # Get the input data from the entry widgets
    input_data = [[float(entry.get()) for entry in input_entries]]

    # Make a prediction using the trained model
    prediction = rf.predict(input_data)

    # Update the output label with the prediction
    output_label.configure(text=f"Predicted Values: CO:{prediction[0][0]:.2f}, Benzene: {prediction[0][1]:.2f}, NOX:{prediction[0][2]:.2f}")

    # Create a plot of the predicted values vs. the actual values for each target variable
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].scatter(y_test['target_carbon_monoxide'], y_pred[:, 0], s=10)
    axs[0].set_xlabel('Actual CO')
    axs[0].set_ylabel('Predicted CO')
    axs[1].scatter(y_test['target_benzene'], y_pred[:, 1], s=10)
    axs[1].set_xlabel('Actual Benzene')
    axs[1].set_ylabel('Predicted Benzene')
    axs[2].scatter(y_test['target_nitrogen_oxides'], y_pred[:, 2], s=10)
    axs[2].set_xlabel('Actual NOX')
    axs[2].set_ylabel('Predicted NOX')
    plt.show()


# Load the data into a Pandas dataframe
data = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\mlproject\\train.csv')

# Separate the features (inputs) and targets
features = data.drop(['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides'], axis=1)
targets = data[['target_carbon_monoxide', 'target_benzene', 'target_nitrogen_oxides']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Train a Random Forest Regressor on the training set
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

msle = np.sqrt(mean_squared_log_error(y_test, y_pred, multioutput='raw_values')).mean()

root = tk.Tk()
root.title("Air Pollution Measurements Prediction")
root.geometry("1000x1000")

# bg_image = Image.open(os.path.join(os.path.dirname(file), 'C:\\Users\\DELL\\OneDrive\\Desktop\\mlproject\\backg.png'))
bg_image = Image.open(os.path.join(os.path.dirname(__file__), 'C:\\Users\\DELL\\OneDrive\\Desktop\\mlproject\\backg.png'))
bg_image = bg_image.resize((1500, 1000), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

input_frame = ttk.Frame(root, borderwidth=5, relief="ridge")
input_frame.place(relx=0.5, rely=0.15, relwidth=0.6, relheight=0.6, anchor="n")

# feature_labels = ['Deg C', 'Relative Humidity', 'Absolute Humidity', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8']
# feature_entries = []
# Create the input labels and entry widgets
input_labels = ['deg_C', 'relative_humidity', 'absolute_humidity', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
input_entries = []
for i, label in enumerate(input_labels):
    label = ttk.Label(input_frame, text=label, font=('Arial', 14))
    label.grid(row=i, column=0, padx=10, pady=10, sticky=tk.W)
    entry = ttk.Entry(input_frame, font=('Arial', 14))
    entry.grid(row=i, column=1, padx=10, pady=10)
    input_entries.append(entry)
# for i, label in enumerate(feature_labels):
# label = ttk.Label(input_frame, text=label, font=('Arial', 14))
# label.grid(row=i, column=0, padx=10, pady=10)
# entry = ttk.Entry(input_frame, font=('Arial', 14))
# entry.grid(row=i, column=1, padx=10, pady=10)
# feature_entries.append(entry)
def predict():
    # Get the input data from the entry widgets
    input_data = [[float(entry.get()) for entry in input_entries]]

    # Make a prediction using the trained model
    prediction = rf.predict(input_data)

    # Update the output label with the prediction
    output_label.configure(text=f"Predicted Values: CO:{prediction[0][0]:.2f}, Benzene: {prediction[0][1]:.2f}, NOX:{prediction[0][2]:.2f}")
      # Create a plot of the predicted values vs. the actual values for each target variable
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].scatter(y_test['target_carbon_monoxide'], y_pred[:, 0], s=10)
    axs[0].set_xlabel('Actual CO')
    axs[0].set_ylabel('Predicted CO')
    axs[1].scatter(y_test['target_benzene'], y_pred[:, 1], s=10)
    axs[1].set_xlabel('Actual Benzene')
    axs[1].set_ylabel('Predicted Benzene')
    axs[2].scatter(y_test['target_nitrogen_oxides'], y_pred[:, 2], s=10)
    axs[2].set_xlabel('Actual NOX')
    axs[2].set_ylabel('Predicted NOX')
    plt.show()

# Create the output label
output_label = ttk.Label(root, text="", font=('Arial', 14))
output_label.place(relx=0.5, rely=0.7, anchor="n")

# Create the predict button
predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.place(relx=0.5, rely=0.8, relwidth=0.1, relheight=0.1, anchor="n")

# Start the main event loop
root.mainloop()
