import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

files = {
    "S&P": "Processed_S&P.csv",
    "RUSSELL": "Processed_RUSSELL.csv",
    "NYSE": "Processed_NYSE.csv",
    "NASDAQ": "Processed_NASDAQ.csv",
    "DJI": "Processed_DJI.csv"
}

# File selection dropdown
st.sidebar.title("Dataset Selection")
selected_file = st.sidebar.selectbox('Choose dataset', list(files.keys()))

# Load selected file
df = pd.read_csv(files[selected_file])

# Visualization
st.title(f'{selected_file} Stock Trend Prediction Using LSTM')

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load Model
model = load_model('LSTM_keras_model.h5')

# Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
y_pred = model.predict(X_test)

# Scaling up
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Display RMSE and MAE
st.subheader('Evaluation Metrics')
st.write(f"Root Mean Squared Error (RMSE): **{rmse:.2f}**")
st.write(f"Mean Absolute Error (MAE): **{mae:.2f}**")

# Plotting
st.subheader('Model Prediction Plot')
fig2 = plt.figure(figsize=(6,4))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
