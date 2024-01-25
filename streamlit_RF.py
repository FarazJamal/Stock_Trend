import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# File paths for the CSV files
files = {
    "S&P": "Processed_S&P.csv",
    "RUSSELL": "Processed_RUSSELL.csv",
    "NYSE": "Processed_NYSE.csv",
    "NASDAQ": "Processed_NASDAQ.csv",
    "DJI": "Processed_DJI.csv"
}

def process_dataset(file):
    df = pd.read_csv(file)
    
    # Data Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    # Feature Engineering (using 'Close' to create a moving average)
    df['MA_Close'] = df['Close'].rolling(window=5).mean().shift(1)  # 5 days moving average
    
    # Feature and Target Selection
    features = ['Close', 'MA_Close']
    df.dropna(inplace=True)
    X = df[features]
    y = df['Close'].shift(-1)  # Predicting the next day's closing price
    y = y[:-1]
    X = X[:-1]

    return X, y

def train_and_evaluate(X, y):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ## Model Development
    # RandomForest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # GradientBoosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    # Model Evaluation
    results = []
    predictions_rf = rf_model.predict(X_test_scaled)
    predictions_gb = gb_model.predict(X_test_scaled)

    for model, name, predictions in zip([rf_model, gb_model], ['RandomForest', 'GradientBoosting'], [predictions_rf, predictions_gb]):
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        results.append((name, np.sqrt(mse), mae, predictions))

    return y_test, results

# Streamlit app layout
st.sidebar.title("Dataset Selection")
selected_dataset = st.sidebar.selectbox("Choose dataset", list(files.keys()))

# To display processed data and results
if selected_dataset:
    st.title(f'{selected_dataset} Stock Trend Prediction Using Random Forest')
    X, y = process_dataset(files[selected_dataset])

    # Training and Evaluation
    y_test, model_results = train_and_evaluate(X, y)
    st.subheader("Evaluation Metrics")
    for result in model_results:
        rmse_value = f"{result[1]:.2f}"
        mae_value = f"{result[2]:.2f}"
        st.write(f"{result[0]} - RMSE: **{rmse_value}**, MAE: **{mae_value}**")

    # Plotting
    st.subheader("Model Prediction Plot")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='Original Prices', color='blue')
    for result in model_results:
        ax.plot(result[3], label=f'{result[0]} Predicted Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)