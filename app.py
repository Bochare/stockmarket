import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import altair as alt
import requests
import os

# Define the URL of the model file on GitHub
MODEL_URL = "https://github.com/Bochare/stockmarket/raw/main/Stock%20Predictions%20Model1.keras"
MODEL_PATH = "Stock_Predictions_Model1.keras"

def download_model():
    """Download the model file if it doesn't already exist."""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model file...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        st.success("Model downloaded successfully!")

# Download the model
download_model()

# Load the model
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
# Set up the Streamlit app header
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for view selection
if "view" not in st.session_state:
    st.session_state.view = "Stock Data"  # Default view

# Define a function to set the view based on button clicks
def set_view(view_name):
    st.session_state.view = view_name

# Display all available stocks in a data table
st.header("Stock Market Dashboard")
st.subheader("Available Stocks")
available_stocks = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA', 'RELIANCE.NS', 'SBIN.NS', 'HDFCBANK.NS', 'NIFTY_FIN_SERVICE.NS']
all_data = []

# Fetch historical data for all available stocks
for stock in available_stocks:
    data = yf.download(stock, start='2024-01-01', end='2024-11-20', interval='1d')
    if not data.empty:
        latest_close = data['Close'].iloc[-1] if 'Close' in data.columns else None
        all_data.append({"Stock": stock, "Latest Close Price": latest_close})

stocks_df = pd.DataFrame(all_data)
st.dataframe(stocks_df)

# Sidebar for searching a particular stock and selecting date range
st.sidebar.header("Search and Analyze Stocks")
stock = st.sidebar.selectbox("Select a Stock", available_stocks)

# Date input for start and end date
start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 11, 20))

# Validate date range
if start_date > end_date:
    st.sidebar.error("End Date must be after Start Date.")
else:
    # Download data for the selected stock with the user-provided date range
    data = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # Handle empty data
    if data.empty:
        st.error(f"No data available for the selected stock: {stock}")
    else:
        # Add buttons for navigation
        st.sidebar.subheader("Select View")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            if st.button("Stock Data"):
                set_view("Stock Data")
        with col2:
            if st.button("Moving Averages"):
                set_view("Moving Averages")
        with col3:
            if st.button("Price Prediction"):
                set_view("Price Prediction")

        # Render content based on the selected view
        if st.session_state.view == "Stock Data":
            st.subheader(f"Stock Data for {stock} ({start_date} to {end_date})")
            st.write(data)

        elif st.session_state.view == "Moving Averages":
            st.subheader("Moving Averages")
            ma_50_days = data.Close.rolling(50).mean()
            ma_100_days = data.Close.rolling(100).mean()
            ma_200_days = data.Close.rolling(200).mean()

            # Plot: Price vs MA50
            fig1 = plt.figure(figsize=(8, 6))
            plt.plot(ma_50_days, 'r', label="MA50")
            plt.plot(data.Close, 'g', label="Close Price")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Show at most 10 evenly spaced dates
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines for better visual aid
            st.plotly_chart(fig1, use_container_width=True)

            # Plot: Price vs MA50 vs MA100
            fig2 = plt.figure(figsize=(8, 6))
            plt.plot(ma_50_days, 'r', label="MA50")
            plt.plot(ma_100_days, 'b', label="MA100")
            plt.plot(data.Close, 'g', label="Close Price")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Show at most 10 evenly spaced dates
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines for better visual aid
            st.plotly_chart(fig2)

            # Plot: Price vs MA100 vs MA200
            fig3 = plt.figure(figsize=(8, 6))
            plt.plot(ma_100_days, 'r', label="MA100")
            plt.plot(ma_200_days, 'b', label="MA200")
            plt.plot(data.Close, 'g', label="Close Price")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Price")
            st.plotly_chart(fig3)

        elif st.session_state.view == "Price Prediction":
            st.subheader("Price Prediction")

            # Split data into training (80%) and testing (20%) sets
            data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
            data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

            # Normalize the data using MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_train_scaled = scaler.fit_transform(data_train)
            pas_100_days = data_train_scaled[-100:]  # Last 100 days from training data
            data_test_scaled = scaler.transform(data_test)
            combined_data = np.concatenate((pas_100_days, data_test_scaled), axis=0)

            # Prepare the data for prediction
            x, y = [], []
            for i in range(100, combined_data.shape[0]):
                x.append(combined_data[i - 100:i])  # Previous 100 days as input
                y.append(data_test_scaled[i - 100, 0])  # Actual price at the target day

            x, y = np.array(x), np.array(y)

            # Make predictions for the testing set
            predictions_scaled = model.predict(x)
            predictions = scaler.inverse_transform(predictions_scaled)  # Rescale predictions
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1))  # Rescale actual prices

            # Extend predictions for the next 30 days
            future_predictions = []
            last_input = combined_data[-100:]  # Take the last 100 days as input

            for _ in range(30):  # Predict for 30 days
                next_pred_scaled = model.predict(last_input.reshape(1, 100, 1))
                future_predictions.append(scaler.inverse_transform(next_pred_scaled)[0][0])  # Rescale each prediction
                last_input = np.append(last_input[1:], next_pred_scaled, axis=0)

            # Create a DataFrame for future predictions
            last_date = data.index[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

            # Plot: Original vs Predicted Prices (testing set)
            st.subheader("Original Price vs Predicted Price")
            fig4 = plt.figure(figsize=(8, 6))
            plt.plot(data.index[-len(actual_prices):], actual_prices, 'g', label="Actual Price")
            plt.plot(data.index[-len(predictions):], predictions, 'r', label="Predicted Price")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            st.plotly_chart(fig4)

            # Plot: Future Predictions
            st.subheader("30-Day Future Predictions")
            fig5 = plt.figure(figsize=(12, 6))  # Increased figure size for better readability
            plt.plot(future_df['Date'], future_df['Predicted Price'], 'b', label="Future Predictions")
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.xticks(rotation=45, fontsize=10)  # Rotate and resize x-axis labels for clarity
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Show at most 10 evenly spaced dates
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines for better visual aid
            st.plotly_chart(fig5)


            # Display future predictions in a table
            st.subheader("Future Predictions Table")
            st.write(future_df)
