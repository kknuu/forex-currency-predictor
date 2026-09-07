import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')

# Title
st.title("Future Forex Currency Price Prediction Model")

# Currency options
options = {
    "AUSTRALIAN DOLLAR": "AUSTRALIA - AUSTRALIAN DOLLAR/US$",
    "EURO": "EURO AREA - EURO/US$",
    "NEW ZEALAND DOLLAR": "NEW ZEALAND - NEW ZELAND DOLLAR/US$",
    "GREAT BRITAIN POUNDS": "UNITED KINGDOM - UNITED KINGDOM POUND/US$",
    "BRAZILIAN REAL": "BRAZIL - REAL/US$",
    "CANADIAN DOLLAR": "CANADA - CANADIAN DOLLAR/US$",
    "CHINESE YUAN": "CHINA - YUAN/US$",
    "HONG KONG DOLLAR": "HONG KONG - HONG KONG DOLLAR/US$",
    "INDIAN RUPEE": "INDIA - INDIAN RUPEE/US$",
    "KOREAN WON": "KOREA - WON/US$",
    "MEXICAN PESO": "MEXICO - MEXICAN PESO/US$",
    "SOUTH AFRICAN RAND": "SOUTH AFRICA - RAND/US$",
    "SINGAPORE DOLLAR": "SINGAPORE - SINGAPORE DOLLAR/US$",
    "DANISH KRONE": "DENMARK - DANISH KRONE/US$",
    "JAPANESE YEN": "JAPAN - YEN/US$",
    "MALAYSIAN RINGGIT": "MALAYSIA - RINGGIT/US$",
    "NORWEGIAN KRONE": "NORWAY - NORWEGIAN KRONE/US$",
    "SWEDISH KRONA": "SWEDEN - KRONA/US$",
    "SRI LANKAN RUPEE": "SRI LANKA - SRI LANKAN RUPEE/US$",
    "SWISS FRANC": "SWITZERLAND - FRANC/US$",
    "NEW TAIWAN DOLLAR": "TAIWAN - NEW TAIWAN DOLLAR/US$",
    "THAI BAHT": "THAILAND - BAHT/US$",
}

# Directories
MODELS_DIR = "models"
DATA_FILE = "Foreign_Exchange_Rates.xls"

# Best models for each currency
BEST_MODELS = {
    "AUSTRALIA - AUSTRALIAN DOLLAR/US$": "LSTM",
    "EURO AREA - EURO/US$": "LSTM",
    "NEW ZEALAND - NEW ZELAND DOLLAR/US$": "LSTM",
    "UNITED KINGDOM - UNITED KINGDOM POUND/US$": "Prophet",
    "BRAZIL - REAL/US$": "ARIMA",
    "CANADA - CANADIAN DOLLAR/US$": "LSTM",
    "CHINA - YUAN/US$": "Prophet",
    "HONG KONG - HONG KONG DOLLAR/US$": "ARIMA",
    "INDIA - INDIAN RUPEE/US$": "ARIMA",
    "KOREA - WON/US$": "ARIMA",
    "MEXICO - MEXICAN PESO/US$": "ARIMA",
    "SOUTH AFRICA - RAND/US$": "ARIMA",
    "SINGAPORE - SINGAPORE DOLLAR/US$": "LSTM",
    "DENMARK - DANISH KRONE/US$": "ARIMA",
    "JAPAN - YEN/US$": "ARIMA",
    "MALAYSIA - RINGGIT/US$": "LSTM",
    "NORWAY - NORWEGIAN KRONE/US$": "ARIMA",
    "SWEDEN - KRONA/US$": "Prophet",
    "SRI LANKA - SRI LANKAN RUPEE/US$": "ARIMA",
    "SWITZERLAND - FRANC/US$": "LSTM",
    "TAIWAN - NEW TAIWAN DOLLAR/US$": "Prophet",
    "THAILAND - BAHT/US$": "ARIMA",
}

# Create model file name
def get_model_filename(currency):
    filename = (
        currency
        .replace("/", "_")
        .replace(" ", "_")
        .replace("$", "")
        .replace("-", "_")
    )
    return f"{filename}_model"

# Load pre-trained model
def load_model(currency):
    model_type = BEST_MODELS[currency]
    base_filename = get_model_filename(currency)

    # LSTM
    if model_type == "LSTM":
        filename = f"{base_filename}.keras"
        path = os.path.join(MODELS_DIR, filename)
        from tensorflow.keras.models import load_model
        model = load_model(path)
        return model

    # ARIMA / PROPHET
    filename = f"{base_filename}.pkl"
    path = os.path.join(MODELS_DIR, filename)
    model = joblib.load(path)
    return model

# Load historical data
def load_data():
    data = pd.read_csv(DATA_FILE)
    return data

# ARIMA forecast
def forecast_arima(model, forecast_length):
    forecast_length = int(forecast_length)
    prediction = model.predict(n_periods=forecast_length)
    prediction = np.asarray(prediction)
    return pd.DataFrame({"Forecast": prediction})

# PROPHET forecast
def forecast_prophet(model, forecast_length, last_date):
    forecast_length = int(forecast_length)
    future = model.make_future_dataframe(periods=forecast_length, freq="D")
    forecast = model.predict(future)
    prediction = forecast[["ds", "yhat"]].iloc[-forecast_length:].copy()
    prediction.columns = ["Date", "Forecast"]
    prediction = prediction.reset_index(drop=True)
    return prediction

# LSTM FORECAST
def forecast_lstm(model, data, currency, forecast_length, input_sequence=30):
    forecast_length = int(forecast_length)

    # Extract historical values
    values = pd.to_numeric(data[currency], errors="coerce").dropna().to_numpy()

    # Determine expected input sequence
    try:
        expected_sequence_length = (model.input_shape[1])
        if expected_sequence_length is not None:
            input_sequence = int(expected_sequence_length)
    except Exception:
        pass

    # Start with most recent observations
    history = list(values[-input_sequence:])
    predictions = []

    # Recursive forecasting
    for _ in range(forecast_length):
        sequence = np.array(history[-input_sequence:], dtype=float)
        X = sequence.reshape(1, input_sequence, 1)
        prediction = model.predict(X, verbose=0)
        next_value = float(np.asarray(prediction).flatten()[0])
        predictions.append(next_value)
        history.append(next_value)

    # Create forecast dates
    if "Time Serie" in data.columns:
        dates = pd.to_datetime(data["Time Serie"], errors="coerce").dropna()
        if len(dates) > 0:
            last_date = dates.iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),periods=forecast_length,freq="D")
            return pd.DataFrame({"Date": future_dates, "Forecast": predictions})
    return pd.DataFrame({"Forecast": predictions})


# GENERAL FORECAST FUNCTION
def make_forecast(currency, forecast_length, data):
    model_type = BEST_MODELS[currency]

    # Load pre-trained model
    model = load_model(currency)

    # ARIMA
    if model_type == "ARIMA":
        forecast_df = forecast_arima(model, forecast_length)

    # PROPHET
    elif model_type == "Prophet":
        last_date = None
        if "Time Serie" in data.columns:
            dates = pd.to_datetime(data["Time Serie"], errors="coerce").dropna()
            if not dates.empty:
                last_date = dates.max()
        forecast_df = forecast_prophet(model, forecast_length, last_date)

    # LSTM
    elif model_type == "LSTM":
        forecast_df = forecast_lstm(model, data, currency, forecast_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return forecast_df, model_type

# STREAMLIT form
with st.form(key="user_form"):
    selected_option = st.selectbox("Choose a currency:", list(options.keys()))
    forecast_length = st.number_input("Days to forecast", min_value=1, max_value=100, value=1, step=1)
    submit_button = st.form_submit_button(label="Generate Predictions")

# Generate forecast
if submit_button:
    selected_currency = options[selected_option]

    # Load data
    data = load_data()

    # Get selected model type
    model_type = BEST_MODELS[selected_currency]

    # Generate forecast
    forecast_df, model_type = make_forecast(selected_currency, forecast_length, data)
    if "Date" in forecast_df.columns:
        st.line_chart(forecast_df, x="Date", y="Forecast")
    else:
        st.line_chart(forecast_df, y="Forecast")

    st.dataframe(forecast_df, use_container_width=True)