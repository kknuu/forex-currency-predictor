# Forex Currency Predictor
An end-to-end machine learning project for forecasting foreign exchange rates across multiple currencies. The project covers data analysis, time-series modelling, model evaluation, interactive visualisation with Streamlit, and containerisation with Docker.

## Overview
This project explores foreign exchange data and applies machine learning and time-series forecasting techniques to predict future currency exchange rates.

The project was developed as an end-to-end workflow:
Data -> Exploration -> Preprocessing -> Model Training -> Evaluation -> Forecasting -> Streamlit Application -> Docker

The application allows users to select a currency and generate forecasts for a chosen prediction length.

## Objectives
The main objectives of this project were to:
- Explore historical foreign exchange data
- Clean and preprocess time-series data
- Analyse trends and relationships between currencies
- Experiment with forecasting techniques
- Evaluate model performance using appropriate error metrics
- Select suitable models for forecasting
- Build an interactive Streamlit application
- Package the application using Docker

## Dataset
The project uses the Foreign_Exchange_Rates.xls dataset containing historical exchange rate data for multiple currencies.

The dataset is stored in `data/Foreign_Exchange_Rates.xls`

The analysis notebook handles the loading, cleaning, data conversion, and exploration of the dataset, contained in `analysis.ipynb`

## Methodology
### 1. Data preprocessing
   Workflow includes:
   - Loading the dataset with Pandas
   - Converting date values into an appropriate date time format
   - Handling missing values
   - Selecting relevant currency series
   - Preparing the data for time-series modelling
   
### 2. Exploratory Data Analysis
   The data is explored using visualisations to identify:
   - Historical price trends
   - Currency movements
   - Correlations between currencies
   - Missing data
   - Potential patterns in the time series
   
### 3. Model development
   Multiple forecasting approaches were investigated and compared.
   
   Models used in the project include:
   - ARIMA
   - Prophet
   - LSTM
     
### 4. Model evaluation
   Models for each currency are evaluated using error metrics such as:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)
     
   The final model selection for each currency is based on the lowest RMSE score.

## Streamlit Application
The project includes an interactive Streamlit application allowing users to:
- Select a currency
- Choose the amount of days to forecast
- Load the corresponding trained model
- Generate future predictions
- View the forecast in tabular form
- Visualise the forecast using charts

### Run locally
Install the dependencies:

`pip install -r requirements.txt`

Start the application:

`streamline run app.py`

## Docker
Used to run the Streamlit application inside a container.

Build the Docker image:
`docker build -t forex-predictor .`

Run the container:
`docker run -p 8501:8501 forex-predictor`

Then open **http://localhost:8501** in your browser and select the currency you want to forecast.

## Project structure
    forex-currency-predictor
    |
    ├── data/
    |   └── Foreign_Exchange_Rates.xls
    |
    ├── analysis.ipynb
    ├── app.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md

## File descriptions

| Files             | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `analysis.ipynb`  | Data exploration, preprocessing, modelling and evaluation    |
| `app.py`          | Streamlit application for generating forecasts               |
| `data/`           | Historical foreign exchange dataset                          |
| `Dockerfile`      | Instructions for containerising the application.             |
| `requirements.txt`| Python dependencies                                          |
| `README.md`      | Project documentation                                        |
