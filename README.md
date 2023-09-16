# Investment_prediction

🎥 Demo: You can watch a video demo of my project

https://github.com/Anilkamat27/Investment_prediction/assets/71273338/e79fc4ae-2acc-4268-83ad-94eedfd54a66

My project is focused on stock price prediction using time series analysis. We leverage historical stock data, employ ARIMA (AutoRegressive Integrated Moving Average) models for forecasting, and evaluate various model configurations. It is a one-click stock price prediction solution that automatically fetches and transforms historical stock data from the internet with a single click.
It checks for stationarity and identifies the best values for 'p,' 'd,' and 'q' to train an ARIMA model, enabling you to predict the next day's stock price based on the entered stock symbol.

### create a environment
```
conda create -p venv python==3.11
conda activate venv/
```
### Install all necessary libraries
```
pip install -r requirements.txt
```
### To run project in local host 
```
python app.py
```
