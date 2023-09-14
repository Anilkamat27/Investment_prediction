import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from logger import logging
import traceback
import sys

logging.basicConfig(filename='stock_forecast.log', level=logging.ERROR)

def load_stock_data(stock_symbol):
    try:
        # Download stock data from Yahoo Finance
        stock_data = yf.download(stock_symbol, period="3y", auto_adjust=True)
        return stock_data
    except Exception as e:
        logging.error(f"Error loading stock data: {str(e)}")
        return None

def transform_stock_data(stock_data):
    try:
        stock_data = stock_data[['Close']]
        print(stock_data.head())
        return stock_data
    except Exception as e:
        logging.error(f"Error transforming stock data: {str(e)}")
        return None


def make_predictions(model, test_data):
    try:
        predictions = []
        history = [x for x in test_data]

        for t in range(len(test_data)):
            model_fit = model
            forecast = model_fit.forecast(steps=1)
            next_day_price = forecast[0]
            predictions.append(next_day_price)
            history.append(next_day_price)

        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        return None




def test_stationarity(timeseries):
    try:
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        plt.figure(figsize=(18, 8))
        plt.grid('both')
        plt.plot(timeseries, color='blue', linewidth=3)
        plt.plot(rolmean, color='red', linewidth=3)
        plt.plot(rolstd, color='black', linewidth=4)
        plt.legend(loc='best', fontsize=20, shadow=True, facecolor='lightpink', edgecolor='k')
        plt.title('Rolling Mean and Standard Deviation', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        #plt.show(block=False)
        print("Results of Dickey-Fuller test")
        adft = adfuller(timeseries, autolag='AIC')
        output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
        for key, values in adft[4].items():
            output['critical value (%s)' % key] = values
        print(output)
        return True
    except Exception as e:
        logging.error(f"Error testing stationarity: {str(e)}")
        return False

def train_arima_model(X, y, arima_order):
    try:
        print("in")
        history = [x for x in X]  # Initialize with historical data
        predictions = list()
        #print(X.head())
        #print(y.head())
        for t in range(len(y)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(y[t]) # Update history with the predicted value, not y[t]
            
        rmse = np.sqrt(mean_squared_error(y, predictions))
        print("out")
        return rmse, model_fit  # Return both RMSE and the trained ARIMA model
    except Exception as e:
        logging.error(f"Error training ARIMA model: {str(e)}")
        return None, None


def train_arima_model2(X, arima_order):
    try:
        history = [x for x in X]
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        return model_fit  # Return the trained ARIMA model
    except Exception as e:
        logging.error(f"Error training ARIMA model: {str(e)}")
        return None

def evaluate_model(X_train, y_test, p_values, d_values, q_values):
    try:
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        model = train_arima_model2(X_train, order)
                        if model is not None:
                            predictions = list()
                            for t in range(len(y_test)):
                                yhat = model.forecast()[0]
                                predictions.append(yhat)
                                #X_train.append(y_test[t])
                            rmse = np.sqrt(mean_squared_error(y_test, predictions))
                            if rmse is not None and rmse < best_score:
                                best_score, best_cfg = rmse, order
                            print('ARIMA%s RMSE=%.3f' % (order, rmse))
                    except Exception as e:
                        logging.error(f"Error evaluating ARIMA model for order {order}: {str(e)}")
                        continue
        print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
        return best_cfg  # Return the best ARIMA order
    except Exception as e:
        logging.error(f"Error evaluating models: {str(e)}")
        return None



def predict_stock_price(model, train_data):
    try:
        history = [x for x in train_data]
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        next_day_price = forecast[0]
        return next_day_price
    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        error_details = "".join(traceback_str)
        logging.error(f"Error predicting stock price: {error_message}\n{error_details}")
        return None

def main(stock_symbol):
    stock_data = load_stock_data(stock_symbol)
    if stock_data is None:
        return
    
    stock_data = transform_stock_data(stock_data)
    
    if stock_data is None:
        return

    if not test_stationarity(stock_data['Close']):
        return
    p_values = range(0, 3)
    d_values = range(0, 3)
    q_values = range(0, 3)
    
    best_order = evaluate_model(stock_data['Close'][:-60], stock_data['Close'][-60:], p_values, d_values, q_values)
    
    if best_order is None:
        return

    rmse, model = train_arima_model(stock_data['Close'][:-60], stock_data['Close'][-60:],best_order)

    if model is None:
        return


    predictions = make_predictions(model, stock_data['Close'][-60:])

    if predictions is not None:
        mse, mae = rmse, np.sqrt(mean_squared_error(stock_data['Close'][-60:], predictions))
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
    
        # Predict next day's stock price
        next_day_price = predictions[-1]  # Get the last prediction as the next day's price
        print("Predicted Next Day's Stock Price:", next_day_price)
        return next_day_price
    
    return None 

if __name__ == "__main__":
    try:
        file_path = r"C:\Users\anilk\Desktop\investment Prediction\ADANIENT_NS.csv"
        main(file_path)
    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        error_details = "".join(traceback_str)
        logging.error(f"Unhandled error: {error_message}\n{error_details}")
