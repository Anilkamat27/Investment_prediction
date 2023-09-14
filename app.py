from flask import Flask, request, render_template
import main  # Import your main code as a module
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

# Define a route to display the prediction form
@app.route('/predict')
def prediction_form():
    return render_template('prediction.html', prediction=None, color_class=None)

# Define a route to accept input data and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request (you may need to adjust this depending on how the data is sent)
        data = request.form  # Assuming form data is sent

        # Extract the stock_symbol from the input data
        stock_symbol = data.get('stock_symbol')

        # Call the main function from your main code
        prediction_result = main.main(stock_symbol)

        if prediction_result is not None:
            # Calculate the previous day's date (one day before today)
            today = datetime.now()
            previous_day_date = today - timedelta(days=1)

            # Fetch historical data for the stock using yfinance
            stock_data = yf.download(stock_symbol, start=previous_day_date, end=today)

            if not stock_data.empty:
                # Extract the previous day's closing price
                previous_day_price = stock_data['Close'].iloc[-1]

                # Calculate whether the predicted price is higher or lower
                if float(prediction_result) > previous_day_price:
                    color_class = 'green'
                elif float(prediction_result) < previous_day_price:
                    color_class = 'red'
                else:
                    color_class = 'default'  # No color change if prices are equal

                # Render the prediction result in the HTML template with the color class
                return render_template('prediction.html', prediction=prediction_result, color_class=color_class, stock_symbol=stock_symbol)

            else:
                return render_template('prediction.html', prediction='Failed to fetch historical data', color_class=None, stock_symbol=stock_symbol)

        else:
            return render_template('prediction.html', prediction='Failed to make predictions', color_class=None, stock_symbol=stock_symbol)

    except Exception as e:
        return render_template('prediction.html', prediction=str(e), color_class=None, stock_symbol=stock_symbol)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=True)
