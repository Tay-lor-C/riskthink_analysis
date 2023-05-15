from flask import Flask, jsonify, request
import joblib
import logging 
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# set up logging
app.logger.setLevel(logging.INFO)
app.logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.info('App Started')


# load trained RF model
model = joblib.load('rf_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
	
	print('hello-world')
	# Get request data


	data = request.get_json()

	# extract moving average and moving median features from data
	vol_moving_avg = data['vol_moving_avg']
	adj_close_rolling_med = data['adj_close_rolling_med']

	# make prediction
	prediction = int(model.predict([[vol_moving_avg, adj_close_rolling_med]]))

	# return prediction as JSON
	return jsonify({'prediction' : prediction})

if __name__ == '__main__':
	app.run(port=8081, debug = True)


