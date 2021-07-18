import os
import requests
import bq
import joblib
import model
import numpy as np
import pandas as pd
import sklearn
import xgboost

from flask import Flask
from flask_restful import Resource, Api, reqparse


app = Flask(__name__)
api = Api(app)

feature_list = [
    'Age',
  'Education',
  'Default',
  'Balance',
  'HHInsurance',
  'CarLoan',
  'LastContactDay',
  'NoOfContacts',
  'DaysPassed',
  'PrevAttempts',
  'Job_admin.',
  'Job_blue-collar',
  'Job_entrepreneur',
  'Job_housemaid',
  'Job_management',
  'Job_retired',
  'Job_self-employed',
  'Job_services',
  'Job_student',
  'Job_technician',
  'Job_unemployed',
  'Job_nan',
  'Communication_cellular',
  'Communication_telephone',
  'Communication_nan',
  'Outcome_failure',
  'Outcome_other',
  'Outcome_success',
  'Outcome_nan',
  'Call_duration ',
  'Period_of_day_call',
  'Marital_divorced',
  'Marital_married',
  'Marital_single',
  'LastContactMonth_apr',
  'LastContactMonth_aug',
  'LastContactMonth_dec',
  'LastContactMonth_feb',
  'LastContactMonth_jan',
  'LastContactMonth_jul',
  'LastContactMonth_jun',
  'LastContactMonth_mar',
  'LastContactMonth_may',
  'LastContactMonth_nov',
  'LastContactMonth_oct',
  'LastContactMonth_sep'
]

class Prediction(Resource):


    def get(self):

        parser = reqparse.RequestParser()
        for feature in feature_list:
            parser.add_argument(feature)


        args = parser.parse_args() # dictionary of args


        np_array = np.fromiter(args.values(), dtype=float)  # convert to np array
        np_array = np.array(np_array).reshape(1,46)  # reshape to correct format
        df = pd.DataFrame(np_array, columns = feature_list)  # convert to dataframe

        def get_model():
            return model.run_()

        trained_model = get_model()

        prediction = trained_model.predict(df)
        prediction = dict(enumerate(prediction.flatten(), 1)) #reshape prediction, into json compatible dictionary format
        prediction[1] = int(prediction[1]) # prediction needs to be of type int for json, not float
        out = {'Prediction': prediction}


        return out, 200

api.add_resource(Prediction, '/')

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=True, port=server_port, host='0.0.0.0')
