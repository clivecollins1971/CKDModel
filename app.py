from flask import Flask, request
from flask_restful import Api, Resource
import pandas as pd
# from sklearn.externals import joblib
import pickle

app = Flask(__name__)
api = Api(app)

# Load pipeline and model using the binary files
model_lr = pickle.load(open('logistic_model.pkl', 'rb'))
model_svm = pickle.load(open('svm_model.pkl', 'rb'))
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

# Function to test if the request contains multiple
def islist(obj):
  return True if ("list" in str(type(obj))) else False


class PredsLr(Resource):
  def put(self):
    json_ = request.json
    # If there are multiple records to be predicted, directly convert the request json file into a pandas dataframe
    if islist(json_['age']):
      entry = pd.DataFrame(json_)
    # In the case of a single record to be predicted, convert json request data into a list and then to a pandas
    # dataframe
    else:
      entry = pd.DataFrame([json_])
    # Transform request data record/s using the pipeline
    entry_transformed = pipeline.transform(entry)
    # Make predictions using transformed data
    prediction = model_lr.predict(entry_transformed)
    res = {'predictions': {}}
    # Create the response
    for i in range(len(prediction)):
      res['predictions'][i + 1] = str(prediction[i])
    return res, 200 # Send the response object


class PredsSvm(Resource):
  def put(self):
    json_ = request.json
    # If there are multiple records to be predicted, directly convert the request json file into a pandas dataframe
    if islist(json_['age']):
      entry = pd.DataFrame(json_)
    # In the case of a single record to be predicted, convert json request data into a list and then to a pandas
    # dataframe
    else:
      entry = pd.DataFrame([json_])
    # Transform request data record/s using the pipeline
    entry_transformed = pipeline.transform(entry)
    # Make predictions using transformed data
    prediction = model_svm.predict(entry_transformed)
    res = {'predictions': {}}
    # Create the response
    for i in range(len(prediction)):
      res['predictions'][i + 1] = str(prediction[i])
    return res, 200 # Send the response object


api.add_resource(PredsLr, '/predictlr')

api.add_resource(PredsSvm, '/predictsvm')

if __name__ == "__main__":
  app.run(debug = True)