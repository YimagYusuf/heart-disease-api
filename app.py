from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load your model
model = joblib.load('heart_disease_model.pkl')

# Set up Flask
app = Flask(__name__)
CORS(app)

predictors = ["Age", "Gender", "Heart rate", "Systolic blood pressure",
              "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data], columns=predictors)
    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)






# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd

# # Load the trained model
# model = joblib.load('heart_disease_model.pkl')

# # List of expected input features
# predictors = ["Age", "Gender", "Heart rate", "Systolic blood pressure",
#               "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get JSON data
#     data = request.get_json(force=True)
    
#     # Turn JSON into a DataFrame
#     input_df = pd.DataFrame([data], columns=predictors)
    
#     # Make a prediction
#     prediction = model.predict(input_df)[0]
    
#     return jsonify({'prediction': int(prediction)})

# if __name__ == '__main__':
#     app.run(debug=True)