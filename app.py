from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# List of expected input features
predictors = ["Age", "Gender", "Heart rate", "Systolic blood pressure",
              "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data
    data = request.get_json(force=True)
    
    # Turn JSON into a DataFrame
    input_df = pd.DataFrame([data], columns=predictors)
    
    # Make a prediction
    prediction = model.predict(input_df)[0]
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)