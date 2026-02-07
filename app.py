from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("churn_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Customer Churn Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json(force=True)

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df).max()

        # Return result
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5050)  # Use 5050 to avoid port conflicts
