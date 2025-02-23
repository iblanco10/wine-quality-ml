from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("wine_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Wine Quality Prediction API is running!",
        "usage": "Send a POST request to /predict with JSON data in the format: {'features': [fixed_acidity, volatile_acidity, ...]}"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        
        # Scale input features
        features_scaled = scaler.transform(features)
        
        # Predict wine quality
        prediction = model.predict(features_scaled)
        
        # Convert to integer since quality scores are whole numbers
        predicted_quality = int(round(prediction[0]))

        return jsonify({"predicted_quality": predicted_quality})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app (for local testing)
if __name__ == "__main__":
    app.run(debug=True)
