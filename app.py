from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load the trained model and scaler
model = joblib.load("wine_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Route for home page (Interactive form)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get form data
            features = [
                float(request.form["fixed_acidity"]),
                float(request.form["volatile_acidity"]),
                float(request.form["citric_acid"]),
                float(request.form["residual_sugar"]),
                float(request.form["chlorides"]),
                float(request.form["free_sulfur_dioxide"]),
                float(request.form["total_sulfur_dioxide"]),
                float(request.form["density"]),
                float(request.form["pH"]),
                float(request.form["sulfates"]),
                float(request.form["alcohol"]),
            ]
            
            # Scale input features
            features_scaled = scaler.transform([features])

            # Predict wine quality
            prediction = model.predict(features_scaled)
            predicted_quality = int(round(prediction[0]))

            return render_template("index.html", prediction=predicted_quality)
        
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", prediction=None)

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
