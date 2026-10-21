
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# index.html is in the same folder as app.py
app = Flask(__name__, template_folder=BASE_DIR)

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully")
except Exception:
    app.logger.exception("Failed to load model")
    model = None


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if model is None:
        return (
            "Model could not be loaded. Check the VS Code terminal "
            "for the model-loading traceback.",
            500
        )

    if request.method == "POST":
        try:
            car_name = request.form.get("Car_Name", "").strip()
            year = int(request.form["Year"])
            present_price = float(request.form["Present_Price"])
            driven_kms = int(request.form["Driven_kms"])
            fuel_type = request.form["Fuel_Type"]
            selling_type = request.form["Selling_type"]
            transmission = request.form["Transmission"]
            owner = int(request.form["Owner"])

            # Validate input
            if not car_name:
                raise ValueError("Please enter a car name.")

            if not 1900 <= year <= 2026:
                raise ValueError("Enter a valid manufacturing year.")

            if present_price <= 0:
                raise ValueError("Present price must be greater than zero.")

            if driven_kms < 0:
                raise ValueError("Kilometers driven cannot be negative.")

            if fuel_type not in ["Petrol", "Diesel", "CNG"]:
                raise ValueError("Invalid fuel type.")

            if selling_type not in ["Dealer", "Individual"]:
                raise ValueError("Invalid seller type.")

            if transmission not in ["Manual", "Automatic"]:
                raise ValueError("Invalid transmission.")

            if owner not in [0, 1, 2, 3]:
                raise ValueError("Invalid owner value.")

            # Match the original training feature names
            input_data = pd.DataFrame([{
                "Car_Name": car_name,
                "Year": year,
                "Present_Price": present_price,
                "Driven_kms": driven_kms,
                "Fuel_Type": fuel_type,
                "Selling_type": selling_type,
                "Transmission": transmission,
                "Owner": owner
            }])

            # Predict selling price
            predicted_price = float(model.predict(input_data)[0])

            prediction = f"{predicted_price:.2f}"

        except Exception as exc:
            app.logger.exception("Prediction failed")
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
