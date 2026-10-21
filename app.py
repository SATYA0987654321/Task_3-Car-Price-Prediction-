from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained ML pipeline
model = joblib.load("car_price_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            car_data = {
                "Car_Name": request.form["Car_Name"].strip(),
                "Year": int(request.form["Year"]),
                "Present_Price": float(request.form["Present_Price"]),
                "Driven_kms": int(request.form["Driven_kms"]),
                "Fuel_Type": request.form["Fuel_Type"],
                "Selling_type": request.form["Selling_type"],
                "Transmission": request.form["Transmission"],
                "Owner": int(request.form["Owner"])
            }

            if not car_data["Car_Name"]:
                raise ValueError("Please enter the car name.")

            if not 1990 <= car_data["Year"] <= 2026:
                raise ValueError("Please enter a valid manufacturing year.")

            if car_data["Present_Price"] <= 0:
                raise ValueError("Present price must be greater than zero.")

            if car_data["Driven_kms"] < 0:
                raise ValueError("Kilometers driven cannot be negative.")

            input_df = pd.DataFrame([car_data])

            predicted_price = model.predict(input_df)[0]
            prediction = f"{predicted_price:.2f}"

        except (ValueError, KeyError) as e:
            error = str(e)
        except Exception:
            error = "Prediction failed. Please check the entered details."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
