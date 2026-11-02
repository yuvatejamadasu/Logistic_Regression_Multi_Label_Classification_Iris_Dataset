from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# --------------------------------------------------
# Load the trained Logistic Regression model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Logistic_Regression_Multi_Label_Classification_Iris.pkl"
)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Class labels used while training the model
IRIS_CLASSES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


# --------------------------------------------------
# Main route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    input_values = {
        "sepal_length": "",
        "sepal_width": "",
        "petal_length": "",
        "petal_width": ""
    }

    if request.method == "POST":

        try:
            # Read the form values
            sepal_length = float(request.form.get("sepal_length", ""))
            sepal_width = float(request.form.get("sepal_width", ""))
            petal_length = float(request.form.get("petal_length", ""))
            petal_width = float(request.form.get("petal_width", ""))

            # Basic validation
            values = [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]

            if not all(np.isfinite(value) for value in values):
                raise ValueError("Please enter valid finite numbers.")

            if any(value <= 0 for value in values):
                raise ValueError("Measurements must be greater than zero.")

            # Keep the values visible after submitting the form
            input_values = {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }

            # Feature order must match the training data
            input_data = np.array([[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]])

            # Predict the class
            predicted_class = int(model.predict(input_data)[0])

            prediction = IRIS_CLASSES.get(
                predicted_class,
                "Unknown Iris Species"
            )

            # Get probability for the predicted class
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = round(
                    float(np.max(probabilities)) * 100,
                    2
                )

        except ValueError as exc:
            error = str(exc) or "Please enter valid numeric measurements."

        except Exception:
            error = (
                "Unable to generate a prediction. "
                "Please verify the model file and input values."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        input_values=input_values
    )


# --------------------------------------------------
# Run the application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)