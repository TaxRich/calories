import joblib
import numpy as np

def predict_calories(input_features):
    model = joblib.load("../src/calorie_model.pkl")
    prediction = model.predict([input_features])
    return prediction[0]

if __name__ == "__main__":
    sample_input = [1, 30, 150, 60, 10,5,3]  # Example input
    result = predict_calories(sample_input)
    print(f"Predicted Calories Burned: {result}")
