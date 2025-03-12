from src.train_model import train_model
from src.predict import predict_calories

if __name__ == "__main__":
    train_model()
    sample_input = [1, 30, 150, 60, 10,5, 3]  # Example input
    result = predict_calories(sample_input)
    print(f"Predicted Calories Burned: {result}")
