import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model():
    calories_df = pd.read_csv("../data/calories.csv")
    exercise_df = pd.read_csv("../data/exercise.csv")

    data = pd.concat([exercise_df, calories_df["Calories"]], axis=1)
    data.replace({"Gender": {"male": 0, "female": 1}}, inplace=True)

    X = data.drop(columns=["User_ID", "Calories"], axis=1)
    y = data["Calories"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model trained! MAE: {mae}, R2 Score: {r2}")

    joblib.dump(model, "../src/calorie_model.pkl")  

if __name__ == "__main__":
    train_model()
