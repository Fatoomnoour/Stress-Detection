from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

import numpy as np
from pydantic import BaseModel

app = FastAPI(
    title="Stress Level Prediction API",
    description="API for predicting stress level based on health metrics",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor
try:
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("stress_predictor.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model files: {str(e)}")


# Define request model
class StressPredictionInput(BaseModel):
    gender: str
    age: int
    occupation: str
    sleep_duration: float
    sleep_quality: int
    bmi_category: str
    heart_rate: int
    daily_steps: int
    systolic_bp: int
    diastolic_bp: int


# Define response model
class PredictionData(BaseModel):
    prediction: int


class StressPredictionOutput(BaseModel):
    success: bool
    data: PredictionData
    message: str


@app.post("/predict", response_model=StressPredictionOutput)
async def predict_stress_level(data: StressPredictionInput):
    """
    Predict stress level based on health metrics

    Parameters:
    - **gender**: Male/Female
    - **age**: User's age (18-100)
    - **occupation**: User's occupation
    - **sleep_duration**: Hours of sleep per night
    - **sleep_quality**: Quality of sleep (1-10 scale)
    - **bmi_category**: Underweight/Normal/Overweight/Obese
    - **heart_rate**: Resting heart rate (bpm)
    - **daily_steps**: Average daily steps
    - **systolic_bp**: Systolic blood pressure
    - **diastolic_bp**: Diastolic blood pressure

    Returns:
    - Predicted stress level (0-10 scale)
    """
    try:
        # Create DataFrame from input data
        input_data = {
            "Gender": data.gender,
            "Age": data.age,
            "Occupation": data.occupation,
            "Sleep Duration": data.sleep_duration,
            "Quality of Sleep": data.sleep_quality,
            "BMI Category": data.bmi_category,
            "Heart Rate": data.heart_rate,
            "Daily Steps": data.daily_steps,
            "Systolic BP": data.systolic_bp,
            "Diastolic BP": data.diastolic_bp,
        }

        input_df = pd.DataFrame([input_data])

        # Calculate derived features
        input_df["BP_Ratio"] = input_df["Diastolic BP"] / input_df["Systolic BP"]
        input_df["Age_Group"] = pd.cut(
            input_df["Age"],
            bins=[18, 30, 45, 60, 100],
            labels=["Young", "Adult", "Mid-Age", "Senior"],
        )

        # Preprocess and predict
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)

        return {
            "success": True,
            "data": {"prediction": int(prediction[0])},
            "message": "Prediction completed successfully",
        }

        # return StressPredictionOutput(
        #     status="success",
        #     prediction=int(prediction[0]),
        #     message="Prediction completed successfully",

        # )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Stress Level Prediction API",
        "documentation": "/docs",
        "version": "1.0.0",
    }
