import json
import joblib
import mlflow
import logging
import uvicorn
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException, Request
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize FastAPI app
app = FastAPI(
    title="🚗 GetAround Pricing Prediction API",
    description="An API for predicting rental price per day using an MLflow model",
    version="1.0.0",
)

@app.get("/",  summary="Welcome Message", tags=["General"])
async def index():
    """
    Endpoint that serves a welcome message to the user.
    
    This is the default endpoint of the API, which is called when the user accesses 
    the root URL. It provides a brief description of the API's functionality and 
    guides the user to the documentation page for further details.
    
    Returns:
        str: A message explaining the purpose of the API and providing a link 
             to the documentation page.
    """
    
    # The message that will be returned when the root URL is accessed
    message = "Hello! This is an API that predicts the rental price per day of a car. Visit `/docs` for API documentation (https://farabouna-getaroudapispace.hf.space/docs)."
    
    # Return the welcome message to the user
    return message

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://farabouna-GetAroundPricing.hf.space/")  

# Load MLflow model
logged_model = 'runs:/c6c51acc60404547aa3eb7dafa254989/model'
try:
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

# Get latest run ID
experiment_name = "Pricing_prediction_experiment"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if runs.empty:
    raise Exception("No runs found for the experiment.")

latest_run_id = runs.iloc[0]["run_id"]

# Load pre-trained preprocessor
try:
    preprocessor_path = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path="model/preprocessor.pkl")
    preprocessor = joblib.load(preprocessor_path)
except Exception as e:
    print(f"Error loading preprocessor: {e}")
    preprocessor = None

# Define Pydantic model for input validation
class InputData(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# Define the expected features
expected_columns = ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color', 'car_type',
                    'private_parking_available', 'has_gps', 'has_air_conditioning', 'automatic_car',
                    'has_getaround_connect', 'has_speed_regulator', 'winter_tires']


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict", tags=["Machine Learning"])
async def predict(input_data: List[InputData]):
    """
    Predicts the rental price per day of a car using the provided input data.
    
    Expected input format is a list of objects with keys:
    - model_key, mileage, engine_power, fuel, paint_color, car_type, 
      private_parking_available, has_gps, has_air_conditioning, 
      automatic_car, has_getaround_connect, has_speed_regulator, winter_tires.
    
    Returns a list of predictions or an error message if something goes wrong.
    """
    if loaded_model is None:
        logger.error("Failed to load the ML model.")
        return {"error": "An error occurred while loading the model. Please try again later."}

    if preprocessor is None:
        logger.error("Failed to load the preprocessor.")
        return {"error": "An error occurred while loading the preprocessing step. Please try again later."}

    try:
        # Log received input
        logger.info(f"Received input: {input_data}")
        logger.info(f"Input Data length: {len(input_data)}")

        # Convert input data to DataFrame
        input_df = pd.DataFrame([item.dict() for item in input_data])  # Convert the List[InputData] to DataFrame

        # Validate that all required columns exist
        missing_columns = [col for col in expected_columns if col not in input_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")  # Log the error
            return {"error": f"Missing required columns: {missing_columns}. Please check your input data."}  # # Return the error response to the user

        # Log the received DataFrame
        logger.info(f"Input DataFrame shape: {input_df.shape}")
        logger.info(f"Input DataFrame columns: {list(input_df.columns)}")

        # Loop over categorical features to see if we need to adjust the encoder
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if isinstance(transformer, Pipeline):
                # If the transformer is a pipeline
                for step_name, step in transformer.steps:
                    if isinstance(step, OneHotEncoder):
                        # Check if we need to adjust the drop parameter
                        for feature in columns:
                            if input_df[feature].nunique() == 1:  # Check if only one category exists
                                # Update the encoder to not drop any category
                                step.drop = None  # Disable 'drop="first"' for this column
                        break  # We only have one OneHotEncoder, so break after adjusting it

        # Apply preprocessing (Use transform() instead of fit_transform() because model is already trained)
        try:
            preprocessed_data = preprocessor.transform(input_df)
            preprocessed_df = pd.DataFrame(preprocessed_data)
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return {"error": "An error occurred while preprocessing the input data. Please check the input format."}

        # Log preprocessed data
        logger.info(f"Preprocessed DataFrame shape: {preprocessed_df.shape}")

        # Make predictions
        try:
            predictions = loaded_model.predict(preprocessed_df)
            return {"predictions": predictions.tolist()}
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": "An error occurred while making the prediction. Please check your input and try again."}

    except json.JSONDecodeError:
        logger.error("Invalid JSON format in the request body.")
        return {"error": "Invalid JSON format in the request body. Please ensure the payload is a valid JSON."}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "An unexpected error occurred. Please try again later."}

# Run API
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Run on port 8080
