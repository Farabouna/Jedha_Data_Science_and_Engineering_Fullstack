import os
import mlflow
import joblib
import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
pricing_df = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv')

# Drop the unnamed first column from pricing_df
pricing_df = pricing_df.drop(pricing_df.columns[0], axis=1)

# Split dataset into X features and Target variable
target_name = 'rental_price_per_day'
Y = pricing_df.loc[:, target_name]
X = pricing_df.drop(target_name, axis=1)

# Split our training set and our test set 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create pipeline for numeric features
numeric_features = ['mileage', 'engine_power']  
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Create pipeline for categorical features
categorical_features = ['model_key', 'fuel', 'paint_color', 'car_type', 'private_parking_available', 'has_gps', 'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
#categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])

# Get combined unique categories from train and test and sort numerical categories
all_categories = {}
for feature in categorical_features:
    combined_unique = pd.concat([X_train[feature], X_test[feature]]).unique()
    
    # Sort if the feature is numerical
    if pd.api.types.is_numeric_dtype(combined_unique):
        combined_unique = np.sort(combined_unique)  
    
    all_categories[feature] = combined_unique.tolist()

# Create OneHotEncoder with predefined categories
categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(
        categories=[all_categories[feature] for feature in categorical_features],
        drop="first", handle_unknown="ignore", sparse_output=False 
    ))
])

# Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# Preprocessings on train set
print("Performing preprocessings on train set...")
X_train = preprocessor.fit_transform(X_train)
print("...Done.")

# Preprocessings on test set
print("Performing preprocessings on test set...")
X_test = preprocessor.transform(X_test) 
print("...Done.")

# Set variables for environment
EXPERIMENT_NAME="Pricing_prediction_experiment"

# Set tracking URI to Huggingface application
mlflow.set_tracking_uri("https://farabouna-GetAroundPricing.hf.space/")

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Call mlflow autolog
mlflow.sklearn.autolog()

with mlflow.start_run(experiment_id = experiment.experiment_id):
    
    # Convert sparse matrix to dataframe
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    
    # Train model
    model = LinearRegression()
    print("Training model...")
    model.fit(X_train_df, Y_train)
    print("...Done.")

    # Predictions on training set
    print("Predictions on training set...")
    Y_train_pred = model.predict(X_train_df)
    print("...Done.")

    # Predictions on test set
    print("Predictions on test set...")
    Y_test_pred = model.predict(X_test_df)
    print("...Done.")

    # Print scores using appropriate metrics for regression
    R2_score_train = r2_score(Y_train, Y_train_pred)
    R2_score_test = r2_score(Y_test, Y_test_pred)

    mse_train = mean_squared_error(Y_train, Y_train_pred)
    mse_test = mean_squared_error(Y_test, Y_test_pred)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    print("R2 score on training set: {}".format(R2_score_train))
    print("R2 score on test set: {}".format(R2_score_test))
    print("MSE on training set: {}".format(mse_train))
    print("MSE on test set: {}".format(mse_test))
    print("RMSE on training set: {}".format(rmse_train))
    print("RMSE on test set: {}".format(rmse_test))

    # Log Metric 
    mlflow.log_metric("R2 score on training set", R2_score_train)
    mlflow.log_metric("R2 score on test set", R2_score_test)
    mlflow.log_metric("MSE on training set", mse_train)
    mlflow.log_metric("MSE on test set", mse_test)
    mlflow.log_metric("RMSE on training set", rmse_train)
    mlflow.log_metric("RMSE on test set", rmse_test)
    
    # Save preprocessor inside "model" directory
    os.makedirs("model", exist_ok=True)  # Ensure "model" directory exists
    preprocessor_path = "model/preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)

    # Log Model & Preprocessor in the same directory
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact(preprocessor_path, artifact_path="model")