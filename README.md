# House Price Prediction Machine Learning Project

## Overview

This project demonstrates an end-to-end machine learning pipeline for predicting house prices using the California Housing Dataset. The solution includes data preprocessing, exploratory data analysis, model training with hyperparameter tuning, evaluation, and deployment via a FastAPI REST API.

The API is containerized using Docker and is hosted on **Render** for public access.

## Project Structure

- **train_model.py**: Loads data, performs EDA, preprocesses features, trains and tunes a Random Forest Regressor, evaluates the model, and saves both the model and scaler.
- **app.py**: Implements a FastAPI application with an endpoint `/predict` that accepts JSON-formatted input and returns a predicted house price.
- **Dockerfile**: Containerizes the FastAPI app for easy deployment.
- **requirements.txt**: Lists the required Python packages.
- **correlation_heatmap.png**: Saved correlation heatmap image from EDA.

## Model Artifacts

- **Pre-trained Model and Scaler**:  
  The trained model (`model.pkl`) and scaler (`scaler.pkl`) files are available for quick evaluation.  
  **Download Link**: [Google Drive Model Artifacts]([https://drive.google.com/placeholder_link](https://drive.google.com/file/d/19cVR5WxiP_DWueXdb6A6GBDIOfEG4h77/view?usp=sharing)) 

## How to Run the Project

### 1. Train the Model (Optional)
If you want to retrain the model, run:
```bash
python train_model.py
```
This script will preprocess the data, train the model, tune hyperparameters, and save the model and scaler.

### 2. Run the API Locally
Ensure the `model.pkl` and `scaler.pkl` files are available. Then, start the FastAPI application:
```bash
uvicorn app:app --reload
```
Access the interactive API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Docker Deployment
To deploy the API using Docker:

**Build the Docker Image:**
```bash
docker build -t house-price-api .
```

**Run the Docker Container:**
```bash
docker run -d -p 8000:8000 house-price-api
```
Access the API at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Hosted on Render
The API is deployed on Render for public access.  
**Live API URL**: [Update this with your Render deployment link]

## Approach and Decisions

### Data Preprocessing & EDA

#### Exploratory Data Analysis (EDA):
- Visualized feature distributions and correlations.
- Generated a correlation heatmap (`correlation_heatmap.png`) to analyze feature-target relationships.

#### Preprocessing Steps:
- Handled missing values.
- Applied feature scaling to normalize numeric features.
- Encoded categorical variables.
- Selected the most relevant features for training.

### Model Selection & Optimization

#### Model Choice:
- A Random Forest Regressor was selected due to its robustness and ability to model non-linear relationships.

#### Hyperparameter Tuning:
- Used `GridSearchCV` to optimize `n_estimators`, `max_depth`, and `min_samples_split` for the best model performance.

#### Evaluation Metrics:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score (Coefficient of Determination)**

### Deployment Strategy

#### API Development:
- FastAPI was chosen for its lightweight, high-performance, and easy-to-use structure.
- The `/predict` endpoint accepts JSON-formatted input, applies feature scaling, and returns the predicted house price.

#### MLOps Practices:
- Model and scaler are saved as `.pkl` files for consistent inference.
- Docker containerization is used for cross-platform deployment.
- Logging and error handling ensure robustness in production.

## Additional Enhancements

### Logging & Error Handling:
- Integrated logging for tracking API usage and debugging.
- Proper exception handling to prevent errors during requests.

### Cloud Deployment & Model Versioning:
- Hosted on Render for easy access.
- Future versions can integrate MLflow or DVC for model tracking.

### Frontend Integration:
- A simple UI was added for users to interact with the API.

## API Usage Guide

### **Endpoint:** `/predict`
**Method:** `POST`

#### Input Format:
```json
{
  features.MedInc,
  features.HouseAge,
  features.AveRooms,
  features.AveBedrms,
  features.Population,
  features.AveOccup,
  features.Latitude,
  features.Longitude
}
```
(Refer to the API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for detailed parameters.)

#### Output:
```json
{
  "predicted_price": 450000.75
}
```

## Conclusion

This project showcases key machine learning engineering skills, including:
- **Data preprocessing and feature engineering**
- **Model training and hyperparameter tuning**
- **Deployment via FastAPI with Docker**
- **Hosting on Render for public accessibility**
- **MLOps best practices for reproducibility**

With model artifacts available via Google Drive and a hosted API on Render, this project is ready for quick evaluation and real-world integration.

