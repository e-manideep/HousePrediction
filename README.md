# House Price Prediction Machine Learning Project

## Overview

This project demonstrates an end-to-end machine learning pipeline for predicting house prices using the California Housing Dataset. The solution includes data preprocessing, exploratory data analysis, model training with hyperparameter tuning, evaluation, and deployment via a FastAPI REST API.

## Project Structure

- **train_model.py**: Loads data, performs EDA, preprocesses features, trains and tunes a Random Forest Regressor, evaluates the model, and saves both the model and scaler.
- **app.py**: Implements a FastAPI application with an endpoint `/predict` that accepts JSON-formatted input and returns a predicted house price.
- **Dockerfile**: (Optional) Containerizes the FastAPI app for easy deployment.
- **requirements.txt**: Lists the required Python packages.
- **correlation_heatmap.png**: Saved correlation heatmap image from EDA.

## How to Run the Project

### 1. Train the Model
Run the training script to generate the model and scaler files:
```bash
python train_model.py
```

### 2. Run the API Locally
Once the model is trained, start the FastAPI application:
```bash
uvicorn app:app --reload
```
Access the interactive API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Docker Deployment (Optional)
Build and run the Docker container:
```bash
docker build -t house-price-api .
docker run -d -p 8000:8000 house-price-api
```

## Approach and Decisions

- **Data Preprocessing & EDA**: The dataset is first explored to understand feature distributions and relationships. A correlation heatmap is generated for visual insight.
- **Model Selection & Optimization**: A Random Forest Regressor was selected due to its robustness and ability to model non-linear relationships. GridSearchCV was used to optimize hyperparameters such as `n_estimators`, `max_depth`, and `min_samples_split`.
- **Deployment**: FastAPI was chosen for its performance and simplicity. The API exposes a `/predict` endpoint that scales input features before passing them to the model.
- **MLOps Practices**: The project includes saving the model and scaler, containerization with Docker, and clear documentation for reproducibility.

## Conclusion

This project demonstrates key skills in data preprocessing, model training, hyperparameter optimization, API deployment, and basic MLOps practices. The code is modular and well-documented, making it easy to extend or integrate into larger systems.
