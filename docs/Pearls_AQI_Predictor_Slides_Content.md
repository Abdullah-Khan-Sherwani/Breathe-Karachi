# Pearls AQI Predictor

**Goal:** Predict the Air Quality Index (AQI) in your city in the next 3 days using a 100% serverless stack.

---

## High-Level Overview

To achieve this Air Quality Index (AQI) prediction service, the system is broken down into the following high-level components:
1. **Raw Data:** Fetching weather and pollution data from APIs.
2. **Feature Generation:** Processing raw data into features and storing them in a Feature Store.
3. **Model Training:** Training models on features and saving them to a Model Registry.
4. **Web App (Forecast):** Retrieving features and models to serve predictions to the user.

*(The entire process is orchestrated and automated via CI/CD tools like GitHub Actions).*

---

## 1. Feature Pipeline

Write a Python script that handles the data ingestion and processing:

* **Fetch Data:** Retrieves raw weather and pollutant data from an external API (e.g., AQICN or OpenWeather).
* **Compute Features:** Transforms raw data into model inputs (features) and outputs (targets).
  * *Required features:* Time-based features (hour, day, month) and derived features (like AQI change rate).
* **Store Features:** Saves these processed features into a Feature Store.
  * *Suggested Platforms:* Hopsworks or Vertex AI (Free tiers available).

---

## 2. Backfill Historical Data

To build a robust training dataset:
* Run the feature generation script for a range of past dates.
* This process extracts historical raw data and computes the necessary historical features and targets to train your machine learning models.

---

## 3. Training Pipeline

Write a Python script dedicated to model training:

* **Fetch Data:** Retrieves historical features and targets from the Feature Store.
* **Train & Evaluate:** Trains and evaluates the best ML model possible for the data.
  * *Models to Explore:* Scikit-learn models (Random Forest, Ridge Regression) and advanced frameworks like TensorFlow or PyTorch.
  * *Evaluation Metrics:* Assess performance using RMSE, MAE, and R².
* **Store Model:** Saves the best trained model into the Model Registry.

---

## 4. Automate Pipeline Runs (CI/CD)

Create a CI/CD pipeline to fully automate the system:
* **Feature Script:** Schedule to run every hour.
* **Training Script:** Schedule to run every day.
* *Suggested Tools:* Apache Airflow, GitHub Actions, or other preferred CI/CD platforms.

---

## 5. Web App Dashboard

Develop a web application to showcase your model's predictions:

* **Load Assets:** Retrieves the trained model and the latest features directly from the Feature Store/Model Registry.
* **Compute Predictions:** Generates forecasts for the next 3 days.
* **Display Data:** Presents the results on a simple, descriptive, and interactive dashboard.
* *Suggested Tech Stack:* Streamlit or Gradio for the frontend; Flask or FastAPI for the backend logic.

---

## Guidelines & Best Practices

* **Exploratory Data Analysis (EDA):** Perform EDA to identify underlying data trends.
* **Model Diversity:** Utilize a variety of forecasting models, ranging from traditional statistical modeling to deep learning.
* **Explainability:** Use SHAP or LIME to provide feature importance explanations.
* **Alerts:** Implement a notification/alert system for hazardous AQI levels.

---

## Final Submission Requirements

Your final submission must include:
1. An end-to-end AQI prediction system.
2. A scalable, automated pipeline.
3. An interactive dashboard showcasing real-time and forecasted AQI data.
4. A detailed report documenting everything you managed to achieve.
