# Diabetes_prediction# Diabetes Prediction

This project predicts the likelihood of diabetes using the **Pima Indians Diabetes dataset**. It uses a **Logistic Regression model** to classify whether a patient has diabetes or not based on diagnostic measurements.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

Diabetes prediction is a binary classification problem where we predict whether a patient has diabetes based on diagnostic measurements. In this project, we:
- Clean and preprocess the dataset.
- Handle imbalanced data using techniques like **SMOTE**.
- Train a **Logistic Regression model** to predict diabetes.
- Evaluate the model's performance using accuracy, precision, recall, and F1-score.

---

## Dataset

The dataset used is the [Pima Indians Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database), which contains:
- **768 records** of female patients of Pima Indian heritage.
- **8 features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target**: `Outcome` (1 for diabetes, 0 for no diabetes).

---

## Project Structure

The project is organized as follows:

diabetes_prediction/
│
├── data/
│ └── diabetes.csv # Raw dataset
├── models/
│ └── diabetes_model.pkl # Trained Logistic Regression model
│ └── scaler.pkl # StandardScaler for preprocessing
├── scripts/
│ └── diabetes_prediction.py # Python script for the project
├── images/ # Visualizations (optional)
│ └── feature_distribution.png
├── README.md # Project documentation
└── requirements.txt # List of dependencies

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/benny-daniel6/Diabetes_prediction.git
   cd diabetes_prediction
2. Install the required dependencies:
   pip install -r requirements.txt
3. Download the dataset:
   Download the Pima Indians Diabetes dataset.
   Place the diabetes.csv file in the data/ folder.
## Usage
Running the Script
To train the model and evaluate its performance, run the Python script:

bash:
python scripts/diabetes_prediction.py

What the Script Does:
1. Loads the dataset from data/diabetes.csv.
2. Preprocesses the data:
Handles missing values (if any).
Standardizes the features using StandardScaler.
3. Handles imbalanced data using SMOTE (Synthetic Minority Oversampling Technique).
4. Trains a Logistic Regression model on the training set.
5. Evaluates the model on the test set and prints:
Accuracy
Classification report (precision, recall, F1-score)
Confusion matrix
6. Saves the trained model and scaler to the models/ folder.

## Results
Model Performance:
Accuracy: The model achieves an accuracy of ~79% on the test set.
Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.77      0.78       149
           1       0.78      0.81      0.79       151

    accuracy                           0.79       300
   macro avg       0.79      0.79      0.79       300
weighted avg       0.79      0.79      0.79       300

Confusion Matrix:
 [[114  35]
 [ 29 122]]

 ## Contributing
Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE] file for details.