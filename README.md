# Disease Prediction Using Machine Learning

This project demonstrates a machine learning-based disease prediction system that uses patient symptoms to predict the most probable disease. The model is trained on a medical dataset of various symptoms and diseases and utilizes three classifiers: Support Vector Classifier (SVC), Naive Bayes, and Random Forest. The final prediction is made by taking the mode of all three classifiers' predictions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Overview

The goal of this project is to predict diseases based on symptoms provided by the user. We have used three different machine learning models (SVM, Naive Bayes, Random Forest) and combined their predictions to improve accuracy. The model has been trained and tested on a labeled dataset, where each disease is represented by a combination of symptoms.

## Dataset

The dataset used for training and testing contains a list of symptoms and the corresponding prognosis (disease). You can download the dataset from Kaggle: [Disease Prediction Dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning).

The dataset has the following structure:
- **Columns**: Symptoms (as binary values: `1` for presence and `0` for absence).
- **Target Label**: The disease (prognosis) corresponding to the symptoms.

## Models Used

We have used the following models for disease prediction:
1. **Support Vector Classifier (SVC)**
2. **Naive Bayes Classifier**
3. **Random Forest Classifier**

The final prediction is based on the mode of the predictions from these three models.

## Usage

1. Run the Jupyter notebook or Python script to train the models and make predictions. The dataset is split into training and testing sets.
2. To make predictions based on user input, use the `predictDisease` function. You can pass symptoms in a comma-separated string.

    Example:
    ```python
    prediction = predictDisease("Skin Peeling, Red Sore Around Nose, Drying And Tingling Lips")
    print(f"Predicted Disease: {prediction['final_prediction']}")
    ```

## Results

The combined model's performance is evaluated using a test dataset. Below are some of the key metrics:
- **Accuracy** on training and test sets for each model.
- **Confusion Matrix** for each model.

After training, the final model gives a combined prediction based on the individual models' outputs.

## Acknowledgements

- Dataset provided by Kaggle: [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning).
- This project uses Scikit-Learn, Seaborn, Pandas, and Matplotlib.

