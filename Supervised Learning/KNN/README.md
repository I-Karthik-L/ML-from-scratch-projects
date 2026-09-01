# K-Nearest Neighbors (KNN) Medical Data Classification

This repository contains a Python practice session focused on implementing the K-Nearest Neighbors (KNN) machine learning algorithm. It demonstrates how to preprocess tabular data, scale features, and evaluate model performance using two different real-world medical datasets.

## Project Overview
The notebook explores supervised learning for classification tasks. It starts with a fundamental toy example to explain how KNN works under the hood, and then scales up to apply the algorithm to predict medical outcomes based on patient metrics.

The notebook covers predictions for:
1.  **Diabetes:** Predicting the onset of diabetes based on diagnostic measures.
2.  **Lung Cancer:** Classifying lung cancer risk based on specific patient attributes.

## What This Code Does
The notebook walks through a standard Machine Learning classification pipeline using `scikit-learn` and `pandas`:

*   **Basic KNN Implementation:** A hardcoded toy dataset to demonstrate how the `KNeighborsClassifier` maps 2D points and predicts classes based on proximity.
*   **Data Splitting:** Uses `train_test_split` to divide the medical datasets into training (80%) and testing (20%) sets to ensure fair evaluation.
*   **Feature Scaling:** Implements `MinMaxScaler` to normalize the input features. This is a crucial step for distance-based algorithms like KNN, ensuring that features with larger numeric ranges don't dominate the distance calculations.
*   **Hyperparameter Tuning Setup:** Demonstrates initializing the model with different `n_neighbors` values (e.g., K=17 for Diabetes, K=5 for Lung Cancer).
*   **Model Evaluation:** Uses multiple metrics to assess model health, including:
    *   `accuracy_score`
    *   `ConfusionMatrixDisplay` (visualizing True Positives, False Positives, etc.)
    *   `classification_report` (Precision, Recall, and F1-Score)

## How to Run This Code (Google Colab)
This notebook is configured to run in **Google Colab**. To practice with this code yourself, follow these steps:

1. **Download the Datasets:** Download the file provided in this repository the `diabetes.csv` and `lung_cancer_examples.csv` files to your local computer.
2. **Open the Notebook:** Upload the notebook file to Google Colab and open it.
3. **Upload the Data to Colab:** Click the folder icon on the left sidebar of your Colab interface and upload the two `.csv` files directly into the `/content/` directory.
4. **Execute:** Run the cells sequentially. You can experiment by changing the `n_neighbors` value in the code to see how it affects the accuracy and confusion matrix!

## Dependencies
If you prefer to run this environment locally, you will need the following Python libraries installed:
*   `pandas`
*   `scikit-learn`
*   `matplotlib` (required for confusion matrix plotting)

You can install these via pip:
`pip install pandas scikit-learn matplotlib`
