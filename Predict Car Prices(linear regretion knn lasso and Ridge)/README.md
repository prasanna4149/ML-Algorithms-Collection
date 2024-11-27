
# Car Price Prediction: Linear Regression, KNN, Lasso, and Ridge

## Overview

This project focuses on predicting car prices using multiple machine learning models, including **Linear Regression**, **K-Nearest Neighbors (KNN)**, **Lasso Regression**, and **Ridge Regression**. The aim is to evaluate and compare these techniques to determine the most effective approach for predicting car prices based on various features.

## Dataset

The dataset used in this project contains information about cars, including features like:
- **Make/Model**: The manufacturer and model of the car.
- **Year**: Manufacturing year.
- **Engine Size**: Engine capacity in liters.
- **Mileage**: Total distance covered by the car.
- **Fuel Type**: Type of fuel used (e.g., petrol, diesel).
- **Transmission**: Gear type (manual or automatic).
- **Price**: Target variable representing the price of the car.

The data undergoes preprocessing steps such as handling missing values, feature encoding, and normalization before being used for model training.

## Models and Techniques

1. **Linear Regression**  
   A simple and interpretable baseline model to establish a linear relationship between features and the target variable.

2. **K-Nearest Neighbors (KNN)**  
   A non-parametric model that predicts prices based on the similarity between cars.

3. **Lasso Regression**  
   Adds L1 regularization to linear regression, promoting feature selection by penalizing less significant coefficients.

4. **Ridge Regression**  
   Similar to Lasso but employs L2 regularization, reducing the impact of less significant coefficients without eliminating them.

## Evaluation

The models are evaluated using metrics such as:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

These metrics provide insights into the models' accuracy and ability to generalize to unseen data.

## Key Findings

- **Performance Comparison**: Results highlight which model performs best under different conditions.
- **Regularization Effects**: Insights into how Lasso and Ridge improve model performance by handling multicollinearity.

## Requirements

To replicate this analysis, you need the following dependencies:
- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Install the required packages using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Notebook

1. Clone this repository or download the notebook file.
2. Ensure the dataset file is placed in the same directory as the notebook.
3. Launch the notebook using Jupyter:
   ```bash
   jupyter notebook "car price with linear regretion knn lasso and Ridge.ipynb"
   ```
4. Execute the cells sequentially to preprocess data, train models, and evaluate results.

## Results and Insights

- The optimal model and its hyperparameters.
- The importance of feature selection and preprocessing in predictive modeling.
- Recommendations for future work, such as testing additional algorithms or refining the dataset.

## Acknowledgements

Special thanks to the creators of the dataset and open-source libraries used in this analysis.

---

Feel free to contribute or raise issues to improve this project!
