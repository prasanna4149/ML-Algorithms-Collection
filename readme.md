# Housing Price Prediction: Simple Linear Regression

## Overview

This project focuses on predicting house prices using **Simple Linear Regression**. It involves data preprocessing, exploratory data analysis (EDA), feature scaling, and building a regression model to estimate housing prices based on given features. The goal is to showcase a step-by-step approach for effective predictive modeling.

---

## Dataset

The dataset (`Housing.csv`) contains information about houses, including the following features:

- **Area**: Size of the house (in square feet).
- **Bedrooms**: Number of bedrooms.
- **Bathrooms**: Number of bathrooms.
- **Stories**: Number of stories.
- **Parking**: Number of parking spaces.
- **Mainroad, Guestroom, Basement, Hotwaterheating, Airconditioning, Prefarea**: Binary indicators (Yes/No) converted to numerical values.
- **Furnishingstatus**: Indicates the furnishing status (fully, semi, unfurnished).
- **Price**: Target variable representing the price of the house.

The dataset undergoes preprocessing, including handling categorical variables, scaling numerical features, and removing missing values, to ensure robust model training.

---

## Steps and Methodology

### 1. **Data Cleaning and Preprocessing**
- Converted categorical features into numerical values using `LabelEncoder`.
- Checked for and handled missing values.
- Scaled numerical features using `StandardScaler` to improve model performance.

### 2. **Exploratory Data Analysis (EDA)**
- Analyzed the dataset's structure and statistical properties.
- Visualized the correlation between numeric features using a heatmap.

### 3. **Model Development**
- Split the dataset into training and testing sets (80/20 ratio).
- Built a **Simple Linear Regression** model using `LinearRegression` from `scikit-learn`.

### 4. **Model Evaluation**
- Evaluated model performance using:
  - **Mean Squared Error (MSE)**
  - **R² Score**
- Visualized the relationship between actual and predicted prices using scatter plots.

### 5. **Model Coefficients**
- Extracted the model’s coefficients and intercept to understand the relationship between features and target.

---

## Results and Evaluation

- **Mean Squared Error (MSE):** Quantifies the average squared difference between actual and predicted prices.
- **R² Score:** Measures the proportion of variance in the target variable explained by the model.
- **Visualization:** A scatter plot shows the alignment of predictions with actual values, with a red line representing perfect prediction.

---

## Dependencies

To run this project, ensure the following libraries are installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install these dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
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
