# Stock Prices Prediction Analysis

This project analyzes and predicts stock prices using various machine learning algorithms implemented in a Jupyter Notebook. The dataset used is `World-Stock-Prices-Dataset.csv`. Below is an overview of the steps and methodologies used in the analysis.

## Dataset
The dataset contains historical stock prices and associated metadata. Features include:
- `Country`: The country where the stock is listed.
- `Ticker`: Stock ticker symbol.
- `Industry_Tag`: The sector or industry classification.
- `Brand_Name`: The company's brand.
- `Close`: The closing price of the stock.

## Requirements
Ensure you have the following Python libraries installed:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Install missing libraries using pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Steps in the Analysis

### 1. Data Loading and Exploration
- Load the dataset using `pandas`.
- Inspect the dataset with `.info()` and `.describe()`.
- Check for unique values in columns such as `Country`, `Ticker`, and `Industry_Tag`.

### 2. Data Cleaning
- Handle missing values by imputing them with the mean.
- Drop unnecessary columns (e.g., `Date`).
- Encode categorical columns (`Brand_Name`, `Ticker`, `Industry_Tag`, `Country`) using `LabelEncoder`.

### 3. Feature Selection and Splitting
- Define `X` (features) and `y` (target: `Close`).
- Split data into training and testing sets using an 80-20 split with `train_test_split`.

### 4. Algorithms Used for Prediction

#### 4.1 Linear Regression
- Handle missing values using `SimpleImputer`.
- Train the `LinearRegression` model.
- Evaluate using MAE, MSE, and RMSE.

#### 4.2 Decision Tree Regressor
- Train the `DecisionTreeRegressor` model with specified hyperparameters.
- Evaluate using MSE and RMSE.

#### 4.3 K-Nearest Neighbors (KNN)
- Train the `KNeighborsRegressor` model with `n_neighbors=5`.
- Evaluate using MSE and RMSE.

#### 4.4 Ridge Regression
- Train the `Ridge` regression model with `alpha=1.0`.
- Evaluate using MSE and RMSE.

#### 4.5 Lasso Regression
- Train the `Lasso` regression model with `alpha=1.0`.
- Evaluate using MSE and RMSE.

#### 4.6 Polynomial Regression
- Transform features using `PolynomialFeatures` with `degree=2`.
- Train a `LinearRegression` model on transformed features.
- Evaluate using MSE and RMSE.

#### 4.7 Bayesian Ridge Regression
- Train the `BayesianRidge` regression model.
- Evaluate using MSE and RMSE.

### 5. Visualizations
- Compare actual vs predicted stock prices for each model using scatter plots.
- Add gridlines and proper labels for clarity.

## Results
Metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are used to evaluate the performance of each model. Visualizations further help assess the accuracy of predictions.

## File Structure
- **`World-Stock-Prices-Dataset.csv`**: Input dataset.
- **`stock_prices_analysis.ipynb`**: Jupyter Notebook containing the full analysis and code.

## How to Run
1. Clone the repository.
2. Place the dataset (`World-Stock-Prices-Dataset.csv`) in the same directory as the notebook.
3. Open the Jupyter Notebook and run all cells sequentially.

## Dependencies
- Python 3.7+
- Jupyter Notebook


## Acknowledgements

Special thanks to the creators of the dataset and open-source libraries used in this analysis.

---

Feel free to contribute or raise issues to improve this project!

## Author
Prasanna Patil
