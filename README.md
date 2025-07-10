# House Price Prediction

This project uses supervised machine learning to predict house prices based on various features such as number of bedrooms, bathrooms, square footage, location, and more. It demonstrates a full end-to-end data science workflow — from raw data exploration to model evaluation — using a real-world housing dataset.

## What this project includes

- Exploratory Data Analysis (EDA) to identify trends, distributions, and outliers
- Feature engineering using combinations and transformations (e.g., age from `yr_built`)
- Handling of categorical and numerical data
- Model training using regression algorithms such as Linear Regression and Random Forest
- Performance evaluation using RMSE, MAE, and R² Score
- Interpretation of feature importance to understand key pricing factors

## Technologies used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn for modeling and evaluation
- Jupyter Notebook for development

## What you will learn

- How to structure and clean a real-world dataset
- The importance of preprocessing (scaling, encoding, feature selection)
- When to apply regression models and how to evaluate them
- Visualizing relationships between features and price
- Translating domain understanding into feature engineering

## Dataset Description

The dataset consists of 21,613 records and 21 columns, including:

- **Target**: `price` — the sale price of the house
- **Key features**:  
  `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`,  
  `waterfront`, `view`, `condition`, `grade`,  
  `sqft_above`, `sqft_basement`, `zipcode`, `lat`, `long`, `yr_built`, etc.

There are no missing values in the dataset.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/nikhilmahapatro/House_Price_Prediction.git
   cd House_Price_Prediction
