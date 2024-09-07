# UK-House-SalePrice-Predictive-Modelling-Using-ML

**Business Question**
The UK housing market plays a critical role in the economy, impacting individuals, investors, mortgage lenders, and policymakers. Accurate house price predictions are essential for decision-making in real estate investments, pricing strategies, and market trend analyses. This project aims to forecast UK house prices using machine learning models based on historical data, including property characteristics and location factors.

**Business Goal**
The primary goal is to develop a predictive model that forecasts UK house prices accurately. The project involves:

**Data cleaning and analysis of historical housing data**
Feature selection
Applying machine learning models
Evaluating model performance
Identifying the most influential factors affecting house prices
The ultimate objective is to deliver an accurate and interpretable model, enabling stakeholders to make informed decisions about property investments and market trends.

**Dataset Description**

The dataset contains attributes describing various properties, including physical characteristics, location, and sale details. The training dataset includes the target variable (SalePrice), while the test dataset omits this column. Key variables include:

MSSubClass: Type of dwelling (e.g., 1-Story, Duplex)
MSZoning: General zoning classification (e.g., Residential, Commercial)
LotFrontage: Linear feet of street connected to the property
OverallQual: Material and finish quality (rated 1-10)
GrLivArea: Above-grade living area square feet
SaleType: Type of sale (e.g., Warranty Deed, Cash)
SaleCondition: Condition of the sale (e.g., Normal, Abnormal)
Exploratory Data Analysis (EDA)
Dataset Overview
Training dataset: 1460 rows, 81 columns (including SalePrice)
Test dataset: 1459 rows, 80 columns (excluding SalePrice)
Key Steps:
Identified numerical and categorical features
Detected missing values in both datasets
Visualized the distribution of target and independent variables
Created a correlation heatmap between numerical variables and SalePrice
Insights from EDA:
Missing values need to be imputed or dropped.
Categorical variables should be encoded.
Numerical variables are closely correlated with SalePrice, influencing model selection.
Model Selection
Based on the EDA findings, two predictive models were selected for their ability to handle missing data and categorical variables efficiently:

1. CatBoost
CatBoost is a gradient boosting algorithm that automatically handles categorical features and missing values. Its advanced techniques like gradient and ordered boosting help improve predictive performance and prevent overfitting. It is memory efficient and provides built-in tools for feature importance and model interpretation.

2. TensorFlow Decision Forests (TFDF)
TFDF offers robust decision tree-based models, such as random forests, which efficiently handle large datasets and complex feature interactions. TFDF's flexibility and internal support for preprocessing make it a powerful tool for handling missing values and categorical features. It is optimized for large-scale datasets.

Model Training and Evaluation
Both models were trained and evaluated on validation and test sets. Key steps included:

Data pre-processing (handling missing values, encoding categorical variables)
Training the models using CatBoost and TFDF
Analyzing feature importance for both models
Comparing predictions and feature importance across both models
Conclusion and Recommendations
CatBoost performed well due to its ability to handle categorical variables and missing data natively.
TFDF showed strong scalability and performance with large datasets, providing reliable predictions.
Based on the feature importance analysis, stakeholders can gain insights into the factors most affecting house prices and make data-driven decisions in the UK real estate market.
