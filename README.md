# UK-House-SalePrice-Predictive-Modelling-Using-ML-Python
Predicting UK houses saling price using ML Model in Python Google Colab

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

**Exploratory Data Analysis (EDA)**

**Dataset Overview**

Training dataset: 1460 rows, 81 columns (including SalePrice)
Test dataset: 1459 rows, 80 columns (excluding SalePrice)

**Key Steps for analysis:**

- Identified numerical and categorical features
- Detected missing values in both datasets
- Visualized the distribution of target and independent variables
- Created a correlation heatmap between numerical variables and SalePrice

Model Training and Evaluation comparison:

- Both models were trained and evaluated on validation and test sets. Key steps included:
- Data pre-processing (Splitting dataset, handling target and independent variables etc)
- Training the models using CatBoost and TFDF
- Analyzing feature importance for both models
- Comparing predictions and feature importance across both models

1. Importing Libraries & the Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
!pip install catboost
!pip install tensorflow_decision_forests
from catboost import CatBoostRegressor
import tensorflow_decision_forests as tfdf
import tensorflow as tf

#Loading dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```
2. Exploratory Data Analysis
```python
#Checking dataset 
print("Full dataset shape is {}".format(train_data.shape))
print("Full dataset shape is {}".format(test_data.shape))

train_data.head()
train_data.info()
test_data.info()

#descriptive analysis
print(train_data.describe())
# Identify the numarical & categorical columns
numerical_columns = train_data.select_dtypes(include=['number']).columns
categorical_columns = train_data.select_dtypes(include=['object']).columns
print(numerical_columns)
print(categorical_columns)
# Overview of Missing Values for train_data
missing_values = train_data.isnull().sum()

# Filter columns with missing values
missing_values = missing_values[missing_values > 0]

# Calculate the percentage of missing values
missing_percentage = (missing_values / len(train_data)) * 100

# Combine the counts and percentages into a DataFrame
missing_data_overview = pd.DataFrame({'Missing Values': missing_values,
                                      'Percentage': missing_percentage})

# Display the result
print(missing_data_overview)

# Overview of Missing Values for test_data
missing_values = test_data.isnull().sum()

# Filter columns with missing values
missing_values = missing_values[missing_values > 0]

# Calculate the percentage of missing values
missing_percentage = (missing_values / len(test_data)) * 100

# Combine the counts and percentages into a DataFrame
missing_data_overview = pd.DataFrame({'Missing Values': missing_values,
                                      'Percentage': missing_percentage})

# Display the result
print(missing_data_overview)
```
4. Visual Exploratory Data Analysis
```python   
# Sale price distribution by Plotting
import seaborn as sns
print(train_data['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(train_data['SalePrice'], color='b', bins=100, hist_kws={'alpha': 0.4});

# Analyze the distribution of each categorical feature
for column in categorical_columns:
    print(f"Distribution of {column}:")
    print(train_data[column].value_counts())
    print("\n" + "="*50 + "\n")

# Numerical data distribution by plotting

numerical_columns = train_data.select_dtypes(include=['number'])
numerical_columns.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()

# Co-relation Heatmap of Numerical features & Sale Price (Target Variable)
# Calculate the correlation matrix
correlation_matrix = numerical_columns.corr()

# Set up the matplotlib figure
plt.figure(figsize=(16, 12))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)

# Set the title
plt.title('Correlation Heatmap of Numerical Features and SalePrice', fontsize=16)

# Show the plot
plt.show()
```
**Insights from EDA:**

Missing values need to be imputed or dropped.
Categorical variables should be encoded.
Numerical variables are closely correlated with SalePrice, influencing model selection.

5. **Model Selection**
   
Based on the EDA findings, two predictive models were selected for their ability to handle missing data and categorical variables efficiently:

> CatBoost
CatBoost is a gradient boosting algorithm that automatically handles categorical features and missing values. Its advanced techniques like gradient and ordered boosting help improve predictive performance and prevent overfitting. It is memory efficient and provides built-in tools for feature importance and model interpretation.
```python  
#Splitting the independnet variables and target variable
label = 'SalePrice'
X = train_data.drop(columns=[label, 'Id'])
y = train_data[label]
X_test = test_data.drop(columns=['Id'])

#Processing Categrical Columns to Category Type for CatBoost Model
cat_features = X.select_dtypes(include=['object', 'category']).columns
X[cat_features] = X[cat_features].fillna('missing').astype('category')
X_test[cat_features] = X_test[cat_features].fillna('missing').astype('category')

#Splitting the dataset for CatBoost model training & evaluation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#Training & Fitting the CatBoost Regression Model
catboost_model = CatBoostRegressor(iterations=1000,
                                   learning_rate=0.05,
                                   depth=10,
                                   cat_features=list(cat_features),
                                   verbose=200)
catboost_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

# After training the model we predict the valdation and test sets respectively
catboost_val_preds = catboost_model.predict(X_valid)
catboost_test_preds = catboost_model.predict(X_test)
```
- R Squared Value and Feature Importance for CatBoost Model
```python  
# Calculate R-squared for CatBoost
catboost_r2 = r2_score(y_valid, catboost_val_preds)
print(f"CatBoost R-squared on validation set: {catboost_r2:.4f}")

# Extract and plot feature importances from CatBoost model
catboost_importances = catboost_model.get_feature_importance()
catboost_feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': catboost_importances})
filtered_feature_importance_df = catboost_feature_importance_df[catboost_feature_importance_df['Importance'] > 3]
filtered_feature_importance_df = filtered_feature_importance_df.sort_values(by='Importance', ascending=False)
print(filtered_feature_importance_df)

# Plot top features for CatBoost
top_features = filtered_feature_importance_df['Feature'].tolist()
plot_data = X_train[top_features].copy()
plot_data['SalePrice'] = y_train.values
num_features = len(top_features)
num_rows = (num_features + 2) // 3
num_cols = min(num_features, 3)

plt.figure(figsize=(15, num_rows * 5))
for idx, feature in enumerate(top_features):
    plt.subplot(num_rows, num_cols, idx + 1)
    plt.scatter(plot_data[feature], plot_data['SalePrice'], alpha=0.5)
    plt.title(f'Sale Price vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
plt.tight_layout()
plt.show()
```
- Final Predicted SalePrice(Output) from CatBoost Model
```python 
# Assuming `test_data` and `catboost_test_preds` are already defined
test_ids = test_data['Id']

# Add a dollar sign in front of SalePrice values, formatted as strings
catboost_output = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': ['$' + format(price, ',.2f') for price in catboost_test_preds]
})

# Save the output to a CSV file
catboost_output.to_csv('catboost_results.csv', index=False)

# Display the first few rows of the output to verify
print(catboost_output.head())
```

> TensorFlow Decision Forests (TFDF)

TFDF offers robust decision tree-based models, such as random forests, which efficiently handle large datasets and complex feature interactions. TFDF's flexibility and internal support for preprocessing make it a powerful tool for handling missing values and categorical features. It is optimized for large-scale datasets.

```python 
# Pre-processing dataset for TFDF Model
label = 'SalePrice'  # Replace with the name of your target variable if different

#Splitting the training data into train & valid sets
def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(train_data, test_ratio=0.20)
print(f"{len(train_ds_pd)} examples in training, {len(valid_ds_pd)} examples in validation.")

#Create Tensorflow dataframe
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

#Training the Tensorflow Random Forest Model & Model Evaluation
rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])
rf.fit(x=train_ds)

# Visualize the trained model
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

# Evaluate the model
inspector = rf.make_inspector()
val_evaluation = rf.evaluate(x=valid_ds, return_dict=True)
for name, value in val_evaluation.items():
    print(f"{name}: {value:.4f}")
    
# Get predictions from the Random Forest model
def get_predictions(model, dataset):
    predictions = []
    for features, _ in dataset:
        preds = model.predict(features)
        predictions.extend(preds)
    return np.array(predictions)

# Extract true values from the validation dataset
true_values = np.concatenate([labels for _, labels in valid_ds], axis=0)
predicted_values_tfdf = get_predictions(rf, valid_ds)     
```
- R Squared Value and Feature Importance for TF-DF Model
```python 
def compute_r_squared(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Assign the predicted values to y_pred
y_pred = predicted_values_tfdf

r_squared = compute_r_squared(true_values, y_pred)
print(f"Random Forest R-squared on validation set: {r_squared:.4f}")

# Get and print Feature importances TFDF Model
importances = inspector.variable_importances()
print(f"Available variable importances:")
for importance in importances.keys():
    print("\t", importance)

inspector.variable_importances()["NUM_AS_ROOT"]

# Plot the most important variables
plt.figure(figsize=(12, 4))

# Mean decrease in AUC of the class 1 vs the others.
variable_importance_metric = "NUM_AS_ROOT"
variable_importances = importances[variable_importance_metric]

# Extract the feature name and importance values.
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# Label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

plt.xlabel(variable_importance_metric)
plt.title("NUM AS ROOT of the class 1 vs the others")
plt.tight_layout()
plt.show()
```
- Final Predicted SalePrice(Output) from TFDF Model
```python 
# Ensure 'Id' column is not removed from test dataset
ids = test_data['Id']
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, task=tfdf.keras.Task.REGRESSION)
preds = rf.predict(test_ds)

output = pd.DataFrame({'Id': ids, 'SalePrice': preds.squeeze()})
output['SalePrice'] = output['SalePrice'].apply(lambda x: f"${x:,.2f}")
print(output.head())

# Save the results to a CSV file
output.to_csv('tfdf_results.csv', index=False)
```
6. Combined Visualization of TFDF and CatBoost Models' Imoprtant Variables/Features
```python 
# TFDF Variable Importance
importances = inspector.variable_importances()
variable_importance_metric = "NUM_AS_ROOT"
tfdf_importances = importances[variable_importance_metric]

# Extract the feature name and importance values for TFDF
tfdf_feature_names = [vi[0].name for vi in tfdf_importances]
tfdf_feature_importances = [vi[1] for vi in tfdf_importances]

# CatBoost Variable Importance
catboost_importances = catboost_model.get_feature_importance()
catboost_feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': catboost_importances})
catboost_feature_importance_df = catboost_feature_importance_df[catboost_feature_importance_df['Importance'] > 3]
catboost_feature_importance_df = catboost_feature_importance_df.sort_values(by='Importance', ascending=False)
```

> Combined Plotting
```python 
# Merge and align features from both models
common_features = set(tfdf_feature_names) & set(catboost_feature_importance_df['Feature'])
tfdf_sorted = [(name, importance) for name, importance in zip(tfdf_feature_names, tfdf_feature_importances) if name in common_features]
catboost_sorted = catboost_feature_importance_df[catboost_feature_importance_df['Feature'].isin(common_features)]

tfdf_sorted = sorted(tfdf_sorted, key=lambda x: -x[1])
catboost_sorted = catboost_sorted.sort_values(by='Importance', ascending=False)

tfdf_features, tfdf_importances = zip(*tfdf_sorted)
catboost_features = catboost_sorted['Feature']
catboost_importances = catboost_sorted['Importance']

# Create a DataFrame for plotting
comparison_df = pd.DataFrame({
    'Feature': list(tfdf_features),
    'TFDF Importance': list(tfdf_importances),
    'CatBoost Importance': [catboost_importances[catboost_features[catboost_features == f].index[0]] if f in catboost_features.values else 0 for f in tfdf_features]
})

# Sort the DataFrame by feature importance in ascending order
comparison_df = comparison_df.sort_values(by=['TFDF Importance', 'CatBoost Importance'], ascending=[False, False])

# Plot the feature importances side by side (horizontal bars)
fig, ax = plt.subplots(figsize=(12, 10))

bar_height = 0.2
index = np.arange(len(comparison_df))

bar1 = ax.barh(index - bar_height / 2, comparison_df['TFDF Importance'], bar_height, label='TFDF', color='skyblue')
bar2 = ax.barh(index + bar_height / 2, comparison_df['CatBoost Importance'], bar_height, label='CatBoost', color='lightgreen')

ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Feature Importance Comparison: TFDF vs CatBoost')
ax.set_yticks(index)
ax.set_yticklabels(comparison_df['Feature'])
ax.legend()

# Label each bar with its value
for bar in bar1 + bar2:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, round(bar.get_width(), 2), va='center')

plt.tight_layout()
plt.show()
```

**Findings**

Predicted SalePrice output from Tensorflow RandomForest Model and CatBoost model have been saved in the 'tfdf_results.csv' and 'catboost_results.csv' files.

From the above two model analysis we can observe, for predicting UK house sale prices, CatBoost demonstrates slightly better performance with an R-squared value of 89%, compared to 82% for TFDF. This indicates that CatBoost explains a higher proportion of the variance in house prices and provides more accurate predictions overall.

However, the feature importance comparison suggests that TFDF places greater emphasis on specific features like GarageCars and BsmtQual, which may be more relevant for certain properties. In contrast, CatBoost assigns more balanced importance across a broader range of features, including OverallQual and GrLivArea, making it a well-rounded choice.

**Conclusion and Recommendations**

CatBoost performed well due to its ability to handle categorical variables and missing data natively. TFDF showed strong scalability and performance with large datasets, providing reliable predictions. Based on the feature importance analysis, stakeholders can gain insights into the factors most affecting house prices and make data-driven decisions in the UK real estate market.

> Recommendations for Stakeholders

Model Selection:

CatBoost is the preferred model for predicting UK house prices due to its superior accuracy and effective feature balancing compared to the TensorFlow Random Forest model.

Investment and Improvement Focus:

Stakeholders should focus on enhancing overall property quality and living area, as both models identify these as crucial factors influencing property value. Additionally, improvements to garages and basements are recommended based on the TensorFlow model's sensitivity to these features.

Strategic Insights:

Investors can combine insights from both models to refine future sale price predictions, particularly for properties where features like GarageCars and BsmtQual are highly relevant.

(Note: Please check the Google Colb notebook for details explanation and graphs)

**Author: Debolina Dutta**

**LinkedIn:** https://www.linkedin.com/in/duttadebolina/
