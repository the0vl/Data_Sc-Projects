# Marketing Capstone Project
##### Using 3 Different Machine Learning Methods to Predict Click Through Rate (CTR)

# --- Importing required libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Setting the working directory to the folder containing the dataset
os.chdir('/Users/theovl/Downloads')

# --- Loading the data ---
df = pd.read_csv('data.csv')  # Main dataset
info = pd.read_csv('Data+description+-+sheet1.csv')  # Metadata about the dataset

# Display metadata to understand column descriptions
info

# Display first few rows of the dataset to understand structure
df.head()

# Examine class distribution in the target variable `click`
df.click.value_counts()

# Display basic info about the dataset, including column names and data types
df.info()

# List all column names for easier navigation
df.columns.values

# --- Data Cleaning ---

# Drop columns that are unlikely to influence CTR based on domain knowledge
df2 = df.drop(['dayofweek', 'day', 'hour'], axis=1)

# Check for null values in each column
df2.isnull().sum()

# Visualizing distribution of target variable `y` before dropping it
plt.figure(figsize=(12, 6))
sns.countplot(x='y', data=df2)

# Visualizing distribution of `click` (the actual target variable for modeling)
plt.figure(figsize=(12, 6))
sns.countplot(x='click', data=df2)

# Drop redundant outcome variable `y` as it overlaps with `click`
df2 = df2.drop('y', axis=1)

# Display updated column names after dropping unnecessary columns
df2.columns

# Visualize distribution of categorical variable `device_type`
plt.figure(figsize=(12, 6))
sns.countplot(y='device_type', data=df2)
plt.show()

# Convert the `click` target variable to binary (1 for True, 0 for False)
df2['click'] = df2.click.apply(lambda x: 1 if x == True else 0)

# Display the dataset after initial cleaning
df2

##### Converting Categorical Variables into Dummy Variables #####

# Explanation:
# Creating dummy variables for categorical predictors with a manageable number of unique categories
# Large unique counts in predictors may introduce noise and computational inefficiency.

# Identify predictors with too many unique values
l = []
for i in df2.columns:
    if len(df2[i].unique()) > 20:
        l += [i]

# Identify categorical predictors
object_type_data = df2.select_dtypes(include='object').columns

# Identify predictors to use for dummy variable conversion
predictors = df2.drop('click', axis=1).columns
to_dummy_variables = df2.drop(l, axis=1).columns

# Drop predictors with too many unique categories
df2 = df2.drop(l, axis=1)

# Check data types of remaining predictors
df2.dtypes

# Generate dummy variables for the selected predictors
df2 = pd.get_dummies(data=df2, columns=to_dummy_variables, drop_first=True)

# Rename `click_1` column to `click` for consistency
df2 = df2.rename({'click_1': 'click'}, axis=1)

# --- Splitting Data into Train and Test Sets ---

# Separate features (X) and target variable (y)
X = df2.drop('click', axis=1)
y = df2['click']

# Split the data into training and testing sets (70% train, 30% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Display the shapes of train and test datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# --- Logistic Regression Model (Model 1) ---

# Initialize logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Extract model coefficients
coefficients = model.coef_

# Use the testing data to make predictions
y_pred = model.predict(X_test)

# Evaluate model accuracy on test data
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# Display and plot the confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
metrics.plot_confusion_matrix(model, X_test, y_test)
plt.show()

# ROC-AUC Score and Plot
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
metrics.plot_roc_curve(model, X_test, y_test)
plt.show()

# --- Recursive Feature Elimination (RFE) for Feature Selection (Model 2) ---

from sklearn.feature_selection import RFE

# RFE to select top 15 features
rfe = RFE(model, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)

# Selected features
col = X_train.columns[rfe.support_]
print("Selected features:", col)

# Features excluded by RFE
print("Excluded features:", X_train.columns[~rfe.support_])

# Fit model with selected features and evaluate performance
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
res = logm2.fit()

# Summary of the fitted model
res.summary()

# Predict probabilities for the training data
y_train_pred = res.predict(X_train_sm)

# Create a DataFrame for evaluation
y_train_pred_final = pd.DataFrame({'click': y_train, 'click_Prob': y_train_pred})
y_train_pred_final['predicted'] = y_train_pred_final.click_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Confusion matrix and accuracy for training predictions
confusion = metrics.confusion_matrix(y_train_pred_final.click, y_train_pred_final.predicted)
print(confusion)
print(metrics.accuracy_score(y_train_pred_final.click, y_train_pred_final.predicted))

# --- Varying Feature Counts and Thresholds ---
# Optimizing the number of features and decision thresholds
# --- Finding the Optimal Number of Features ---

# Iteratively test models with different numbers of features using RFE
for i in range(10, 55):  # Testing between 10 and 54 features
    rfe = RFE(model, n_features_to_select=i)
    rfe = rfe.fit(X_train, y_train)
    
    # Extract selected features for the current iteration
    col = X_train.columns[rfe.support_]
    
    # Fit a logistic regression model using the selected features
    X_train_sm = sm.add_constant(X_train[col])
    logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
    res = logm2.fit()
    
    # Display the number of features and evaluate model performance
    print("\n\n")
    print(f"Number of features: {i}")
    model_evaluation(res, X_test, y_test, col)

# Based on the output, the optimal number of features is determined to be 24.
rfe = RFE(model, n_features_to_select=24)
rfe = rfe.fit(X_train, y_train)
col = X_train.columns[rfe.support_]

# Fit the model with the optimal set of features
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
res = logm2.fit()

# Evaluate the model's performance
model_evaluation(res, X_test, y_test, col)

# --- Variance Inflation Factor (VIF) Analysis ---

# Remove features with high multicollinearity to further optimize the model
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF scores for the selected features
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 4)
vif = vif.sort_values(by="VIF", ascending=False)

# Display the VIF values
print(vif)

# Drop features with excessively high VIF values to reduce multicollinearity
optimized_predictors = col.drop(['C16_90', 'C15_728'], axis=1)

# Fit the final model after addressing multicollinearity
X_train_sm = sm.add_constant(X_train[optimized_predictors])
logm3 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
res = logm3.fit()

# Evaluate the updated model
X_sm = sm.add_constant(X_test[optimized_predictors])
y_pred = res.predict(X_sm)

# Create a DataFrame to analyze predictions
y_train_pred_final = pd.DataFrame({'click': y_test, 'click_Prob': y_pred})
y_train_pred_final['predicted'] = y_train_pred_final.click_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Check the overall accuracy
print(metrics.accuracy_score(y_train_pred_final.click, y_train_pred_final.predicted))

# Display confusion matrix
confusion = metrics.confusion_matrix(y_train_pred_final.click, y_train_pred_final.predicted)
print(confusion)

# --- Optimizing Model Thresholds (Model 3) ---

# Adjusting threshold probabilities to find an optimal balance between recall, precision, and accuracy
# Creating a range of threshold values to test
numbers = [float(x) / 10 for x in range(10)]

# Adding new prediction columns for each threshold
for i in numbers:
    y_train_pred_final[i] = y_train_pred_final.click_Prob.map(lambda x: 1 if x > i else 0)

# Create a DataFrame to store evaluation metrics for each threshold
cutoff_df = pd.DataFrame(columns=['Threshold Probability', 'accuracy', 'recall', 'precision'])

# Evaluate each threshold
for i in numbers:
    cm1 = metrics.confusion_matrix(y_train_pred_final.click, y_train_pred_final[i])
    total = sum(sum(cm1))  # Total number of instances
    True_P = cm1[1, 1]  # True Positives
    True_N = cm1[0, 0]  # True Negatives
    False_P = cm1[0, 1]  # False Positives
    False_N = cm1[1, 0]  # False Negatives

    # Calculate metrics
    accuracy = (True_P + True_N) / total
    recall = True_P / (True_P + False_N) if (True_P + False_N) != 0 else 0
    precision = True_P / (True_P + False_P) if (True_P + False_P) != 0 else 0
    
    # Store results
    cutoff_df.loc[i] = [i, accuracy, recall, precision]

# Display the threshold evaluation DataFrame
print(cutoff_df)

# --- Visualizing Threshold Results ---

# Plot accuracy, precision, and recall for different thresholds
sns.lineplot(data=cutoff_df, x='Threshold Probability', y='accuracy', label='Accuracy')
sns.lineplot(data=cutoff_df, x='Threshold Probability', y='precision', label='Precision')
sns.lineplot(data=cutoff_df, x='Threshold Probability', y='recall', label='Recall')
plt.legend()
plt.show()

# The optimal cutoff is identified as approximately 0.2
optimal_threshold = 0.2

# Apply the optimal threshold to the final predictions
y_train_pred_final['predictions'] = y_train_pred_final.click_Prob.map(lambda x: 1 if x > optimal_threshold else 0)

# Evaluate the final model
print("Final Model Accuracy:", metrics.accuracy_score(y_train_pred_final.click, y_train_pred_final.predictions))
print("Final Model Confusion Matrix:")
print(metrics.confusion_matrix(y_train_pred_final.click, y_train_pred_final.predictions))

### Insights ###

# Out of the models built above, the final model offers the best balance between precision, recall, and accuracy.
# While accuracy slightly decreased, this tradeoff results in a better identification of positive cases (clicks).

# Key insights:
# - High accuracy alone is insufficient in this context due to the class imbalance (majority no-clicks).
# - The final model improves prediction of click events, which is more valuable than high overall accuracy.
