import argparse
import pandas as pd
import shap
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

"""
Given a train/validate and test dataset, will train and evaluate a random forest model

Args:
    train_validate_filepath (str)   : Filepath to the train/validate dataset.
    test_filepath (str)             : Filepath to the test dataset.
    bone_type (str)                 : Either t or r, representing which bone you want to make a model for

Returns:
    None: This model will produce the model coefficients, performance metrics, and testing results in the 3 different csv files
"""

# Parsing arguments
parser = argparse.ArgumentParser(description='Support Vector Regression with Cross-Validation on Train-Validate and Test Sets')
parser.add_argument('train_validate_filepath', type=str, help='Path to the train-validate CSV file')
parser.add_argument('test_filepath', type=str, help='Path to the test CSV file')
parser.add_argument('bone_type', type=str, choices=['t', 'r'], help='Specify whether predicting tibia (t) or radius (r) bone strength')
parser.add_argument('ID', type=int, help="Specify the ID that slurm provides")
args = parser.parse_args()

# Function to process dataset
def process_data(data, bone_type):
    X = data.drop(["study_id", "label_t", "label_r"], axis=1)
    y = data[f"label_{bone_type}"]
    return X, y

# Load and process the train-validate and test datasets
train_validate_data = pd.read_csv(args.train_validate_filepath)
test_data = pd.read_csv(args.test_filepath)
X_train_val, y_train_val = process_data(train_validate_data, args.bone_type)
X_test, y_test = process_data(test_data, args.bone_type)

# Define the parameter space for BayesSearchCV
param_spaces = {
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),  
    'degree': Integer(1, 5),  
    'gamma': Real(1e-4, 1e-1, prior='log-uniform'),  
    'coef0': Real(0.0, 10.0),  
    'tol': Real(1e-5, 1e-1, prior='log-uniform'),  
    'C': Real(0.1, 100.0, prior='log-uniform'),  
    'epsilon': Real(0.01, 1.0),  
    'shrinking': Categorical([True, False]),  
    'max_iter': Integer(100, 20000),  
}

# Initialize BayesSearchCV
opt = BayesSearchCV(
    estimator=SVR(),
    search_spaces=param_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    return_train_score=True,
    random_state=42
    
)

# Fit BayesSearchCV to the train-validate dataset
opt.fit(X_train_val, y_train_val)

# Get the best model and its parameters
best_model = opt.best_estimator_
print("Best score:", opt.best_score_)
print("Best parameters:", opt.best_params_)

# Create a folder with the name ID and ensure all saved files will be in that folder
folder_name = str(args.ID)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save training and test metrics, and SHAP values
base_filename = os.path.splitext(os.path.basename(args.train_validate_filepath))[0]
shap_filename = os.path.join(folder_name, f"{base_filename}_SHAP_Values.csv")
training_metrics_filename = os.path.join(folder_name, f"{base_filename}_TrainingMetrics.csv")
test_metrics_filename = os.path.join(folder_name, f"{base_filename}_TestMetrics.csv")

# Evaluate the best model on the test dataset
y_pred_test = best_model.predict(X_test)
test_metrics = {
    'Test MSE': mean_squared_error(y_test, y_pred_test),
    'Test R2': r2_score(y_test, y_pred_test),
    'Test MAE': mean_absolute_error(y_test, y_pred_test),
}

# Obtain and save training performance metrics
training_metrics = {
    'Training Mean Score': opt.cv_results_['mean_train_score'].max(),  # Adjusted to get the maximum mean score
    'Training Std Score': opt.cv_results_['std_train_score'][opt.cv_results_['mean_train_score'].argmax()]  # Adjusted accordingly
}

# Save training metrics to a CSV file
pd.DataFrame([training_metrics]).to_csv(training_metrics_filename, index=False)
print(f"Training metrics saved to {training_metrics_filename}")
pd.DataFrame([test_metrics]).to_csv(test_metrics_filename, index=False)
print(f"Test metrics saved to {test_metrics_filename}")

# Saving SHAP values 
explainer = shap.Explainer(best_model.predict, shap.sample(X_train_val, 100)) 
shap_values = explainer(shap.sample(X_test, 100))  
shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)  
shap_df.to_csv(shap_filename, index=False)
print(f"SHAP values saved to {shap_filename}")

print(f"All files have been saved in the folder: {folder_name}")
print("SVR complete")

