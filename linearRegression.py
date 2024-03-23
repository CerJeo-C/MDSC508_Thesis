import argparse
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap


"""
Given a train/validate and test dataset, will train and evaluate a linear regression model

Args:
    train_validate_filepath (str)   : Filepath to the train/validate dataset.
    test_filepath (str)             : Filepath to the test dataset.
    bone_type (str)                 : Either t or r, representing which bone you want to make a model for

Returns:
    None: This model will produce the model coefficients, performance metrics, and testing results in the 3 different csv files
"""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Linear Regression with Cross-Validation and SHAP Analysis')
parser.add_argument('train_validate_filepath', type=str, help='Path to the train-validate CSV file')
parser.add_argument('test_filepath', type=str, help='Path to the test CSV file')
parser.add_argument('bone_type', type=str, choices=['t', 'r'], help='Bone type (tibia or radius)')
parser.add_argument('ID', type=int, help="Specify the ID that slurm provides")
args = parser.parse_args()

# Load datasets
train_validate_data = pd.read_csv(args.train_validate_filepath)
test_data = pd.read_csv(args.test_filepath)

# Process data
def process_data(data, bone_type):
    X = data.drop("study_id", axis=1)
    bone_label = 'label_t' if bone_type == 't' else 'label_r'
    X = X.drop(['label_r', 'label_t'], axis=1)
    y = data[bone_label]
    return X, y

X_train_val, y_train_val = process_data(train_validate_data, args.bone_type)
X_test, y_test = process_data(test_data, args.bone_type)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare to collect metrics
metrics_list = []

# Cross-validation
for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    metrics_list.append({'MSE': mse, 'R2': r2, 'MAE': mae})

# Train final model
final_model = LinearRegression()
final_model.fit(X_train_val, y_train_val)
y_pred_test = final_model.predict(X_test)

test_metrics = {
    'Test MSE': mean_squared_error(y_test, y_pred_test),
    'Test R2': r2_score(y_test, y_pred_test),
    'Test MAE': mean_absolute_error(y_test, y_pred_test)
}

# Create a folder with the name ID and ensure all saved files will be in that folder
folder_name = str(args.ID)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save predicted and true values for the test dataset
test_results_df = pd.DataFrame({'True Value': y_test, 'Predicted Value': y_pred_test})
test_results_file = os.path.join(folder_name, f"{os.path.splitext(os.path.basename(args.train_validate_filepath))[0]}_Test_Results.csv")
test_results_df.to_csv(test_results_file, index=False)

# Compute SHAP values
explainer = shap.Explainer(final_model, X_train_val)
shap_values = explainer(X_test)

# Export SHAP values
shap_values_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
shap_output_file = os.path.join(folder_name, f"{os.path.splitext(os.path.basename(args.train_validate_filepath))[0]}_SHAPValues.csv")
shap_values_df.to_csv(shap_output_file, index=False)
print(f"SHAP values exported to {shap_output_file}.")

# Export metrics and coefficients
metrics_df = pd.DataFrame(metrics_list)
metrics_results_file = os.path.join(folder_name,f"{os.path.splitext(os.path.basename(args.train_validate_filepath))[0]}_CV_Metrics.csv")
metrics_df.to_csv(metrics_results_file, index=False)

test_metrics_df = pd.DataFrame([test_metrics])
metrics_results_file = os.path.join(folder_name, f"{os.path.splitext(os.path.basename(args.train_validate_filepath))[0]}_Test_Metrics.csv")
test_metrics_df.to_csv(metrics_results_file, index=False)

coefficients_df = pd.DataFrame([final_model.coef_], columns=X_train_val.columns)
coefficients_results_file = os.path.join(folder_name,f"{os.path.splitext(os.path.basename(args.train_validate_filepath))[0]}_Coefficients.csv")
coefficients_df.to_csv(coefficients_results_file, index=False)
