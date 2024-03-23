import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_and_normalize_data(csv_file_path, exclude_column):
    # Load the data
    data = pd.read_csv(csv_file_path)

    # Exclude the specified column from normalization and encoding
    if exclude_column in data.columns:
        excluded_data = data[[exclude_column]]
        data = data.drop(columns=[exclude_column])
    else:
        excluded_data = pd.DataFrame()

    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Normalize numerical columns
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # One-hot encode categorical columns (if needed)
    if len(categorical_cols) > 0:
        one_hot_encoder = OneHotEncoder()
        one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_cols]).toarray()
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
        
        # Drop original categorical columns and concatenate the one-hot encoded columns
        data = data.drop(categorical_cols, axis=1)
        data = pd.concat([data, one_hot_encoded_df], axis=1)

    # Re-add the excluded column
    data = pd.concat([excluded_data, data], axis=1)

    return data

def save_normalized_data(data, output_file_path):
    # Save the normalized data to a CSV file
    data.to_csv(output_file_path, index=False)
    print(f"Data saved to {output_file_path}")

csv_file_path = r"C:\Users\potat\OneDrive\Desktop\Year 4\MDSC508\DataPreprocessing\merged_data.csv"
exclude_column = 'study_id' 
normalized_data = load_and_normalize_data(csv_file_path, exclude_column)

# Save the normalized data
output_file_path = 'tabular_data.csv'
save_normalized_data(normalized_data, output_file_path)
