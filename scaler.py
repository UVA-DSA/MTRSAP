import os
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Define the directory containing CSV files
folder_path = "./segmentation_masks/pca_features"


# Initialize an empty DataFrame to store all the data
all_data = pd.DataFrame()

# Iterate through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Append the data to the combined DataFrame
        all_data = pd.concat([all_data, df], axis=0)

# Learn a StandardScaler on all the data
scaler = StandardScaler()
scaler.fit(all_data)

# Iterate through each CSV file again and transform the data
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Transform the data using the learned scaler
        scaled_data = scaler.transform(df)
        
        # Save the transformed data back to the same path
        transformed_file_path = os.path.join("./segmentation_masks/pca_features_normalized", filename)
        pd.DataFrame(scaled_data, columns=df.columns).to_csv(transformed_file_path, index=False)

print("Scalings and saving complete.")

# # Initialize an empty DataFrame to store all the data
# all_data = pd.DataFrame()

# # Iterate through each CSV file in the folder
# for filename in tqdm(os.listdir(folder_path), desc='Loading Files'):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(folder_path, filename)
        
#         # Load the CSV file into a DataFrame
#         df = pd.read_csv(file_path)
        
#         # Append the data to the combined DataFrame
#         all_data = pd.concat([all_data, df], axis=0)

# # Define the number of principal components to keep
# n_components = 32  # Adjust this as needed

# # Initialize the PCA transformer
# print("Learning PCA")
# pca = PCA(n_components=n_components)

# # Fit PCA on all the data
# pca.fit(all_data)

# # Iterate through each CSV file again and transform the data using PCA
# for filename in tqdm(os.listdir(folder_path), desc='Applying PCA'):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(folder_path, filename)
        
#         # Load the CSV file into a DataFrame
#         df = pd.read_csv(file_path)
        
#         # Transform the data using the learned PCA transformer
#         transformed_data = pca.transform(df)
        
#         # Save the transformed data back to the same path
#         transformed_file_path = os.path.join("./segmentation_masks/pca_features_normalized", filename)
#         pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(n_components)]).to_csv(transformed_file_path, index=False, header=False)

# print("PCA transformation and saving complete.")
