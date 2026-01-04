#############################################################################################################
# Required Python Packages
#############################################################################################################
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib


#############################################################################################################
# File Paths
#############################################################################################################
INPUT_PATH = "Mall_Customers.csv"
MODEL_PATH = "kmeans_customer_segmentation.joblib"
IMAGE_DIR = "generated_images"


#############################################################################################################
# Headers
#############################################################################################################
FEATURE_HEADERS = ['Annual Income (k$)', 'Spending Score (1-100)']


#############################################################################################################
# Function name :       create_image_directory
# Description :         Create directory to store generated plots
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def create_image_directory():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"\nImage directory created : {IMAGE_DIR}")
    else:
        print(f"\nImage directory already exists : {IMAGE_DIR}")


#############################################################################################################
# Function name :       save_plot
# Description :         Save matplotlib plot into image directory
# Input :               File name
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def save_plot(file_name):
    path = os.path.join(IMAGE_DIR, file_name)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved : {path}")


#############################################################################################################
# Function name :       dataset_statistics
# Description :         Display dataset statistics
# Input :               Dataset
# Output :              Dataset summary
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def dataset_statistics(dataset):
    border = '-' * 120
    print(border)
    print("DATASET STATISTICS")
    print(border)
    print(f"Dataset Shape : {dataset.shape}")
    print("\nFirst 5 Records:")
    print(dataset.head())
    print("\nNull Values:")
    print(dataset.isnull().sum())
    print("\nStatistical Summary:")
    print(dataset.describe())
    print(border)


#############################################################################################################
# Function name :       plot_elbow_method
# Description :         Determine optimal K using Elbow Method
# Input :               Scaled features
# Output :              Elbow plot
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def plot_elbow_method(features):
    wcss = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    save_plot("elbow_method.png")
    plt.show()


#############################################################################################################
# Function name :       apply_kmeans
# Description :         Apply KMeans clustering
# Input :               Features, number of clusters
# Output :              Trained model and cluster labels
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def apply_kmeans(features, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)
    return kmeans, labels


#############################################################################################################
# Function name :       visualize_clusters
# Description :         Visualize customer clusters
# Input :               Dataset
# Output :              Cluster plot
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def visualize_clusters(dataset):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=dataset['Annual Income (k$)'],
        y=dataset['Spending Score (1-100)'],
        hue=dataset['Cluster'],
        palette='viridis'
    )
    plt.title("Customer Segmentation using K-Means")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    save_plot("customer_clusters.png")
    plt.show()


#############################################################################################################
# Function name :       save_model
# Description :         Save trained KMeans model
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"\nModel saved at : {path}")


#############################################################################################################
# Function name :       load_model
# Description :         Load saved KMeans model
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"\nModel loaded from : {path}")
    return model


#############################################################################################################
# Function name :       main
# Description :         Program execution starts here
# Author :              Shubham Londhe
# Date :                10/11/2025
#############################################################################################################
def main():
    border = '-' * 120
    print(border)
    print(" " * 35 + "CUSTOMER SEGMENTATION USING K-MEANS")
    print(border)

    # Create directory for images
    create_image_directory()

    # Load dataset
    dataset = pd.read_csv(INPUT_PATH)

    # Dataset statistics
    dataset_statistics(dataset)

    # Select features
    X = dataset[FEATURE_HEADERS]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    plot_elbow_method(X_scaled)

    # Apply KMeans (K = 5)
    kmeans_model, dataset['Cluster'] = apply_kmeans(X_scaled, k=5)

    # Save model
    save_model(kmeans_model)

    # Load model (verification)
    load_model()

    # Visualize clusters
    visualize_clusters(dataset)

    print(border)
    print(" " * 45 + "PROGRAM TERMINATED")
    print(border)


#############################################################################################################
# Application Starter
#############################################################################################################
if __name__ == "__main__":
    main()
