# Customer Segmentation Using K-Means Clustering

This project implements **Customer Segmentation** for a retail store using the
**K-Means clustering algorithm**. The goal is to group customers based on their
purchase behavior so that businesses can better understand customer patterns
and plan targeted marketing strategies.

---

## 📌 Project Objectives

- Perform customer segmentation using K-Means clustering
- Identify meaningful customer groups based on purchasing behavior
- Determine the optimal number of clusters using the Elbow Method
- Visualize customer clusters
- Save the trained clustering model using `joblib`
- Automatically save all generated plots

---

## 📊 Dataset Information

**Dataset Name:** Mall Customer Segmentation Dataset  
**Source:** Kaggle  
**Link:**  
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Dataset Attributes
- CustomerID – Unique customer identifier
- Gender – Gender of the customer
- Age – Age of the customer
- Annual Income (k$) – Annual income in thousand dollars
- Spending Score (1-100) – Score assigned based on customer behavior

---

## 🧠 Features Used for Clustering

- Annual Income (k$)
- Spending Score (1-100)

---

## 📁 Project Structure

project_folder/
│
├── Mall_Customers.csv
├── CustomerSegmentationKMeans.py
├── requirements.txt
├── kmeans_customer_segmentation.joblib
│
└── generated_images/
    ├── elbow_method.png
    └── customer_clusters.png

---

## ⚙️ Installation & Setup

### Install Required Libraries
pip install -r requirements.txt

### Download Dataset
- Download Mall_Customers.csv from Kaggle
- Place it in the project directory

---

## ▶️ How to Run

python CustomerSegmentationKMeans.py

---

## 🔍 Methodology

1. Load the customer dataset
2. Display dataset statistics
3. Select relevant features for clustering
4. Normalize features using StandardScaler
5. Apply the Elbow Method to determine optimal clusters
6. Train K-Means clustering model
7. Assign cluster labels to customers
8. Visualize customer clusters
9. Save trained model using joblib
10. Load saved model for verification

---

## 📈 Visualizations Generated

All plots are automatically saved in the generated_images folder:
- Elbow Method graph
- Customer cluster visualization

---

## 💾 Model Persistence

The trained K-Means model is saved as:
kmeans_customer_segmentation.joblib

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 👨‍💻 Author

Shubham Londhe  
Date: 10/11/2025

---

## ✅ Conclusion

This project demonstrates an end-to-end implementation of K-Means clustering
for customer segmentation, including preprocessing, visualization,
model persistence, and analysis.
