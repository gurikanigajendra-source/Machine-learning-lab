import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Load the dataset
df = pd.read_csv("dataset/customer_segmentation_dataset.csv")

# Display the first 5 rows
print(df.head())

# Check the number of rows and columns
print("Shape of dataset:")
print(df.shape)

# Check column names and data types
print("\nDataset information:")
print(df.info())

# Generate statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Visualise numerical features
features = ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency"]

for feature in features:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Customers")
    plt.show()

# Correlation between numerical features
plt.figure(figsize=(8, 6))

correlation = df[features].corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Between Customer Features")
plt.show()

from sklearn.preprocessing import StandardScaler

# Select features for clustering
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled feature data:")
print(X_scaled[:5])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Find the optimal number of clusters using Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

# Fit the model and assign cluster labels
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display customers with their cluster
print("\nCustomer clusters:")
print(df[["CustomerID", "Age", "AnnualIncome",
          "SpendingScore", "PurchaseFrequency", "Cluster"]])

# Visualise K-Means clusters
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="Set1",
    s=100,
    edgecolor="black"
)

# Convert cluster centres back to the original scale
centres = scaler.inverse_transform(kmeans.cluster_centers_)

# Feature order: Age, AnnualIncome, SpendingScore, PurchaseFrequency
plt.scatter(
    centres[:, 1],
    centres[:, 2],
    marker="X",
    s=250,
    color="black",
    edgecolor="white",
    label="Centroids"
)

plt.title("K-Means Customer Segmentation with Centroids")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# Analyse each customer cluster
cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster Summary:")
print(cluster_summary.round(2))

# Count customers in each cluster
print("\nNumber of customers in each cluster:")
print(df["Cluster"].value_counts().sort_index())


from sklearn.metrics import silhouette_score

# Calculate Silhouette Score
silhouette = silhouette_score(X_scaled, df["Cluster"])

print("\nSilhouette Score:")
print(round(silhouette, 3))

from sklearn.cluster import DBSCAN

# Apply DBSCAN

# Apply DBSCAN with selected parameters
dbscan = DBSCAN(eps=0.5, min_samples=3)

df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

print("\nDBSCAN Cluster Results:")
print(df[["CustomerID", "DBSCAN_Cluster"]])

print("\nNumber of customers in each DBSCAN cluster:")
print(df["DBSCAN_Cluster"].value_counts().sort_index())

# Analyse DBSCAN clusters
dbscan_summary = df.groupby("DBSCAN_Cluster")[features].mean()

print("\nDBSCAN Cluster Summary:")
print(dbscan_summary.round(2))

# Visualise DBSCAN clusters
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="DBSCAN_Cluster",
    palette="tab10",
    s=100,
    edgecolor="black"
)

plt.title("DBSCAN Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="DBSCAN Cluster")
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# Compare K-Means and DBSCAN

kmeans_clusters = df["Cluster"].nunique()

dbscan_clusters = len(
    set(df["DBSCAN_Cluster"]) - {-1}
)

dbscan_noise = (df["DBSCAN_Cluster"] == -1).sum()

print("\n--- K-Means vs DBSCAN Comparison ---")

print("K-Means clusters:", kmeans_clusters)
print("K-Means Silhouette Score:", round(silhouette, 3))

print("DBSCAN clusters:", dbscan_clusters)
print("DBSCAN noise points:", dbscan_noise)



plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="Set1",
    s=120,
    edgecolor="black"
)

plt.title("Customer Segmentation Scatter Plot")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Customer Cluster")
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-Means clustering graph
sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="Set1",
    s=100,
    edgecolor="black",
    ax=axes[0]
)

axes[0].set_title("K-Means Clustering (K = 4)")
axes[0].set_xlabel("Annual Income")
axes[0].set_ylabel("Spending Score")
axes[0].legend(title="K-Means Cluster")

# DBSCAN clustering graph
sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="DBSCAN_Cluster",
    palette="tab10",
    s=100,
    edgecolor="black",
    ax=axes[1]
)

axes[1].set_title("DBSCAN Clustering")
axes[1].set_xlabel("Annual Income")
axes[1].set_ylabel("Spending Score")
axes[1].legend(title="DBSCAN Cluster")

plt.suptitle("Comparison of K-Means and DBSCAN", fontsize=16)
plt.tight_layout()

plt.savefig(
    "kmeans_dbscan_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()