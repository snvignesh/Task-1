#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

print(customers.head())
print(products.head())
print(transactions.head())

print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

customers.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)merged_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')
print(merged_data.head())from sklearn.preprocessing import LabelEncoder

customer_features = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'Category': lambda x: x.mode()[0]
}).reset_index()

le = LabelEncoder()
customer_features['Category'] = le.fit_transform(customer_features['Category'])
print(customer_features.head())from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(customer_features.iloc[:, 1:])

similarity_matrix = cosine_similarity(features_scaled)

similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

lookalike = {}
for customer in similarity_df.index:
    similar_customers = similarity_df.loc[customer].sort_values(ascending=False)[1:4]
    lookalike[customer] = list(zip(similar_customers.index, similar_customers.values))

import csv
with open('YourName_Lookalike.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['CustomerID', 'Lookalikes'])
    for key, value in lookalike.items():
        writer.writerow([key, value])region_sales = merged_data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
print(region_sales)

region_sales.plot(kind='bar', title='Total Sales by Region', color='skyblue')
plt.xlabel('Region')
plt.ylabel('Total Sales (USD)')
plt.show()

category_sales = merged_data.groupby('Category')['Quantity'].sum().sort_values(ascending=False)
print(category_sales)

category_sales.plot(kind='bar', title='Most Popular Product Categories', color='orange')
plt.xlabel('Product Category')
plt.ylabel('Total Quantity Sold')
plt.show()

customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
signup_trend = customers['SignupDate'].dt.year.value_counts().sort_index()

signup_trend.plot(kind='line', title='Customer Signup Trend Over Years', marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Signups')
plt.show()customer_segmentation = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'TransactionID': 'count'
}).reset_index()

customer_segmentation.rename(columns={'TransactionID': 'Frequency'}, inplace=True)
print(customer_segmentation.head())from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

features_scaled = scaler.fit_transform(customer_segmentation.iloc[:, 1:])

kmeans = KMeans(n_clusters=5, random_state=42)
customer_segmentation['Cluster'] = kmeans.fit_predict(features_scaled)

db_index = davies_bouldin_score(features_scaled, customer_segmentation['Cluster'])
print(db_index)from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features_scaled)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=customer_segmentation['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Customer Segments')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')
plt.show() Task-1
