# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df = pd.read_csv("online_bookstore_dataset.csv")

# Set up plot style for readability
sns.set(style="whitegrid")

# 1. Overview of the Data
# Display the first few rows of the dataset to understand its structure
print("Data Overview:")
print(df.head())

# Display basic statistics for each numerical column
print("\nSummary Statistics:")
print(df.describe())

# Check the correlation between numerical features
correlation_matrix = df[['quantity', 'unit_price', 'discount', 'total_order_value']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# 2. Order Trends
# Convert 'purchase_date' to datetime format and extract 'year' and 'month'
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['order_year'] = df['purchase_date'].dt.year
df['order_month'] = df['purchase_date'].dt.month

# Orders per month-year
monthly_orders = df.groupby(['order_year', 'order_month']).size().reset_index(name='order_count')
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_orders, x='order_month', y='order_count', hue='order_year', marker="o")
plt.title("Monthly Orders Count by Year")
plt.xlabel("Month")
plt.ylabel("Order Count")
plt.legend(title="Year")
plt.show()

# Average order value over time
monthly_avg_order_value = df.groupby(['order_year', 'order_month'])['total_order_value'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_order_value, x='order_month', y='total_order_value', hue='order_year', marker="o")
plt.title("Average Order Value by Month and Year")
plt.xlabel("Month")
plt.ylabel("Average Order Value")
plt.legend(title="Year")
plt.show()

# 3. Product Analysis
# Top 10 most popular product categories
top_categories = df['product_category'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories.values, y=top_categories.index)
plt.title("Top 10 Most Popular Product Categories")
plt.xlabel("Order Count")
plt.ylabel("Product Category")
plt.show()

# Average total amount spent per product category
avg_total_amount_per_category = df.groupby('product_category')['total_order_value'].mean().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_total_amount_per_category.values, y=avg_total_amount_per_category.index)
plt.title("Average Total Amount per Top 10 Product Categories")
plt.xlabel("Average Total Amount")
plt.ylabel("Product Category")
plt.show()

# 4. Insights for Machine Learning
# Average order value distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['total_order_value'], kde=True)
plt.title("Distribution of Total Order Value")
plt.xlabel("Total Order Value")
plt.ylabel("Frequency")
plt.show()

# Visualize the relationship between order quantity and total spending
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='quantity', y='total_order_value', alpha=0.6)
plt.title("Relationship Between Order Quantity and Total Order Value")
plt.xlabel("Order Quantity")
plt.ylabel("Total Order Value")
plt.show()

# Summary of Key Insights
print("\nKey Insights:")
print("- Strong correlation between quantity and total order value can help in segmentation.")
print("- Monthly order trends and average order value provide a basis for sales prediction.")
print("- Popular product categories and high-spending categories can aid in product recommendations.")
