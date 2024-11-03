import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('online_bookstore_dataset.csv')

# Step 1: Handle missing values (if any)
df.dropna(inplace=True)

# Step 2: Encode categorical variables
# Encoding 'product_category' for machine learning compatibility
label_encoder = LabelEncoder()
df['product_category_encoded'] = label_encoder.fit_transform(df['product_category'])

# Drop non-numeric columns that are not needed
df.drop(['customer_id', 'order_id', 'purchase_date', 'product_id', 'product_category'], axis=1, inplace=True)

# Step 3: Normalize numerical features for better model performance
scaler = StandardScaler()
numerical_features = ['quantity', 'unit_price', 'discount', 'total_order_value']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Step 4: Feature Reduction using PCA on numeric data only
pca = PCA(n_components=0.95)  # Retain 95% of the variance
df_pca = pca.fit_transform(df)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_pca, df['total_order_value'], test_size=0.2, random_state=42)

# Save preprocessed data
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

print("Data preparation completed and files saved.")
