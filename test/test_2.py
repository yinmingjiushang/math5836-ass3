import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the abalone dataset
data = pd.read_csv('../data/abalone_data.csv')

# Define age classes based on the rings
age_bins = [0, 7, 10, 15, float('inf')]
age_labels = ['Class 1: 0-7 years', 'Class 2: 8-10 years', 'Class 3: 11-15 years', 'Class 4: >15 years']
data['Age Class'] = pd.cut(data['Rings'], bins=age_bins, labels=age_labels, right=True)

# Display the first few rows to understand the dataset structure
print(data.head())

# Update feature column names to match dataset
feature_columns = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']

# Distribution of classes
plt.figure(figsize=(10, 6))
sns.countplot(x='Age Class', data=data, palette='viridis')
plt.title('Distribution of Age Classes')
plt.xlabel('Age Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Distribution of features
plt.figure(figsize=(15, 10))
data[feature_columns].hist(bins=20, figsize=(15, 10), layout=(3, 3), color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Features')
plt.show()

# Pairplot for visualizing relationships
sns.pairplot(data[feature_columns + ['Age Class']], hue='Age Class', palette='husl', plot_kws={'alpha': 0.5})
plt.show()

# Summary statistics for each feature
summary_stats = data[feature_columns].describe()
print("Summary Statistics for Each Feature:\n", summary_stats)

# Save the summary statistics as a CSV file
summary_stats.to_csv('../data/summary_statistics.csv', index=True)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data[feature_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()
