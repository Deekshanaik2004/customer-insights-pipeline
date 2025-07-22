import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Step 1: Load and clean the data --------------------

df = pd.read_csv('data/transactions.csv', parse_dates=['invoice_date'])

# Remove duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("\n✅ Data loaded and cleaned.")
print(df.head())

# -------------------- Step 2: Cohort Analysis --------------------

df['InvoiceMonth'] = df['invoice_date'].dt.to_period('M')
df['CohortMonth'] = df.groupby('customer_id')['invoice_date'].transform('min').dt.to_period('M')
df['CohortIndex'] = (df['InvoiceMonth'].dt.year - df['CohortMonth'].dt.year) * 12 + \
                    (df['InvoiceMonth'].dt.month - df['CohortMonth'].dt.month) + 1

cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['customer_id'].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='customer_id')

# Plot retention heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(cohort_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Retention Heatmap")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Cohort Month")
plt.show()

# -------------------- Step 3: Segmentation --------------------

# Frequency = number of transactions
# Average Spend = average amount spent per transaction
agg_data = df.groupby('customer_id').agg({
    'invoice_date': 'count',
    'amount': 'mean'
}).rename(columns={'invoice_date': 'frequency', 'amount': 'avg_spend'})

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
agg_data['cluster'] = kmeans.fit_predict(agg_data)

print("\n📊 Customer Segments:")
print(agg_data.head())

# Plot segments
plt.figure(figsize=(8, 6))
sns.scatterplot(data=agg_data, x='frequency', y='avg_spend', hue='cluster', palette='Set2')
plt.title('Customer Segmentation')
plt.xlabel('Transaction Frequency')
plt.ylabel('Average Spend')
plt.grid(True)
plt.show()

# -------------------- Step 4: A/B Testing --------------------

# Just a simulation: split customers randomly by ID for A/B test
group_a = df[df['customer_id'] % 2 == 0]['amount']
group_b = df[df['customer_id'] % 2 != 0]['amount']

# Perform t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"\n🔬 A/B Test Results:\nT-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")

# Plot A/B test result
plt.figure(figsize=(8, 5))
sns.kdeplot(group_a, label='Group A', shade=True)
sns.kdeplot(group_b, label='Group B', shade=True)
plt.title("A/B Test Spend Distribution")
plt.legend()
plt.grid(True)
plt.show()
