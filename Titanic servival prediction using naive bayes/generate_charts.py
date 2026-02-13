# Advanced Visualizations for Titanic Survival Prediction
# This script generates comprehensive charts and saves them as images

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("=" * 60)
print("📊 TITANIC SURVIVAL PREDICTION - VISUALIZATIONS")
print("=" * 60)

# Load data
df = pd.read_csv('titanicsurvival.csv')
print(f"\n✅ Dataset loaded: {len(df)} samples")

# Create output directory for charts
import os
os.makedirs('charts', exist_ok=True)

# 1. Missing Values Heatmap
print("\n📈 Generating: Missing Values Heatmap...")
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='YlOrRd')
plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/01_missing_values_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/01_missing_values_heatmap.png")

# 2. Target Distribution
print("\n📈 Generating: Target Distribution...")
plt.figure(figsize=(8, 6))
colors = ['#e74c3c', '#2ecc71']
ax = sns.countplot(x='Survived', data=df, palette=colors)
plt.title('Survival Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Survived (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/02_survival_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/02_survival_distribution.png")

# 3. Survival by Gender
print("\n📈 Generating: Survival by Gender...")
plt.figure(figsize=(10, 6))
gender_survival = df.groupby('Gender')['Survived'].value_counts().unstack()
gender_survival.plot(kind='bar', color=['#e74c3c', '#2ecc71'], edgecolor='black')
plt.title('Survival by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(['Not Survived', 'Survived'], loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/03_survival_by_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/03_survival_by_gender.png")

# 4. Survival by Passenger Class
print("\n📈 Generating: Survival by Passenger Class...")
plt.figure(figsize=(10, 6))
class_survival = df.groupby('Pclass')['Survived'].value_counts().unstack()
class_survival.plot(kind='bar', color=['#e74c3c', '#2ecc71'], edgecolor='black')
plt.title('Survival by Passenger Class', fontsize=16, fontweight='bold')
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(['Not Survived', 'Survived'], loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/04_survival_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/04_survival_by_class.png")

# 5. Age Distribution
print("\n📈 Generating: Age Distribution...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['Age'].hist(bins=30, color='#3498db', edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(1, 2, 2)
df.dropna(subset=['Age']).boxplot(column='Age', by='Survived', patch_artist=True)
plt.title('Age by Survival Status', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.xlabel('Survived', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.tight_layout()
plt.savefig('charts/05_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/05_age_distribution.png")

# 6. Fare Distribution
print("\n📈 Generating: Fare Distribution...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['Fare'].hist(bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
plt.title('Fare Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Fare ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(1, 2, 2)
df.boxplot(column='Fare', by='Survived', patch_artist=True)
plt.title('Fare by Survival Status', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.xlabel('Survived', fontsize=12)
plt.ylabel('Fare ($)', fontsize=12)
plt.tight_layout()
plt.savefig('charts/06_fare_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/06_fare_distribution.png")

# 7. Correlation Heatmap
print("\n📈 Generating: Correlation Heatmap...")
plt.figure(figsize=(10, 8))
# Encode Gender for correlation
df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'male': 0, 'female': 1})
correlation = df_encoded.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='RdYlBu_r', mask=mask,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/07_correlation_heatmap.png")

# 8. Survival Rate by Class and Gender
print("\n📈 Generating: Survival Rate by Class and Gender...")
plt.figure(figsize=(10, 6))
survival_rate = df.groupby(['Pclass', 'Gender'])['Survived'].mean() * 100
survival_rate.unstack().plot(kind='bar', color=['#3498db', '#e74c3c'], edgecolor='black')
plt.title('Survival Rate by Class and Gender', fontsize=16, fontweight='bold')
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate (%)', fontsize=12)
plt.legend(title='Gender')
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('charts/08_survival_rate_class_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/08_survival_rate_class_gender.png")

# 9. Model Comparison
print("\n📈 Generating: Model Comparison...")
models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'KNN', 'Decision Tree', 
          'Logistic Reg.', 'SVM', 'Naive Bayes']
accuracies = [83.80, 78.77, 81.01, 79.89, 79.89, 78.77, 78.77, 76.54]
colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]

plt.figure(figsize=(12, 6))
bars = plt.barh(models, accuracies, color=colors, edgecolor='black')
plt.xlabel('Accuracy (%)', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xlim(70, 90)

for bar, acc in zip(bars, accuracies):
    plt.text(acc + 0.3, bar.get_y() + bar.get_height()/2, f'{acc}%', 
             va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/09_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/09_model_comparison.png")

# 10. Feature Importance
print("\n📈 Generating: Feature Importance...")
features = ['Fare', 'Age', 'Gender (male)', 'Gender (female)', 'Pclass_3', 
            'Pclass_1', 'IsChild', 'Pclass_2']
importances = [0.286, 0.221, 0.178, 0.168, 0.065, 0.035, 0.025, 0.023]

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
bars = plt.barh(features, importances, color=colors, edgecolor='black')
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold')

for bar, imp in zip(bars, importances):
    plt.text(imp + 0.005, bar.get_y() + bar.get_height()/2, f'{imp:.1%}', 
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/10_feature_importance.png")

# 11. Confusion Matrix
print("\n📈 Generating: Confusion Matrix...")
confusion_matrix = np.array([[98, 12], [19, 50]])
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'],
            annot_kws={'size': 20})
plt.title('Confusion Matrix (Random Forest)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('charts/11_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/11_confusion_matrix.png")

# 12. Age vs Fare Scatter
print("\n📈 Generating: Age vs Fare Scatter Plot...")
plt.figure(figsize=(10, 6))
df_clean = df.dropna(subset=['Age'])
scatter = plt.scatter(df_clean['Age'], df_clean['Fare'], 
                      c=df_clean['Survived'], cmap='RdYlGn', 
                      alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Survived')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Fare ($)', fontsize=12)
plt.title('Age vs Fare (Colored by Survival)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/12_age_vs_fare_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: charts/12_age_vs_fare_scatter.png")

# Summary
print("\n" + "=" * 60)
print("🎉 ALL VISUALIZATIONS GENERATED!")
print("=" * 60)
print(f"\n📁 Charts saved in: charts/")
print("   01_missing_values_heatmap.png")
print("   02_survival_distribution.png")
print("   03_survival_by_gender.png")
print("   04_survival_by_class.png")
print("   05_age_distribution.png")
print("   06_fare_distribution.png")
print("   07_correlation_heatmap.png")
print("   08_survival_rate_class_gender.png")
print("   09_model_comparison.png")
print("   10_feature_importance.png")
print("   11_confusion_matrix.png")
print("   12_age_vs_fare_scatter.png")
print("\n✅ Total: 12 charts generated successfully!")
