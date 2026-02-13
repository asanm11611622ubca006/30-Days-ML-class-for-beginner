# Customer Spend Analysis Using K-Means Clustering - Project Workthrough

## Project Overview
This project analyzes customer spending patterns based on their income levels using K-Means Clustering, a machine learning algorithm that groups similar customers together. This helps businesses identify different customer segments for targeted marketing strategies.

---

## What is K-Means Clustering?
K-Means is a simple clustering algorithm that divides data points into K groups based on similarity:
- **Goal**: Find groups of customers with similar income-to-spending patterns
- **How it works**: Calculates distances between customers and groups them into clusters
- **Beginner-friendly**: Easy to understand and interpret results

---

## Project Step-by-Step Plan

### **Step 1: Data Loading & Exploration** 📊
- Load the dataset.csv file (contains INCOME and SPEND columns)
- Display first few rows to understand the data
- Check basic statistics (mean, min, max, standard deviation)
- Visualize data with a scatter plot to see the relationship between income and spending

### **Step 2: Data Preparation** 🧹
- Check for missing values
- Scale/normalize the data (important because income and spend values have different ranges)
  - This ensures both variables are equally weighted in the algorithm
- Prepare the data in the format required by K-Means

### **Step 3: Find Optimal Number of Clusters** 🎯
- Use the "Elbow Method" to determine the best number of clusters
  - Test different K values (2, 3, 4, 5, 6, 7, 8, 9, 10)
  - Calculate inertia (how tight the clusters are)
  - Plot inertia vs K values - look for the "elbow" point where inertia stops decreasing significantly
  - This prevents under-clustering (too few groups) or over-clustering (too many groups)

### **Step 4: Apply K-Means Clustering** 🔄
- Based on the elbow method, apply K-Means with optimal K value
- Assign each customer to a cluster
- Display cluster centers (representative customer profiles)

### **Step 5: Visualize Results** 📈
- **Elbow curve**: Shows why we chose the optimal K value
- **Scatter plot with clusters**: Color-coded points showing which cluster each customer belongs to
- **Cluster centers**: Mark the centroid of each cluster
- **Cluster statistics**: Summary table showing average income and spend per cluster

### **Step 6: Business Insights** 💡
- Interpret clusters (e.g., "Budget-conscious customers", "Premium customers", "Balanced spenders")
- Analyze characteristics of each segment
- Provide actionable recommendations

---

## Expected Outcomes

**Graphs You'll See:**
1. Initial data scatter plot (before clustering)
2. Elbow curve (to show optimal K selection)
3. Final clustered data visualization (colored by cluster)
4. Cluster centers marked on the plot

**Insights You'll Get:**
- Customer segments based on income and spending patterns
- Average income and spending per segment
- Business recommendations for each customer group

---

## Libraries Used
- **pandas**: Data loading and manipulation
- **numpy**: Numerical calculations
- **scikit-learn**: K-Means algorithm and preprocessing
- **matplotlib & seaborn**: Visualizations and graphs
- **plotly** (optional): Interactive visualizations

---

## Dataset Info
- **Rows**: 305 customers
- **Columns**: 
  - INCOME: Annual customer income
  - SPEND: Annual customer spending

---

## Output Files
After running the notebook, you'll have:
1. A Jupyter Notebook (.ipynb) with all code and visualizations
2. Generated graphs embedded in the notebook
3. Clear interpretations of customer segments

---

## Difficulty Level: ⭐ Beginner-Friendly
- Each step is explained in detail
- Code is well-commented
- Visualizations help understand the results
- No advanced math knowledge required

---

**Ready to proceed?** Once you approve this plan, I'll create the complete Jupyter Notebook with all code and graphs!
