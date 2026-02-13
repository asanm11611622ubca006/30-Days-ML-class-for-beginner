"""
Movie Recommendation System using SVD
This script executes the recommendation system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("🎬 MOVIE RECOMMENDATION SYSTEM USING SVD")
print("="*80)

# Load ratings data
print("\n📂 Step 1: Loading Data...")
ratings = pd.read_csv(
    'ratings.dat',
    sep='::',
    engine='python',
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    encoding='latin-1'
)

# Load movies data
movies = pd.read_csv(
    'movies.dat',
    sep='::',
    engine='python',
    names=['movie_id', 'title', 'genre'],
    encoding='latin-1'
)

print("✅ Data loaded successfully!")
print(f"\n📊 Ratings dataset shape: {ratings.shape}")
print(f"📊 Movies dataset shape: {movies.shape}")

# Dataset Statistics
n_users = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()
n_ratings = len(ratings)

print(f"\n{'='*80}")
print(f"📊 DATASET STATISTICS:")
print(f"{'='*80}")
print(f"👥 Total Users: {n_users:,}")
print(f"🎬 Total Movies: {n_movies:,}")
print(f"⭐ Total Ratings: {n_ratings:,}")
print(f"\n📈 Average ratings per user: {n_ratings/n_users:.1f}")
print(f"📈 Average ratings per movie: {n_ratings/n_movies:.1f}")
print(f"🎯 Matrix Sparsity: {100 - (n_ratings / (n_users * n_movies)) * 100:.2f}% empty")

# Create user-item matrix
print(f"\n{'='*80}")
print("🔨 Step 2: Creating User-Item Matrix...")
print(f"{'='*80}")

user_item_matrix = ratings.pivot(
    index='user_id',
    columns='movie_id',
    values='rating'
)

print(f"✅ Matrix created!")
print(f"📐 Matrix Shape: {user_item_matrix.shape}")
print(f"   ({user_item_matrix.shape[0]} users × {user_item_matrix.shape[1]} movies)")

# Prepare data for SVD
print(f"\n{'='*80}")
print("🎯 Step 3: Preparing Data for SVD...")
print(f"{'='*80}")

user_item_matrix_filled = user_item_matrix.fillna(0)
user_ratings_mean = user_item_matrix.mean(axis=1)
user_item_matrix_normalized = user_item_matrix_filled.sub(user_ratings_mean, axis=0)

print("✅ Data normalized!")
print(f"💡 Normalized to account for harsh vs. generous raters!")

# Apply SVD
print(f"\n{'='*80}")
print("🔮 Step 4: Applying SVD Decomposition...")
print(f"{'='*80}")

k = 50  # Number of latent factors
U, sigma, Vt = svds(user_item_matrix_normalized.values, k=k)
sigma = np.diag(sigma)

print("✅ SVD completed!")
print(f"\n📐 Matrix Shapes:")
print(f"   U (User features):    {U.shape}")
print(f"   Σ (Singular values):  {sigma.shape}")
print(f"   V^T (Movie features): {Vt.shape}")
print(f"\n💡 Reduced {user_item_matrix.shape[1]} movies to {k} hidden patterns!")

# Reconstruct & Predict
print(f"\n{'='*80}")
print("🎬 Step 5: Reconstructing & Predicting Ratings...")
print(f"{'='*80}")

predicted_ratings_normalized = np.dot(np.dot(U, sigma), Vt)
predicted_ratings = predicted_ratings_normalized + user_ratings_mean.values.reshape(-1, 1)

predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    columns=user_item_matrix.columns,
    index=user_item_matrix.index
)

print("✅ Predicted ratings matrix created!")

# Build Recommendation Function
def recommend_movies(user_id, num_recommendations=10):
    """Recommend movies for a specific user"""
    
    if user_id not in predicted_ratings_df.index:
        return f"❌ User {user_id} not found in dataset!"
    
    user_predictions = predicted_ratings_df.loc[user_id]
    user_rated = user_item_matrix.loc[user_id]
    user_rated_movies = user_rated[user_rated.notna()].index
    
    recommendations = user_predictions.drop(user_rated_movies)
    recommendations = recommendations.sort_values(ascending=False)
    top_recommendations = recommendations.head(num_recommendations)
    
    results = pd.DataFrame({
        'movie_id': top_recommendations.index,
        'predicted_rating': top_recommendations.values
    })
    
    results = results.merge(movies, on='movie_id', how='left')
    return results[['movie_id', 'title', 'genre', 'predicted_rating']]

print("✅ Recommendation function created!")

# Test with User ID 1
print(f"\n{'='*80}")
print(f"🎮 TESTING RECOMMENDATION SYSTEM - USER ID 1")
print(f"{'='*80}")

test_user_id = 1

# Show user's past ratings
user_past_ratings = ratings[ratings['user_id'] == test_user_id].merge(movies, on='movie_id')
print(f"\n📌 USER'S PAST RATINGS (showing first 10):")
print("-"*80)
for idx, row in user_past_ratings[['title', 'rating', 'genre']].head(10).iterrows():
    print(f"⭐ {row['rating']}/5 - {row['title']}")
    print(f"   Genre: {row['genre']}")

# Get recommendations
recommendations = recommend_movies(test_user_id, num_recommendations=10)

print(f"\n{'='*80}")
print(f"🌟 TOP 10 RECOMMENDED MOVIES FOR USER {test_user_id}:")
print(f"{'='*80}\n")

for idx, row in recommendations.iterrows():
    print(f"{idx+1}. {row['title']}")
    print(f"   📁 Genre: {row['genre']}")
    print(f"   ⭐ Predicted Rating: {row['predicted_rating']:.2f}/5.00")
    print()

# Test with another user
print(f"\n{'='*80}")
print(f"🎮 TESTING RECOMMENDATION SYSTEM - USER ID 25")
print(f"{'='*80}\n")

recommendations_25 = recommend_movies(25, num_recommendations=10)

for idx, row in recommendations_25.iterrows():
    print(f"{idx+1}. {row['title']}")
    print(f"   📁 Genre: {row['genre']}")
    print(f"   ⭐ Predicted Rating: {row['predicted_rating']:.2f}/5.00")
    print()

# Evaluate Performance
print(f"\n{'='*80}")
print("📊 MODEL PERFORMANCE EVALUATION")
print(f"{'='*80}")

from sklearn.metrics import mean_squared_error, mean_absolute_error

sample_size = 1000
sample_ratings = ratings.sample(n=sample_size, random_state=42)

actual = []
predicted = []

for _, row in sample_ratings.iterrows():
    user = row['user_id']
    movie = row['movie_id']
    
    if user in predicted_ratings_df.index and movie in predicted_ratings_df.columns:
        actual.append(row['rating'])
        predicted.append(predicted_ratings_df.loc[user, movie])

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print(f"\n✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"✅ MAE (Mean Absolute Error): {mae:.4f}")
print(f"\n💡 Lower values = Better predictions!")
print(f"💡 On average, predictions are off by ~{mae:.2f} stars")

print(f"\n{'='*80}")
print("🎉 RECOMMENDATION SYSTEM SUCCESSFULLY BUILT!")
print(f"{'='*80}")

print("\n📚 WHAT WE DID:")
print("1. ✅ Loaded movie ratings and movie information")
print("2. ✅ Created a User-Item Matrix (users × movies)")
print("3. ✅ Applied SVD to decompose matrix into U, Σ, V^T")
print("4. ✅ Predicted missing ratings by reconstructing the matrix")
print("5. ✅ Built recommendation function to suggest top movies")
print("6. ✅ Evaluated model performance (RMSE & MAE)")

print("\n🧠 KEY CONCEPTS LEARNED:")
print("- Collaborative Filtering (recommending based on similar users)")
print("- SVD (Singular Value Decomposition) for matrix factorization")
print("- Latent Factors (hidden patterns like 'action lover', 'comedy fan')")
print("- Matrix Completion (filling in missing ratings)")

print("\n🚀 You can now:")
print("- Recommend movies for any user ID")
print("- Predict ratings for unwatched movies")
print("- Understand how Netflix-style recommendations work!")

print(f"\n{'='*80}")
print("✨ CONGRATULATIONS! YOU'VE BUILT A RECOMMENDATION SYSTEM! ✨")
print(f"{'='*80}\n")
