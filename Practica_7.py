import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CSV/steam_games.csv")

df['name'] = df['name'].astype(str).str.strip()
df['review_score_desc'] = df['review_score_desc'].astype(str).str.strip().str.lower()

columns_to_fix = ['developers', 'publishers', 'categories', 'genres', 'platforms']
for col in columns_to_fix:
    df[col] = df[col].apply(ast.literal_eval)

df['review_score'] = df['review_score'].replace(0.0, np.nan)
df.rename(columns={"price_initial (USD)": "price_usd"}, inplace=True)

df_clean = df[
    (df['price_usd'].notna()) &
    (df['positive_percentual'].notna()) &
    (df['price_usd'] <= 200) &
    (df['total_reviews'] > 0) &
    ~((df['is_free'] == True) & (df['price_usd'] == 0))
]

X = df_clean[['price_usd', 'positive_percentual']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(df_clean['price_usd'], df_clean['positive_percentual'], c=df_clean['cluster'], cmap='viridis')
plt.xlabel('Precio (USD)')
plt.ylabel('Porcentaje de Reseñas Positivas')
plt.title('Clustering de Juegos con Reseñas (> 0) y Precio (≤ 200 USD)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

print(df_clean[['name', 'price_usd', 'positive_percentual', 'total_reviews', 'cluster']].head())
