import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import chain
import re

df = pd.read_csv("CSV/steam_games.csv")
df['name'] = df['name'].astype(str).str.strip()
df['review_score_desc'] = df['review_score_desc'].astype(str).str.strip().str.lower()
columns_to_fix = ['developers', 'publishers', 'categories', 'genres', 'platforms']
for col in columns_to_fix:
    df[col] = df[col].apply(eval)
df['review_score'] = df['review_score'].replace(0.0, np.nan)
df.rename(columns={"price_initial (USD)": "price_usd"}, inplace=True)

def clean_date(val):
    if isinstance(val, str):
        val = re.sub(r'\s*\d{1,2}:\d{2}:\d{2}\s*(a\.?\s*m\.?|p\.?\s*m\.?)', '', val, flags=re.IGNORECASE)
        val = val.strip()
        if val.lower() == "not released":
            return pd.NaT
        try:
            return pd.to_datetime(val, dayfirst=True, errors='coerce')
        except:
            return pd.NaT
    return pd.NaT

df['release_date'] = df['release_date'].apply(clean_date)
df['release_year'] = df['release_date'].dt.year

games_per_year = df['release_year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.barplot(x=games_per_year.index, y=games_per_year.values, color='skyblue')
plt.xticks(rotation=45)
plt.title("Número de juegos por año")
plt.xlabel("Año")
plt.ylabel("Cantidad de juegos")
plt.tight_layout()
plt.show()

all_genres = [genre for sublist in df['genres'].dropna() for genre in sublist]
genre_counts = Counter(all_genres).most_common(10)
labels, sizes = zip(*genre_counts)
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Top 10 géneros más comunes")
plt.axis('equal')
plt.show()
paid_games = df[(df['price_usd'].notna()) & (df['price_usd'] > 0)]
avg_price_year = paid_games.groupby('release_year')['price_usd'].mean()
plt.figure(figsize=(10, 5))
sns.lineplot(x=avg_price_year.index, y=avg_price_year.values, marker='o')
plt.title("Precio promedio de juegos por año (sin juegos gratis)")
plt.xlabel("Año")
plt.ylabel("Precio promedio (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

genre_price = []
for _, row in df[['genres', 'price_usd']].dropna().iterrows():
    if row['price_usd'] > 0:
        for genre in row['genres']:
            genre_price.append((genre, row['price_usd']))

genre_price_df = pd.DataFrame(genre_price, columns=['genre', 'price_usd'])
top_genres = genre_price_df['genre'].value_counts().nlargest(6).index
plt.figure(figsize=(12, 6))
sns.boxplot(x='genre', y='price_usd', data=genre_price_df[genre_price_df['genre'].isin(top_genres)])
plt.title("Distribución de precios por género (Top 6)")
plt.xlabel("Género")
plt.ylabel("Precio USD")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

platforms_flat = list(chain.from_iterable(df['platforms'].dropna()))
platform_counts = Counter(platforms_flat).most_common(6)
platform_names, counts = zip(*platform_counts)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(platform_names), y=list(counts), palette="viridis")
plt.title("Top 6 plataformas más comunes")
plt.xlabel("Plataforma")
plt.ylabel("Cantidad de juegos")
plt.tight_layout()
plt.show()
