from itertools import chain
import pandas as pd
from tabulate import tabulate
import numpy as np
import ast
import re
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
df = pd.read_csv("CSV/steam_games.csv")

df['name'] = df['name'].astype(str).str.strip()
df['review_score_desc'] = df['review_score_desc'].astype(str).str.strip().str.lower()
columns_to_fix = ['developers', 'publishers', 'categories', 'genres', 'platforms']
for col in columns_to_fix:
    df[col] = df[col].apply(ast.literal_eval)
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
#Ver las fechas de salida y el año con mayor cantidad de juegos lanzados 
df['release_date'] = df['release_date'].apply(clean_date)
df['release_year'] = df['release_date'].dt.year
games_per_year = df.groupby('release_year').size().reset_index(name='game_count')
print(games_per_year.sort_values('release_year'))
max_row = games_per_year.loc[games_per_year['game_count'].idxmax()]
max_year = int(max_row['release_year'])
max_games = int(max_row['game_count'])

print(f"El año con más juegos lanzados fue {max_year} con un total de {max_games} juegos.")

#Veremos la cant de generos, cual es el mas comun o popular

all_genres = [genre for sublist in df['genres'].dropna() for genre in sublist]
genre_counts = Counter(all_genres)
print("🎮 Géneros más comunes en el dataset:\n")
for genre, count in genre_counts.most_common():
    print(f"{genre}: {count} juegos")
most_common_genre, most_common_count = genre_counts.most_common(1)[0]
print(f"\nEl género más popular es: '{most_common_genre}' con {most_common_count} juegos.")
genres_by_year = defaultdict(list)
for _, row in df[['release_year', 'genres']].dropna().iterrows():
    year = int(row['release_year'])
    for genre in row['genres']:
        genres_by_year[year].append(genre)
print("Género más popular por año:\n")
for year in sorted(genres_by_year.keys()):
    counter = Counter(genres_by_year[year])
    most_common_genre, count = counter.most_common(1)[0]
    print(f"{year}: {most_common_genre} ({count} juegos)")

# Analizamos los precios por genero de steam y el promedio del costo de los juegos por año (aqui excluiremos demos o free to play)

genre_price = []
for _, row in df[['genres', 'price_usd']].dropna().iterrows():
    if row['price_usd'] > 0:  
        for genre in row['genres']:
            genre_price.append((genre, row['price_usd']))
genre_price_df = pd.DataFrame(genre_price, columns=['genre', 'price_usd'])
avg_price_per_genre = genre_price_df.groupby('genre')['price_usd'].mean().sort_values(ascending=False)
print("💵 Precio promedio por género (USD) [sin juegos gratis]:\n")
for genre, avg_price in avg_price_per_genre.items():
    print(f"{genre}: ${avg_price:.2f}")
paid_games = df[(df['price_usd'].notna()) & (df['price_usd'] > 0)]
avg_price_year = paid_games.groupby('release_year')['price_usd'].mean().sort_index()
print("\nPrecio promedio por año (USD) [sin juegos gratis]:\n")
for year, avg in avg_price_year.items():
    print(f"{int(year)}: ${avg:.2f}")

# Ahora veremos cuales son los desarrolladores con mayor cantidad de juegos y los de mayores seseñas positivas

df_filtered = df[['developers', 'total_positive']].dropna()
dev_positive_list = []
for _, row in df_filtered.iterrows():
    for dev in row['developers']:
        dev_positive_list.append((dev, row['total_positive']))
dev_positive_df = pd.DataFrame(dev_positive_list, columns=['developer', 'total_positive'])
top_dev_counts = dev_positive_df['developer'].value_counts().head(10)
print("Top 10 desarrolladores con más juegos:\n")
for dev, count in top_dev_counts.items():
    print(f"{dev}: {count} juegos")
top_dev_positive = dev_positive_df.groupby('developer')['total_positive'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 desarrolladores con más reseñas positivas acumuladas:\n")
for dev, total_pos in top_dev_positive.items():
    print(f"{dev}: {int(total_pos):,} reseñas positivas")

platforms_flat = list(chain.from_iterable(df['platforms'].dropna()))

# Contaremos ahora cuales son las plataformas mas usadas (puro team windows)
platform_counts = Counter(platforms_flat)
print("Plataformas más comunes:\n")
for platform, count in platform_counts.most_common():
    print(f"{platform}: {count} juegos")

#Veremos ahora cuantos juegos salen por mes para er los picos por temporada


df_valid_dates = df[df['release_date'].notna()].copy()
df_valid_dates['release_month'] = df_valid_dates['release_date'].dt.month
games_per_month = df_valid_dates['release_month'].value_counts().sort_index()
print("📅 Juegos lanzados por mes (total):\n")
for month, count in games_per_month.items():
    print(f"Mes {month:02d}: {count} juegos")

#Ahora veremos los juegos de pago mejor reseñados en cada año

df_valid = df[(df['release_date'].notna()) & 
              (df['price_usd'] > 0) & 
              (df['total_positive'] > 0)].copy()
df_valid['release_year'] = df_valid['release_date'].dt.year
top_game_by_year = df_valid.loc[df_valid.groupby('release_year')['total_positive'].idxmax()]
top_game_by_year = top_game_by_year.sort_values(by='release_year')
print("🏆 Mejor juego de pago por reseñas positivas cada año:\n")
for _, row in top_game_by_year.iterrows():
    print(f"{int(row['release_year'])} | {row['name'][:40]:40} | {int(row['total_positive']):,} reseñas")
top_paid_games = df_valid.sort_values(by='total_positive', ascending=False).head(10)
print("\n🌟 Top 10 juegos de pago con más reseñas positivas (global):\n")
for _, row in top_paid_games.iterrows():
    print(f"{row['name'][:40]:40} | Año: {int(row['release_year'])} | {int(row['total_positive']):,} reseñas")
#aa