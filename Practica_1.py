import os
import zipfile
import pandas as pd
from tabulate import tabulate
import numpy as np
import ast
# Para la validación de Kaggle lo voy a usar con Github Secrets pq no quiero que me doxxeen
kaggle_username = os.environ.get('KAGGLE_USUARIO')
kaggle_key = os.environ.get('KAGGLE_KEY')

os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

os.system('kaggle datasets download -d srgiomanhes/steam-games-dataset-2025')

with zipfile.ZipFile("steam-games-dataset-2025.zip", "r") as zip_ref:
    zip_ref.extractall("CSV")

df = pd.read_csv("CSV/steam_games.csv")
df['name'] = df['name'].astype(str).str.strip()
df['review_score_desc'] = df['review_score_desc'].astype(str).str.strip().str.lower()
columns_to_fix = ['developers', 'publishers', 'categories', 'genres', 'platforms']
for col in columns_to_fix:
    df[col] = df[col].apply(ast.literal_eval)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date'])
df['review_score'] = df['review_score'].replace(0.0, np.nan)
df.rename(columns={"price_initial (USD)": "price_usd"}, inplace=True)
# Solo puse 10 porque mi laptop se traba con más :c
print(df.head(10).to_csv(index=False))

