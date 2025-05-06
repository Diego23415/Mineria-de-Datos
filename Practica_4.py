from scipy.stats import f_oneway
import pandas as pd
from collections import Counter
import ast
import re

df = pd.read_csv("CSV/steam_games.csv")
df['name'] = df['name'].astype(str).str.strip()
df['review_score_desc'] = df['review_score_desc'].astype(str).str.strip().str.lower()
columns_to_fix = ['developers', 'publishers', 'categories', 'genres', 'platforms']
for col in columns_to_fix:
    df[col] = df[col].apply(ast.literal_eval)
df['review_score'] = df['review_score'].replace(0.0, pd.NA)
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
all_genres = [genre for sublist in df['genres'].dropna() for genre in sublist]
genre_counts = Counter(all_genres)
top_genres = [genre for genre, _ in genre_counts.most_common(5)]
df_top_genres = df[
    (df['genres'].notna()) &
    (df['price_usd'].notna()) &
    (df['price_usd'] > 0)
].copy()
df_top_genres['main_genre'] = df_top_genres['genres'].apply(
    lambda genres: next((g for g in genres if g in top_genres), None)
)
df_top_genres = df_top_genres[df_top_genres['main_genre'].notna()]
grouped_prices = [group['price_usd'].values for _, group in df_top_genres.groupby('main_genre')]
f_stat, p_val = f_oneway(*grouped_prices)
print("Hipótesis inicial:")
print("No hay diferencias significativas en el precio promedio entre los géneros principales de los juegos.")
print("\nResultados del ANOVA:")
print(f"ANOVA F-Statistic: {f_stat:.4f}")
print(f"p-value: {p_val:.4f}")
alpha = 0.05
print("\nConclusión:")
if p_val < alpha:
    print("La hipótesis es rechazada. Existe una diferencia estadísticamente significativa en el precio promedio entre los géneros analizados.")
    print("   Esto sugiere que el género principal de los juegos afecta significativamente su precio.")
else:
    print("No se rechazó la hipótesis. No se encontró una diferencia significativa en el precio promedio entre los géneros.")
    print("   No hay evidencia suficiente para afirmar que el género afecta el precio de los juegos.")
print("El test ANOVA compara las medias de varios grupos (en este caso, géneros) para determinar si existe una diferencia significativa entre ellos.")
print("Si el p-value es menor que el nivel de significancia (alpha), rechazamos la hipótesis nula, lo que indica que al menos un grupo tiene una media significativamente diferente.")
print("Si el p-value es mayor que alpha, no hay suficiente evidencia para rechazar la hipótesis nula y concluimos que las medias son similares.")
