import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ast
import re
from tabulate import tabulate

df = pd.read_csv("CSV/steam_games.csv")
#Se hara una correlacion entre precio y el año se omitiran los juegos gratis o sin fecha de lanzamiento
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

df['release_date'] = df['release_date'].apply(clean_date)
df['release_year'] = df['release_date'].dt.year

df_clean = df[(df['price_usd'].notna()) & (df['release_year'].notna())]
df_clean = df_clean[(df_clean['price_usd'] < 100) & (df_clean['release_year'] >= 2000)]

df_clean = df_clean[df_clean['release_date'].notna()]  
df_clean = df_clean[~((df_clean['is_free'] == True) & (df_clean['price_usd'] == 0))] 

X = df_clean[['release_year']]
y = df_clean['price_usd']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())

y_pred = model.predict(X)

price_mean = y.mean()

plt.figure(figsize=(10, 6))
plt.scatter(df_clean['release_year'], y, alpha=0.3, label="Datos Reales")
plt.plot(df_clean['release_year'], y_pred, color='red', label="Línea de Regresión")
plt.axhline(price_mean, color='green', linestyle='--', label=f"Media del Precio (USD): {price_mean:.2f}")
plt.xlabel("Año de lanzamiento")
plt.ylabel("Precio (USD)")
plt.title("Regresión Lineal: Año de Lanzamiento vs Precio")
plt.legend()
plt.grid(True)
plt.show()

r2 = model.rsquared
print(f"R² Score: {r2:.4f}")
