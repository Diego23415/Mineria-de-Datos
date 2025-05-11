import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate
import ast
import re
from statsmodels.stats.outliers_influence import summary_table

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

df['release_date'] = df['release_date'].apply(clean_date)
df['release_year'] = df['release_date'].dt.year

df_clean = df[(df['price_usd'].notna()) & (df['release_year'].notna())]
df_clean = df_clean[(df_clean['price_usd'] < 100) & (df_clean['release_year'] >= 2000)]
df_clean = df_clean[df_clean['release_date'].notna()]
df_clean = df_clean[~((df_clean['is_free'] == True) & (df_clean['price_usd'] == 0))]

df_grouped = df_clean.groupby('release_year')['price_usd'].mean().reset_index()

X = sm.add_constant(df_grouped['release_year'])
y = df_grouped['price_usd']

model = sm.OLS(y, X).fit()

future_years = pd.DataFrame({'release_year': np.arange(2025, 2031)})
X_future = sm.add_constant(future_years)

predictions = model.get_prediction(X_future)
summary_frame = predictions.summary_frame(alpha=0.05)

future_years['predicted_price'] = summary_frame['mean']
future_years['ci_lower'] = summary_frame['mean_ci_lower']
future_years['ci_upper'] = summary_frame['mean_ci_upper']

print("\n📊 Predicción de precios promedio (USD) por año:\n")
print(tabulate(future_years, headers='keys', tablefmt='pretty', showindex=False))

plt.figure(figsize=(10, 6))

plt.plot(df_grouped['release_year'], y, 'o', label="Histórico", color='blue')

combined_years = pd.concat([df_grouped[['release_year']], future_years[['release_year']]])
combined_X = sm.add_constant(combined_years)
combined_pred = model.predict(combined_X)

plt.plot(combined_years['release_year'], combined_pred, color='red', label="Regresión lineal")

plt.fill_between(future_years['release_year'],
                 future_years['ci_lower'],
                 future_years['ci_upper'],
                 color='orange', alpha=0.3, label='95% IC')

price_mean = y.mean()
plt.axhline(price_mean, color='green', linestyle='--', label=f"Media Global: {price_mean:.2f} USD")

plt.title("Forecast: Precio promedio de juegos por año")
plt.xlabel("Año de lanzamiento")
plt.ylabel("Precio promedio (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
