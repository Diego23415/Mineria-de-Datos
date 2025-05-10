import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


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

df_clean = df[
    (df['price_usd'].notna()) &
    (df['release_year'].notna()) &
    (df['release_date'].notna()) &
    ~((df['is_free'] == True) & (df['price_usd'] == 0))
]


features = ['n_achievements', 'positive_percentual', 'review_score']
df_knn = df_clean[features + ['is_free']].dropna()

X = df_knn[features]
y = df_knn['is_free']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
