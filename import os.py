import os
import zipfile
import pandas as pd
from tabulate import tabulate

# Para la validación de Kaggle lo voy a usar con Github Secrets pq no quiero que me doxxeen
kaggle_username = os.environ.get('KAGGLE_USUARIO')
kaggle_key = os.environ.get('KAGGLE_KEY')

os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

os.system('kaggle datasets download -d srgiomanhes/steam-games-dataset-2025')

with zipfile.ZipFile("steam-games-dataset-2025.zip", "r") as zip_ref:
    zip_ref.extractall("CSV")

df = pd.read_csv("CSV/steam_games.csv")
# Solo puse 10 porque mi laptop se traba con más :c
print(tabulate(df.head(10), headers='keys', tablefmt='grid'))
