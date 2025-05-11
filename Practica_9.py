import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast

df = pd.read_csv("CSV/steam_games.csv")
df['genres'] = df['genres'].apply(ast.literal_eval)

genre_list = df['genres'].dropna().tolist()
all_genres = ' '.join([' '.join(g) for g in genre_list])

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(all_genres)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud de Géneros de Juegos en Steam")
plt.show()
