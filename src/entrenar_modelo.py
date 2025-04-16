import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Cargar el DataFrame
df_done = pd.read_pickle('df_done.pkl')

# Verificar y convertir las columnas a listas si no lo son
columns_to_check = ['genres', 'keywords', 'cast', 'crew']
for col in columns_to_check:
    if not all(isinstance(x, list) for x in df_done[col]):
        df_done[col] = df_done[col].apply(lambda x: x if isinstance(x, list) else [])

# Combinar las columnas en una sola columna llamada 'tags'
df_done['tags'] = df_done['genres'] + df_done['keywords'] + df_done['cast'] + df_done['crew']

# Asegurarse de que 'overview' sea una lista de palabras
df_done['overview'] = df_done['overview'].apply(lambda x: x if isinstance(x, list) else [])

# Concatenar todas las listas en una sola lista
df_done['tags'] = df_done['tags'] + df_done['overview']

# Convertir la lista de tags en una cadena de texto separada por espacios
df_done['tags'] = df_done['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# Verificar la columna 'tags' después de la combinación
print(df_done['tags'].head())

# Verificar si hay valores nulos o vacíos en la columna 'tags'
print(df_done['tags'].isnull().sum())
print(df_done['tags'].str.len().describe())

# Verificar el tipo de datos en la columna 'tags'
print(df_done['tags'].apply(type).value_counts())

# Configurar el TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=None, max_df=0.95, min_df=2, max_features=5000)

# Vectorizar los datos
X = vectorizer.fit_transform(df_done['tags'])

# Verificar el vocabulario
print(vectorizer.get_feature_names_out()[:10])  # Verificar las primeras 10 palabras en el vocabulario

# Verificar la forma de la matriz X
print(X.shape)

# Entrenar el modelo
modelo = NearestNeighbors(metric='cosine', algorithm='brute')
modelo.fit(X)

# Guardar el modelo y el vectorizador
joblib.dump(modelo, '/workspaces/webapp_usingStreamlit/models/knn_neighbors-6_algorithm-brute_metric-cosine.sav')
joblib.dump(vectorizer, '/workspaces/webapp_usingStreamlit/models/tfidf_vectorizer.sav')