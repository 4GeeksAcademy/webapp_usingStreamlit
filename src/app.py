import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

# Usando rutas relativas
model_path = os.path.join(os.getcwd(), 'models', 'knn_neighbors-6_algorithm-brute_metric-cosine.sav')
modelo = joblib.load(model_path)

vectorizer_path = os.path.join(os.getcwd(), 'models', 'tfidf_vectorizer.sav')
vectorizer = joblib.load(vectorizer_path)

# Cargar tus datos
df_done = pd.read_pickle('df_done.pkl')

# Asegurarse de que las columnas estén en el formato adecuado
if 'tags' not in df_done.columns:
    # Combinar las columnas en una sola columna llamada 'tags'
    df_done['tags'] = df_done['genres'] + df_done['keywords'] + df_done['cast'] + df_done['crew']

    # Asegurarse de que 'overview' sea una lista de palabras
    df_done['overview'] = df_done['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Concatenar todas las listas en una sola lista
    df_done['tags'] = df_done['tags'] + df_done['overview']

    # Convertir la lista de tags en una cadena de texto separada por espacios
    df_done['tags'] = df_done['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# Obtener la lista de títulos de películas
titulos = df_done['title'].tolist()

st.title("🎬 Recomendador de Películas")

pelicula_seleccionada = st.selectbox("Selecciona una película:", titulos)

if st.button("Recomendar"):
    # Encontrar el índice de la película seleccionada
    idx = df_done[df_done['title'] == pelicula_seleccionada].index[0]
    
    # Obtener la etiqueta de la película seleccionada
    etiqueta = df_done.iloc[idx]['tags']
    
    # Vectorizar la etiqueta
    etiqueta_vectorizada = vectorizer.transform([etiqueta])
    
    # Buscar los k vecinos más cercanos
    distancias, indices = modelo.kneighbors(etiqueta_vectorizada, n_neighbors=6)  # 6 porque la primera será la misma

    st.subheader("Películas recomendadas:")
    for i in range(1, len(indices[0])):  # Saltamos la primera porque es la misma película
        st.write(df_done.iloc[indices[0][i]]['title'])
