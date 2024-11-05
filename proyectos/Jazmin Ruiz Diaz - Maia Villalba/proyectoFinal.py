import numpy as np
import skfuzzy as fuzz
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import re

"""
MODULO 1: preproceso de datos
"""

datos = pd.read_csv('test_data.csv')

# Definir la función de preprocesamiento
def preprocesar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Corregir contracciones descompuestas
    texto = re.sub(r'\bwon t\b', 'will not', texto)
    texto = re.sub(r'\bcan t\b', 'cannot', texto)
    texto = re.sub(r'\bdon t\b', 'do not', texto)
    texto = re.sub(r'\bdidn t\b', 'did not', texto)
    texto = re.sub(r'\bdoesn t\b', 'does not', texto)
    texto = re.sub(r'\bhasn t\b', 'has not', texto)
    texto = re.sub(r'\bhadn t\b', 'had not', texto)
    texto = re.sub(r'\bshouldn t\b', 'should not', texto)
    texto = re.sub(r'\bwouldn t\b', 'would not', texto)
    texto = re.sub(r'\bcouldn t\b', 'could not', texto)
    texto = re.sub(r'\baren t\b', 'are not', texto)
    texto = re.sub(r'\bisn t\b', 'is not', texto)
    texto = re.sub(r'\bwasn t\b', 'was not', texto)
    texto = re.sub(r'\bweren t\b', 'were not', texto)
    texto = re.sub(r'\bain t\b', 'is not', texto)
    
    # Eliminar URLs
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto, flags=re.MULTILINE)
    
    # Eliminar menciones (@usuario)
    texto = re.sub(r'@\w+', '', texto)
    
    # Eliminar hashtags, dejando solo el texto del hashtag
    texto = re.sub(r'#(\w+)', r'\1', texto)
    
    # Reducir letras repetidas a máximo dos letras consecutivas
    texto = re.sub(r'(.)\1+', r'\1\1', texto)
    
    # Eliminar caracteres especiales y signos de puntuación
    texto = re.sub(r"[^\w\s]", '', texto)
    
    # Eliminar espacios adicionales
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

datos['sentence'] = datos['sentence'].apply(preprocesar_texto)

"""
MODULO 2, 3, 4, 5 y 6
"""

# Declaracion e inicializacion de variables a utilizar y funciones de pertenencua triangulares. Variables difusas
datos['TweetPos'] = 0
datos['TweetNeg'] = 0
datos['Sentiment_Score'] = 0
datos['Sentiment_Label'] = ""
datos['Execution_Time'] = 0


puntajes_salida  = []

x_p = np.arange(0, 1.1, 0.1)  # Positivos: [0, 1]
x_n = np.arange(0, 1.1, 0.1)  # Negativos: [0, 1]
x_op = np.arange(0, 10, 1)    # salida: [0, 10]

p_bajo = fuzz.trimf(x_p, [0, 0, 0.5])
p_medio = fuzz.trimf(x_p, [0, 0.5, 1]) 
p_alto = fuzz.trimf(x_p, [0.5, 1, 1])

n_bajo = fuzz.trimf(x_n, [0, 0, 0.5])
n_medio = fuzz.trimf(x_n, [0, 0.5, 1])
n_alto = fuzz.trimf(x_n, [0.5, 1, 1])

# Funciones de pertenencia triangulares para la salida (op: Negativo, Neutral, Positivo)
op_Neg = fuzz.trimf(x_op, [0, 0, 5])
op_Neu = fuzz.trimf(x_op, [0, 5, 10])
op_Pos = fuzz.trimf(x_op, [5, 10, 10])

# Variables para benchmarks
c_positivos = 0
c_neutrales = 0
c_negativos = 0
tiempo_total = 0

sid = SentimentIntensityAnalyzer()

# Procesar cada tweet en el datosset
for i in range(len(datos)):
    inicio_tiempo = time.time()  # Marcar inicio del tiempo para el tweet
    
    # Obtener el texto del tweet
    text = datos.loc[i, 'sentence']
    
    # Calcular puntajes de sentimiento usando VADER
    puntajes = sid.polarity_scores(text)
    valor_pos  = round(puntajes['pos'], 2)
    valor_neg  = round(puntajes['neg'], 2)
    

    datos.loc[i, 'TweetPos'] = valor_pos
    datos.loc[i, 'TweetNeg'] = valor_neg
    
    # Fuzzificar los puntajes de sentimiento
    pos_bajo = fuzz.interp_membership(x_p, p_bajo, valor_pos )
    pos_medio = fuzz.interp_membership(x_p, p_medio, valor_pos )
    pos_alto = fuzz.interp_membership(x_p, p_alto, valor_pos )

    neg_bajo = fuzz.interp_membership(x_n, n_bajo, valor_neg )
    neg_medio = fuzz.interp_membership(x_n, n_medio, valor_neg )
    neg_alto = fuzz.interp_membership(x_n, n_alto, valor_neg )

    # Reglas según las ecuaciones (15) a (23)
    w_r1 = np.fmin(pos_bajo , neg_bajo)  # Regla 1: Bajo, Bajo -> Neutral
    w_r2 = np.fmin(pos_medio, neg_bajo)  # Regla 2: Medio, Bajo -> Positivo
    w_r3 = np.fmin(pos_alto, neg_bajo)  # Regla 3: Alto, Bajo -> Positivo
    w_r4 = np.fmin(pos_bajo , neg_medio)  # Regla 4: Bajo , Medio -> Negativo
    w_r5 = np.fmin(pos_medio, neg_medio)  # Regla 5: Medio, Medio -> Neutral
    w_r6 = np.fmin(pos_alto, neg_medio)  # Regla 6: Alto, Medio -> Positivo
    w_r7 = np.fmin(pos_bajo , neg_alto)  # Regla 7: Bajo, Alto -> Negativo
    w_r8 = np.fmin(pos_medio, neg_alto)  # Regla 8: Medio, Alto -> Negativo
    w_r9 = np.fmin(pos_alto, neg_alto)  # Regla 9: Alto, Alto -> Neutral

    # Reglas para cada salida
    w_neg = np.fmax(w_r4, np.fmax(w_r7, w_r8))
    w_neu = np.fmax(w_r1, np.fmax(w_r5, w_r9))
    w_pos = np.fmax(w_r2, np.fmax(w_r3, w_r6))

    # Reglas de salida (op_Neg, op_Neu, op_Pos)
    op_activation_bajo = np.fmin(w_neg, op_Neg)
    op_activation_medio = np.fmin(w_neu, op_Neu)
    op_activation_alto = np.fmin(w_pos, op_Pos)


    # Activaciones de las funciones de salida
    aggregated = np.fmax(op_activation_bajo, np.fmax(op_activation_medio, op_activation_alto))

    # Defuzzificación con metodo del centro de area
    salida = round(fuzz.defuzz(x_op, aggregated, 'centroid'), 2)
    puntajes_salida.append(salida)

    # Clasificación del sentimiento según el valor defuzzificado
    if 0 < salida < 3.33:
        sentiment_label = "Negativo"
        c_negativos += 1
    elif 3.33 <= salida < 6.67:
        sentiment_label = "Neutral"
        c_neutrales += 1
    elif 6.67 <= salida <= 10:
        sentiment_label = "Positivo"
        c_positivos += 1
    

    datos.loc[i, 'Sentiment_Score'] = salida
    datos.loc[i, 'Sentiment_Label'] = sentiment_label
    
    # Fin del tiempo
    execution_time = round(time.time() - inicio_tiempo, 4)
    datos.loc[i, 'Execution_Time'] = execution_time
    tiempo_total += execution_time


datos.to_csv('dataset_con_benchmarks.csv', index=False)

print(f"Tweets positivos: {c_positivos}")
print(f"Tweets neutrales: {c_neutrales}")
print(f"Tweets negativos: {c_negativos}")
print(f"Tiempo total de ejecución: {round(tiempo_total, 4)} segundos")
print(f"Tiempo promedio de ejecución por tweet: {round(tiempo_total / len(datos), 4)} segundos")

print(datos)