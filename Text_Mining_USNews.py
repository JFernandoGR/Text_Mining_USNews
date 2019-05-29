# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:16:22 2019

@author: Jairo Fernando Gudiño-Rosero.
"""
#####

# "Los modelos más simples superan generalmente a los más complejos". Nassim Taleb.
# "Con exceso de información, es demasiado fácil obtener relaciones espúreas.
# el éxito de la ciencia radica en saber qué ignorar y no tanto en qué aprender". Yaneer Bar-Yam.

# En las siguientes líneas de código se muestra el desarrollo de la prueba
# técnica.

# Debido a que los kernels de la página de Kaggle son vastos en código y en
# aplicación de algoritmos de clasificación, en el presente archivo NO se
# enfoca en la aplicación de algoritmos distintos sino en la obtención de 
# conclusiones de interés o insights, además de que se desarrollan códigos SIN COPIA.
# Esto, con el objetivo de ser original e innovador en resultados presentados.

######



# Carga de librerías

# Generales
import json
import pandas as pd 
import requests
import matplotlib.pyplot as plt
import numpy as np
import re
import unicodedata
import nltk
import random

# Graficación
import seaborn as sns

# Importar libreria de webscrapping
from bs4 import BeautifulSoup

# Importación de librerías para analizar sentimientos.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Importación de librerías de pre-procesamiento de texto
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Instalación del paquere wordlcloud, siguiendo el tutorial
# https://datatofish.com/install-package-python-using-pip/
# la ruta del Python Script está en
import sys
print(sys.executable)
# Importación del paquete final
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# Librerías de Machine Learning
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold

# Definición de funciones de interés.
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    # Stemmer simplemente elimina los últimos caracteres de las palabras
    # para formar palabras.
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    # La lematización considera el contexto y convierte las palabras a sus formas
    # base.
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

def important_words(words):
    text = ' '.join(string for string in words)
    text = text.replace('movie', '')
    text = text.replace('film', '')
    text = text.replace('new', '')
    text = text.replace('actor', '')
    text = text.replace('time', '')
    text = text.replace('people', '')
    text = text.replace('fan', '')
    text = text.replace('one', '')
    text = text.replace('say', '')
    text = text.replace('first', '')
    text = text.replace('year', '')
    text = text.replace('video', '')
    text = text.replace('trailer', '')
    text = text.replace('star', '')
    text = text.replace('song', '')
    text = text.replace('life', '')
    text = text.replace('best', '')
    text = text.replace('music', '')
    text = text.replace('day', '')
    text = text.replace('nt', '')
    text = text.replace('thing', '')
    text = text.replace('singer', '')
    text = text.replace('every', '')
    text = text.replace('director', '')
    text = text.replace('us', '')
    text = text.replace('way', '')
    text = text.replace('ing', '')
    text = text.replace('two', '')
    text = text.replace('album', '')
    text = text.replace('plunded', '')
    text = text.replace('world', '')
    text = text.replace('actress', '')
    text = text.replace('box', '')
    text = text.replace('office', '')
    text = text.replace('show', '')
    text = text.replace('weekend', '')
    text = text.replace('week', '')
    text = text.replace('season', '')
    text = text.replace('game', '')
    text = text.replace('talk', '')
    text = text.replace('night', '')
    text = text.replace('man', '')
    text = text.replace('women', '')
    text = text.replace('game', '')
    text = text.replace('girl', '')
    text = text.replace('watch', '')
    text = text.replace('tv', '')
    text = text.replace('play', '')
    text = text.replace('look', '')
    text = text.replace('photo', '')
    text = text.replace('character', '')
    text = text.replace('story', '')
    text = text.replace('hollywood', '')
    text = text.replace('perforce', '')
    text = text.replace('share', '')
    text = text.replace('hotel', '')
    text = text.replace('city', '')
    text = text.replace('place', '')
    text = text.replace('trip', '')
    text = text.replace('traveler', '')
    text = text.replace('travel', '')
    text = text.replace('home', '')
    text = text.replace('tip', '')
    text = text.replace('reason', '')
    text = text.replace('list', '')
    text = text.replace('vacation', '')
    text = text.replace('destination', '')
    text = text.replace('visit', '')
    text = text.replace('town', '')
    text = text.replace('tourist', '')
    text = text.replace('experience', '')
    text = text.replace('coury', '')
    text = text.replace('day', '')
    text = text.replace('change', '')
    text = text.replace('world', '')
    text = text.split()
    return text

# Preprocesamiento
# Se extrae en forma de lista los titulares
titulares = []
for line in open('C:/Users/Python/Desktop/News_Category_Dataset_v2.json', 'r'):
    titulares.append(json.loads(line))
# Se integran todos los objetos en un solo data-frame
titularesDF = pd.DataFrame(titulares) 

# A continuación se resuelven las preguntas.

# 1. ¿Qué información útil se puede extraer de los datos? ###########################

# Se puede extraer información de cuáles son las palabras más
# comunes por categoría de noticias y por autor.
# NOTA: Se creó una función important_words desde scratch para extraer info relevante.


#(i) CATEGORÍAS DE NOTICIAS
# groupby agrupa valores por nombre de columna. Size especifica el número y
# se muestran valores de mayor a menor.
noticias_cat = titularesDF.groupby('category').size().sort_values(ascending=False)
noticias_cat

# Graficar categorías de noticias.
cmapper = ['lightskyblue', 'gold', 'r', 'lightcoral', 'm', 'yellowgreen', 'k']
f, ax = plt.subplots(figsize=(25,15))
# Se presenta el top 10.
titularesDF['category'].value_counts().head(10).sort_values(ascending=False).plot.bar(color=cmapper)
plt.xticks(rotation=50 , fontsize = 20)
plt.xlabel("Categorías")
plt.ylabel("Número de artículos", fontsize = 20)
plt.title("Categorías más populares", fontsize = 25)
plt.savefig('C:/Users/Python/Desktop/categories_total.jpeg', dpi=300, bbox_inches='tight')
plt.show()


Categories = np.unique(titularesDF["category"])

# Extracción de contenido de noticias
# Se crea un código para construir wordclouds de interés.
# No se crea un for porque se hizo un fuerte enfoque en eliminar 
# palabras poco ùtiles para generar insights en las principales categorías, y
# se consideró esto como prioritario a simplificar código por cuestiones de tiempo.

# Word Clouds: 

# A. Politics
c = 24
cat_titulares_politics = titularesDF[titularesDF["category"]==Categories[c]]
# Se integra el título de la noticia con la descripción.
cat_titulares_politics['Text']=cat_titulares_politics['headline'] + " " + (cat_titulares_politics['short_description'])
text = ' '.join(string for string in cat_titulares_politics['Text'])
# Se tokeniza el texto obtenido:
words = nltk.word_tokenize(text) # output words es una lista.
# Se hace normalización:
words = normalize(words)
politics_text = important_words(words)

# Se extraen sólo nouns con el fin de detallar mejor el contenido
politics_text = nltk.pos_tag(politics_text)
words_politics = pd.DataFrame(politics_text)
words_politics.columns = ['word','type']
words_politics = words_politics[words_politics.type.str.startswith('N')]
words_politics = ' '.join(string for string in words_politics['word'])

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(words_politics)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# Se puede ver que los temas alrededor de Donald Trump, Hillary Clinton y el partido republicano
# son los más recurrentes en las noticias de política.
# Haciendo stemming el word cloud no cambia significativamente.
stems, lemmas = stem_and_lemmatize(words_politics)
stems_text = ' '.join(string for string in stems)
wordcloud = WordCloud().generate(stems_text)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# B. Wellness
c = 37
cat_titulares_wellness = titularesDF[titularesDF["category"]==Categories[c]]
# Se integra el título de la noticia con la descripción.
cat_titulares_wellness['Text']=cat_titulares_wellness['headline'] + " " + (cat_titulares_wellness['short_description'])
# Se colapsan las celdas
text = ' '.join(string for string in cat_titulares_wellness['Text'])
# Se tokeniza el texto obtenido:
words = nltk.word_tokenize(text)
# Se hace normalización:
words = normalize(words)
wellness_text = important_words(words)

# Se extraen sólo nouns con el fin de detallar mejor el contenido
wellness_text = nltk.pos_tag(wellness_text)
words_wellness = pd.DataFrame(wellness_text)
words_wellness.columns = ['word','type']
words_wellness = words_wellness[words_wellness.type.str.startswith('N')]
words_wellness = ' '.join(string for string in words_wellness['word'])

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(words_wellness)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# Se puede ver que los temas alrededor de salud (ejercicio, cerebro, cancer), estudio y trabajo
# son los más recurrentes en las noticias de bienestar.

# C. Entretenimiento
c = 10
cat_titulares_entertainment = titularesDF[titularesDF["category"]==Categories[c]]
# Se integra el título de la noticia con la descripción.
cat_titulares_entertainment['Text']=cat_titulares_entertainment['headline'] + " " + (cat_titulares_entertainment['short_description'])
text = ' '.join(string for string in cat_titulares_entertainment['Text'])
# Se tokeniza el texto obtenido:
words = nltk.word_tokenize(text)
# Se hace normalización:
words = normalize(words)
# Se extraen sólo palabras importantes
entertainment_text = important_words(words)

# Se extraen sólo nouns con el fin de detallar mejor el contenido
entertainment_text = nltk.pos_tag(entertainment_text)
words_entertainment = pd.DataFrame(entertainment_text)
words_entertainment.columns = ['word','type']
words_entertainment = words_entertainment[words_entertainment.type.str.startswith('N')]
words_entertainment = ' '.join(string for string in words_entertainment['word'])

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(words_entertainment)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# Se puede observar que los nombres más importantes son Trump,
# Beyonce y Taylor Swift. Por temáticas de entretenimiento,
# los relacionados con muerte, guerra y familia son los más importantes,
# Los premios Oscar y las series son tópicos importantes también.



# D. Travel
c = 34
cat_titulares_travel = titularesDF[titularesDF["category"]==Categories[c]]
# Se integra el título de la noticia con la descripción.
cat_titulares_travel['Text']=cat_titulares_travel['headline'] + " " + (cat_titulares_travel['short_description'])
text = ' '.join(string for string in cat_titulares_travel['Text'])
# Se tokeniza el texto obtenido:
words = nltk.word_tokenize(text)
# Se hace normalización:
words = normalize(words)
travel_text = important_words(words)

# Se extraen sólo nouns con el fin de detallar mejor el contenido
travel_text = nltk.pos_tag(travel_text)
words_travel = pd.DataFrame(travel_text)
words_travel.columns = ['word','type']
words_travel = words_travel[words_travel.type.str.startswith('N')]
words_travel = ' '.join(string for string in words_travel['word'])

wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white").generate(words_travel)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# Se puede observar que los temas más importantes en las noticas de viajes
# están relacionadas con viajes por avión (playas, islas) y en  menor medida con
# carreteras

# (ii) CATEGORÍAS DE AUTORES

authors_names = titularesDF.groupby('authors').size().sort_values(ascending=False)
authors_names.head(20)
authors_names.head(350).sum()/authors_names.sum()
350/authors_names.count()
# Existe una alta concentración en el número de artículos escritos por autor.
# Un poco más del 60% está escrito por el 1.25% del total de autores (350).


# TOP 20 DE AUTORES
cmapper = ['lightskyblue', 'gold', 'r', 'lightcoral', 'm', 'yellowgreen', 'k']
f, ax = plt.subplots(figsize=(25,15))
authors_top = pd.DataFrame(authors_names.head(20), columns = ["authors"])
authors_top = pd.DataFrame(authors_top.index, columns = ["authors"]).loc[1:]
authors_names_top = pd.merge(titularesDF, authors_top, on='authors', how='inner')
authors_names_top["authors"].value_counts().sort_values(ascending=False).plot.bar(color=cmapper)
plt.xticks(rotation=50 , fontsize = 20)
plt.xlabel("Autores")
plt.ylabel("Número de artículos")
plt.savefig('C:/Users/Python/Desktop/authors_categories.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# WORDCLOUDS PARA AUTORES

# A. Lee Moran
lee_moran_titulares = titularesDF[titularesDF["authors"].str.contains("Lee Moran")]
# Se integra el título de la noticia con la descripción.
lee_moran_titulares['Text']=lee_moran_titulares['headline'] + " " + (lee_moran_titulares['short_description'])
text = ' '.join(string for string in lee_moran_titulares['Text'])
# Se tokeniza el texto obtenido:
words = nltk.word_tokenize(text)
# Se hace normalización:
words = normalize(words)
lee_moran_text = important_words(words)

# Se extraen sólo nouns con el fin de detallar mejor el contenido
lee_moran_text = nltk.pos_tag(lee_moran_text)
words_lee_moran = pd.DataFrame(lee_moran_text)
words_lee_moran.columns = ['word','type']
words_lee_moran = words_lee_moran[words_lee_moran.type.str.startswith('N')]
words_lee_moran = ' '.join(string for string in words_lee_moran['word'])

wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white").generate(words_lee_moran)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

Words_Number = Counter(words_lee_moran.split()).most_common()


# B. Ron Dicker
ron_dicker_titulares = titularesDF[titularesDF["authors"].str.contains("Ron Dicker")]
ron_dicker_titulares['Text']=ron_dicker_titulares['headline'] + " " + (ron_dicker_titulares['short_description'])
text = ' '.join(string for string in ron_dicker_titulares['Text'])
# Se tokeniza el texto obtenido:
words = nltk.word_tokenize(text)
# Se hace normalización:
words = normalize(words)
ron_dicker_text = important_words(words)
# Se extraen sólo nouns con el fin de detallar mejor el contenido
ron_dicker_text = nltk.pos_tag(ron_dicker_text)
words_ron_dicker = pd.DataFrame(ron_dicker_text)
words_ron_dicker.columns = ['word','type']
words_ron_dicker = words_ron_dicker[words_ron_dicker.type.str.startswith('N')]
words_ron_dicker = ' '.join(string for string in words_ron_dicker['word'])

wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white").generate(words_ron_dicker)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

Counter(words_ron_dicker.split()).most_common()


# 2. ¿Existen estilos de escritura asociados a cada categoría? #########################
# Se realiza un análisis de sentimiento por categoría.

sid = SentimentIntensityAnalyzer()

titularesDF['Text']=titularesDF['headline'] + " " + (titularesDF['short_description'])
Categories = np.unique(titularesDF["category"])
category_scores = []

for c in range(len(Categories)):
  cat_titulares = titularesDF[titularesDF["category"]==Categories[c]]
  corpus = cat_titulares['Text']
  corpus = corpus.apply(word_tokenize)
  corpus = corpus.apply(normalize)
  corpus = corpus.tolist()
  corpus = [' '.join(x) for x in corpus]
  total_scores = []
  for t in range(len(corpus)):
      scores = sid.polarity_scores(corpus[t])
      total_scores.append(scores)
  total_scores = pd.DataFrame(total_scores)
  category_scores.append(total_scores.mean().to_dict())

category_scores = pd.DataFrame(category_scores)
category_scores.index = Categories
category_scores = category_scores.sort_values("compound",ascending=False)

y_pos = np.arange(len(Categories))

f, ax = plt.subplots(figsize=(8,10))
plt.barh(category_scores.index, category_scores["compound"], align='center', alpha=1, color= 'lightblue')
plt.ylabel('Categoría')
plt.title('Puntaje Promedio')
plt.savefig('C:/Users/Python/Desktop/categories.jpeg', dpi=300, bbox_inches='tight')
plt.show()



# 3. ¿Se pueden catalogar las noticias con la descripción y los titulares? Compare su clasificación con las categorías incluidas en el set de datos.
# Se utiliza aquí como algoritmo de clasificación Support Vector Machine con kernel lineal.


# Preprocesamiento.

lemmatizer = WordNetLemmatizer()

general_corpus = titularesDF['Text']
general_corpus = general_corpus.apply(word_tokenize)
general_corpus = general_corpus.apply(normalize)

general_corpus2 = general_corpus.tolist()
general_corpus2 = [' '.join(x) for x in general_corpus2]
general_corpus2 = [lemmatizer.lemmatize(w) for w in general_corpus2]

classification_data = pd.concat([titularesDF['category'].to_frame(), pd.DataFrame(general_corpus2)], axis=1)
classification_data.columns = ["category","text"]


classification_data_filtered = classification_data[pd.notnull(classification_data['text'])]
classification_data_filtered = classification_data_filtered[pd.notnull(classification_data_filtered['category'])]

# Partición de datos en test y train.
random.seed(12345)
X_train, X_test, Y_train, Y_test = train_test_split(classification_data_filtered['text'], classification_data_filtered['category'], test_size=0.30)


cv = CountVectorizer()
cv.fit(X_train)
X_train2 = cv.transform(X_train)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train = tf_transformer.transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test = tf_transformer.transform(X_test_counts)

labels = LabelEncoder()
y_train_labels_fit = labels.fit(Y_train)
Y_train = labels.transform(Y_train)
labels = LabelEncoder()
y_test_labels_fit = labels.fit(Y_test)
clases_list = list(labels.classes_)
Y_test = labels.transform(Y_test)

# Aplicación del algoritmo.
linear_svc = LinearSVC()
clf = linear_svc.fit(X_train,Y_train)

def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()

plot_coefficients(clf, count_vect.get_feature_names())

top_features=20
classifier = clf
feature_names = count_vect.get_feature_names()

# GRÁFICOS!
# 1. Análisis  In-Sample
Y_pred = clf.predict(X_train)
fig = plt.figure()
ax = fig.add_subplot(111)
cm = confusion_matrix(Y_train, Y_pred)
cax = ax.matshow(cm)
plt.title('Matriz de Confusión del Clasificador In-Sample')
fig.colorbar(cax)
plt.xlabel('Predicción')
plt.ylabel('Real')
fig.savefig('C:/Users/Python/Desktop/confusion_matrix_in.jpeg', dpi=300, bbox_inches='tight')
plt.show()

print('Accuracy: ' + str(accuracy_score(Y_train,Y_pred)))
print('Kappa Statistic: ' + str(cohen_kappa_score(Y_train,Y_pred)))
cnf_matrix = confusion_matrix(Y_train, Y_pred)
sensitivity = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity: ', sensitivity)
specificity = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity: ', specificity)

# 2. Análisis  Out-Sample
Y_pred = clf.predict(X_test)
fig = plt.figure()
ax = fig.add_subplot(111)
cm = confusion_matrix(Y_test, Y_pred)
cax = ax.matshow(cm)
plt.title('Matriz de Confusión del Clasificador Out-Sample')
fig.colorbar(cax)
plt.xlabel('Predicción')
plt.ylabel('Real')
fig.savefig('C:/Users/Python/Desktop/confusion_matrix.jpeg', dpi=300, bbox_inches='tight')
plt.show()

print('Accuracy: ' + str(accuracy_score(Y_test,Y_pred)))
print('Kappa Statistic: ' + str(cohen_kappa_score(Y_test,Y_pred)))
cnf_matrix = confusion_matrix(Y_test, Y_pred)
sensitivity = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity: ', sensitivity)
specificity = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity: ', specificity)


# k-Fold Cross-Validation:
# Por cuestiones de tiempo no se prueban resultados pero se presenta aquì
# desarrollado desde scratch como posibilidad de mejora predictiva.
C = np.tile(np.array([0.25,0.5,1]),9)

fold = KFold(10, shuffle=False)

accuracy_accs = []
kappa_accs = []
    
for d in range(27):
    
    accuracy_acc = []
    kappa_acc = []
    
    svm = LinearSVC(C = C[d])
    
    for train_index, test_index in fold.split(X_train): 
       np.random.seed(1)
       svm.fit(X_train.iloc[train_index,:], Y_train.iloc[train_index].values.ravel())
       y_pred_undersample = svm.predict(X_train.iloc[test_index,:].values)
       accuracy_acc.append(accuracy_score(Y_train.iloc[test_index].values,y_pred_undersample))
       kappa_acc.append(cohen_kappa_score(Y_train.iloc[test_index].values,y_pred_undersample))

    accuracy_accs.append(np.mean(accuracy_acc))
    kappa_accs.append(np.mean(kappa_acc))

d_test = pd.DataFrame({'1_C': C,
         '2_Accuracy': accuracy_accs,'3_Kappa': kappa_accs})
C_optimum = C[accuracy_accs == np.max(accuracy_accs)]
print(d_test)
print("Accuracy fue usado para obtener el modelo óptimo utilizando K-Fold. El valor final usado para el modelo fue C = " + str(C_optimum[2]))



# 4. ¿Qué se puede decir de los autores? ###############################

# De los autores se puede identificar el tipo de escritura que ellos tienen.
# y qué categoría de noticias ellos escriben.

sid = SentimentIntensityAnalyzer()

total_scores_general = []
for t in range(len(general_corpus2)):
    scores = sid.polarity_scores(general_corpus2[t])
    total_scores_general.append(scores)

total_scores_general = pd.DataFrame(total_scores_general)


authors_texts = pd.concat([titularesDF['authors'].to_frame(), total_scores_general], axis=1)
authors_texts = authors_texts[pd.notnull(authors_texts['authors'])]
authors_texts = authors_texts.groupby('authors').mean()

authors_names = pd.DataFrame(authors_names.index, columns = ["authors"])
authors_texts_top = pd.merge(authors_names, authors_texts, on='authors', how='left')
authors_texts_top = authors_texts_top.drop(['authors', 'compound'], axis=1)

data_perc = authors_texts_top.divide(authors_texts_top.sum(axis=1), axis=0)
f, ax = plt.subplots(figsize=(10,6))
plt.stackplot(range(data_perc.shape[0]),  data_perc["neg"],  data_perc["neu"],  data_perc["pos"], labels=['Negative','Neutral','Positive'])
plt.legend(loc='right')
plt.margins(0,0)
plt.title('Sentimiento del texto por autor')
plt.savefig('C:/Users/Python/Desktop/sentiments.jpeg' ,dpi=300,bbox_inches="tight")
plt.show()

authors_categories = pd.crosstab(titularesDF.authors,titularesDF.category)
authors_categories["authors"] = authors_categories.index
authors_categories = pd.merge(authors_names, authors_categories, on='authors', how='left')
authors_categories = authors_categories.drop(['authors'], axis=1) # .iloc[1:]
authors_categories = authors_categories.reindex(authors_categories.sum().sort_values(ascending=False).index, axis=1)
n = len(authors_names["authors"])
c = [m+str(z) for m,z in zip(["("]*(n+1),list(range(1,(n+1))))]
positions = [m+str(z) for m,z in zip(c,[")"]*(n+1))]
authors_categories.index = [m+" "+str(n) for m,n in zip(authors_names["authors"].tolist(),positions)]

cat_1 = pd.DataFrame(authors_categories["TRAVEL"].sort_values(ascending=False).head(10).iloc[1:])
f, ax = plt.subplots(figsize=(8,10))
plt.barh(cat_1.index, cat_1["TRAVEL"], align='center', alpha=1, color= 'yellow')
plt.ylabel('Categoría')
plt.title('Autores con mayor número de artículos - Viajes')
plt.savefig('C:/Users/Python/Desktop/travel.jpeg', dpi=300, bbox_inches='tight')
plt.show()

noticias_cat

f, ax = plt.subplots(figsize=(20,6))
authors_categories2 = authors_categories
authors_categories2.index = range(len(authors_names["authors"]))
sns.heatmap(authors_categories2[:200000], vmax = 2, cmap="BuPu")
plt.title('Autores vs. Categorías')
plt.ylabel('Top de autores por artículo')
plt.xlabel('Categorías con muchos artículos                                             <---------------                                     Categorías con pocos artículos')
plt.savefig('C:/Users/Python/Desktop/author_category.jpeg' ,dpi=300,bbox_inches="tight")
plt.show()

########## WEBSCRAPPING ##########################################################
# NOTA: Se intentó hacer webscrapping a los links que aparecen en el dataset,
# pero lamentablemente el formato no está lo suficientemente estandarizado
# para hacer una extracción uniforme de todos los links. Es muy alta la probabilidad
# de obtener resultados poco útiles dado que sin un análisis profundo se extrae demasiado
# texto irrelevante.
# Debido a cuestiones de tiempo, 
# no se avanza más en ese punto  pero a continuación está el código de acceso a las páginas web:

LinkTitulares = titularesDF["link"]
content_links = []
for x in range(len(LinkTitulares)):
  url = LinkTitulares[x]
  headers = {
    'User-Agent': 'Mozilla/5.0',
    'From': 'jairo.gudino@correounivalle.edu.co'
  }
  page = requests.get(url, headers=headers)
  soup = BeautifulSoup(page.text, "html.parser")
  soup.findAll('a')
  # Eliminación de script y elementos de estilo.
  for script in soup(["script", "style"]):
    script.extract()    # rip it out
  # Extracción del texto
  text = soup.get_text()
  # Dividir en líneas y poner espacios entre ellas
  lines = (line.strip() for line in text.splitlines())
  chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
  # Eliminar espacios en blanco
  text = '\n'.join(chunk for chunk in chunks if chunk)
  # Leer entre ET y Download. Para x = 1500 esta subdivisión NO sirve.
  
  head, sep, tail = text.partition('Download')
  content_links.append(head)
  