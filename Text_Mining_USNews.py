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
    text = re.sub(r'\bnt\b','',text)
    text = text.replace('thing', '')
    text = text.replace('singer', '')
    text = text.replace('every', '')
    text = text.replace('director', '')
    text = re.sub(r'\bus\b','',text)
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
    text = re.sub(r'\bman\b','',text)
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
    text = re.sub(r'\blist\b','',text)
    text = text.replace('vacation', '')
    text = text.replace('destination', '')
    text = text.replace('visit', '')
    text = text.replace('town', '')
    text = text.replace('tourist', '')
    text = text.replace('experience', '')
    text = re.sub(r'\bcountry\b','',text)
    text = text.replace('day', '')
    text = text.replace('change', '')
    text = text.replace('world', '')
    text = text.split()
    return text

def important_words_personas_famosas(words):
    text = ' '.join(string for string in words)
    text = re.sub(r'\bfriend\b','',text)
    text = re.sub(r'\bseries\b','',text)
    text = re.sub(r'\boscar\b','',text)
    text = text.replace('award', '')
    text = text.replace('work', '')
    text = text.replace('love', '')
    text = text.replace('war', '')
    text = re.sub(r'\bkid\b','',text)
    text = text.replace('role', '')
    text = text.replace('baby', '')
    text = text.replace('twitter', '')
    text = text.replace('mom', '')
    text = text.replace('comedy', '')
    text = text.replace('daughter', '')
    text = text.replace('death', '')
    text = re.sub(r'\bson\b','',text)
    text = re.sub(r'\bthr\b','',text)
    text = text.replace('release', '')
    text = text.replace('record', '')
    text = text.replace('host', '')
    text = text.replace('reveal', '')
    text = text.replace('scene', '')
    text = text.replace('debut', '')
    text = text.replace('family', '')
    text = re.sub(r'\balt\b','',text)
    text = re.sub(r'\bdond\b','',text)
    text = text.replace('feel', '')
    text = text.replace('men', '')
    text = re.sub(r'\brock\b','',text)
    text = text.replace('book', '')
    text = text.replace('tweet', '')
    text = text.replace('lot', '')
    text = text.replace('heart', '')
    text = text.replace('boy', '')
    text = text.replace('part', '')
    text = text.replace('return', '')
    text = text.replace('review', '')
    text = text.replace('birth', '')
    text = text.replace('word', '')
    text = text.replace('writer', '')
    text = text.replace('clip', '')
    text = text.replace('celebrities', '')
    text = text.replace('episode', '')
    text = text.replace('break', '')
    text = re.sub(r'\bed\b','',text)
    text = text.replace('disney', '')
    text = text.replace('age', '')
    text = text.replace('guy', '')
    text = re.sub(r'\bbill\b','',text)
    text = text.replace('netflix', '')
    text = text.replace('report', '')
    text = text.replace('perform', '')
    text = text.replace('horror', '')
    text = text.replace('cover', '')
    text = text.replace('post', '')
    text = text.replace('globe', '')
    text = text.replace('name', '')
    text = text.replace('band', '')
    text = text.replace('winner', '')
    text = re.sub(r'\bend\b','',text)
    text = text.replace('artist', '')
    text = text.replace('theater', '')
    text = text.replace('summer', '')
    text = text.replace('meet', '')
    text = re.sub(r'\bst\b','',text)
    text = text.replace('producer', '')
    text = re.sub(r'\bwa\b','',text)
    text = re.sub(r'\blet\b','',text)
    text = re.sub(r'\breity\b','',text)
    text = re.sub(r'\bqueion\b','',text)
    text = re.sub(r'\bill\b','',text)
    text = re.sub(r'\blet\b','',text)
    text = re.sub(r'\bfeiv\b','',text)
    text = re.sub(r'\bsnl\b','',text)
    text = re.sub(r'\bre\b','',text)
    text = re.sub(r'\bly\b','',text)
    text = re.sub(r'\bla\b','',text)
    text = re.sub(r'\bcelebrity\b','',text)
    text = re.sub(r'\bgrammy\b','',text)
    text = re.sub(r'\bmion\b','',text)
    text = re.sub(r'\brumor\b','',text)
    text = re.sub(r'\bface\b','',text)
    text = re.sub(r'\bleg\b','',text)
    text = re.sub(r'\byork\b','',text)
    text = re.sub(r'\bmember\b','',text)
    text = re.sub(r'\bvoice\b','',text)
    text = re.sub(r'\bchild\b','',text)
    text = re.sub(r'\bkind\b','',text)
    text = re.sub(r'\bhead\b','',text)
    text = re.sub(r'\btell\b','',text)
    text = re.sub(r'\bfin\b','',text)
    text = re.sub(r'\bpmie\b','',text)
    text = re.sub(r'\bsurprise\b','',text)
    text = re.sub(r'\binagram\b','',text)
    text = re.sub(r'\bact\b','',text)
    text = re.sub(r'\bhelp\b','',text)
    text = re.sub(r'\bmess\b','',text)
    text = re.sub(r'\baudience\b','',text)
    text = re.sub(r'\bjoin\b','',text)
    text = re.sub(r'\bad\b','',text)
    text = re.sub(r'\bline\b','',text)
    text = re.sub(r'\bgett\b','',text)
    text = re.sub(r'\bsupport\b','',text)
    text = re.sub(r'\bfri\b','',text)
    text = re.sub(r'\bip\b','',text)
    text = re.sub(r'\bwife\b','',text)
    text = re.sub(r'\bdict\b','',text)
    text = re.sub(r'\bcouple\b','',text)
    text = re.sub(r'\bmark\b','',text)
    text = re.sub(r'\bnumber\b','',text)
    text = re.sub(r'\bthank\b','',text)
    text = re.sub(r'\bconcert\b','',text)
    text = re.sub(r'\bal\b','',text)
    text = re.sub(r'\bent\b','',text)
    text = re.sub(r'\bdonald\b','',text)
    text = re.sub(r'\bpresident\b','',text)
    text = re.sub(r'\bfestival\b','',text)
    text = re.sub(r'\bfriend\b','',text)
    text = re.sub(r'\bthrs\b','',text)
    text = re.sub(r'\binstagram\b','',text)
    text = re.sub(r'\bmedia\b','',text)
    text = re.sub(r'\bsle\b','',text)
    text = re.sub(r'\bclaim\b','',text)
    text = re.sub(r'\bhouse\b','',text)
    text = re.sub(r'\bperson\b','',text)
    text = re.sub(r'\bmonth\b','',text)
    text = re.sub(r'\bcast\b','',text)
    text = re.sub(r'\breality\b','',text)
    text = re.sub(r'\bnomination\b','',text)
    text = re.sub(r'\binterview\b','',text)
    text = re.sub(r'\bquestion\b','',text)
    text = re.sub(r'\ber\b','',text)
    text = re.sub(r'\boscar\b','',text)
    text = re.sub(r'\bdate\b','',text)
    text = re.sub(r'\bthink\b','',text)
    text = re.sub(r'\bbody\b','',text)
    text = re.sub(r'\bmother\b','',text)
    text = re.sub(r'\bcall\b','',text)
    text = re.sub(r'\bfriends\b','',text)
    text = re.sub(r'\bsee\b','',text)
    text = re.sub(r'\bance\b','',text)
    text = re.sub(r'\bdrama\b','',text)
    text = re.sub(r'\bsex\b','',text)
    text = re.sub(r'\bwalk\b','',text)
    text = re.sub(r'\bschool\b','',text)
    text = re.sub(r'\bdance\b','',text)
    text = re.sub(r'\bjoke\b','',text)
    text = re.sub(r'\boscars\b','',text)
    text = re.sub(r'\bcareer\b','',text)
    text = re.sub(r'\bevent\b','',text)
    text = re.sub(r'\bscreen\b','',text)
    text = re.sub(r'\bhit\b','',text)
    text = re.sub(r'\bals\b','',text)
    text = re.sub(r'\bneed\b','',text)
    text = re.sub(r'\bgroup\b','',text)
    text = re.sub(r'\bparent\b','',text)
    text = re.sub(r'\bdream\b','',text)
    text = re.sub(r'\bparent\b','',text)
    text = re.sub(r'\bchristmas\b','',text)
    text = re.sub(r'\bcase\b','',text)
    text = re.sub(r'\bwoman\b','',text)
    text = re.sub(r'\bbr\b','',text)
    text = re.sub(r'\bkids\b','',text)
    text = re.sub(r'\bchris\b','',text)
    text = re.sub(r'\bteaser\b','',text)
    text = re.sub(r'\bpower\b','',text)
    text = re.sub(r'\bconversation\b','',text)
    text = re.sub(r'\bbattle\b','',text)
    text = re.sub(r'\bhi\b','',text)
    text = re.sub(r'\braper\b','',text)
    text = re.sub(r'\bpicture\b','',text)
    text = re.sub(r'\bfire\b','',text)
    text = re.sub(r'\bmonths\b','',text)
    text = re.sub(r'\bcarpet\b','',text)
    text = re.sub(r'\bfun\b','',text)
    text = re.sub(r'\bmak\b','',text)
    text = re.sub(r'\brumors\b','',text)
    text = re.sub(r'\beye\b','',text)
    text = re.sub(r'\bchoice\b','',text)
    text = re.sub(r'\bidea\b','',text)
    text = re.sub(r'\bfall\b','',text)
    text = re.sub(r'\btrack\b','',text)
    text = re.sub(r'\brapper\b','',text)
    text = re.sub(r'\bmurder\b','',text)
    text = re.sub(r'\bamerica\b','',text)
    text = re.sub(r'\bcritic\b','',text)
    text = re.sub(r'\breunion\b','',text)
    text = re.sub(r'\bproject\b','',text)
    text = re.sub(r'\bamerica\b','',text)
    text = re.sub(r'\bhand\b','',text)
    text = re.sub(r'\bmaker\b','',text)
    text = re.sub(r'\bstudio\b','',text)
    text = re.sub(r'\bcritics\b','',text)
    text = re.sub(r'\brelationship\b','',text)
    text = re.sub(r'\btelevision\b','',text)
    text = re.sub(r'\bassault\b','',text)
    
    # text = text.replace('james', '')
    text = re.sub(r'\bjames\b','',text)
    text = re.sub(r'\bjohn\b','',text)
    
    # Se eliminan los ultra-stars para obtener nueva información #
    text = re.sub(r'\btrump\b','',text)
    text = re.sub(r'\bbeyonce\b','',text)
    text = re.sub(r'\btaylor\b','',text)
    text = re.sub(r'\bswift\b','',text)
    text = text.split()
    return text

def important_words_ciudades(words):
    text = ' '.join(string for string in words)
    text = text.replace('park', '')
    text = text.replace('cities', '')
    text = text.replace('road', '')
    text = text.replace('airpline', '')
    text = text.replace('flight', '')
    text = text.replace('summer', '')
    text = text.replace('family', '')
    text = text.replace('island', '')
    text = text.replace('beach', '')
    text = text.replace('airport', '')
    text = text.replace('food', '')
    text = text.replace('resort', '')
    text = text.replace('plane', '')
    text = text.replace('air', '')
    text = text.replace('view', '')
    text = text.replace('water', '')
    text = text.replace('passenger', '')
    text = text.replace('line', '')
    text = text.replace('book', '')
    text = text.replace('spot', '')
    text = text.replace('state', '')
    text = text.replace('work', '')
    text = text.replace('book', '')
    text = text.replace('bar', '')
    text = text.replace('kid', '')
    text = text.replace('culture', '')
    text = text.replace('mile', '')
    text = text.replace('hour', '')
    text = text.replace('room', '')
    text = text.replace('offer', '')
    text = text.replace('restaura', '')
    text = text.replace('nation', '')
    text = re.sub(r'\bgeta\b','',text)
    text = re.sub(r'\bpart\b','',text)
    text = re.sub(r'\bstreet\b','',text)
    text = text.replace('cruise', '')
    text = text.replace('coast', '')
    text = text.replace('attraction', '')
    text = text.replace('guide', '')
    text = re.sub(r'\bloc\b','',text)
    text = text.replace('head', '')
    text = text.replace('luxury', '')
    text = text.replace('plan', '')
    text = text.replace('tourism', '')
    text = text.replace('fun', '')
    text = text.replace('dream', '')
    text = text.replace('sea', '')
    text = text.replace('tour', '')
    text = text.replace('stay', '')
    text = text.replace('love', '')
    text = re.sub(r'\bre\b','',text)
    text = text.replace('lot', '')
    text = text.replace('feel', '')
    text = re.sub(r'\bwa\b','',text)
    text = text.replace('eat', '')
    text = re.sub(r'\bmoh\b','',text)
    text = text.replace('person', '')
    text = re.sub(r'\bes\b','',text)
    text = re.sub(r'\bfestival\b','',text)
    text = re.sub(r'\bgat\b','',text)
    text = text.replace('art', '')
    text = re.sub(r'\bmak\b','',text)
    text = re.sub(r'\bity\b','',text)
    text = re.sub(r'\btak\b','',text)
    text = re.sub(r'\bfriend\b','',text)
    text = text.replace('check', '')
    text = text.replace('member', '')
    text = text.replace('sit', '')
    text = text.replace('friend', '')
    text = text.replace('port', '')
    text = text.replace('ill', '')
    text = text.replace('unit', '')
    text = text.replace('end', '')
    text = text.replace('car', '')
    text = text.replace('age', '')
    text = text.replace('bath', '')
    text = text.replace('fly', '')
    text = text.replace('budget', '')
    text = text.replace('pack', '')
    text = text.replace('idea', '')
    text = text.replace('help', '')
    text = text.replace('mind', '')
    text = text.replace('mark', '')
    text = text.replace('journey', '')
    text = text.replace('beauty', '')
    text = text.replace('kind', '')
    text = text.replace('accord', '')
    text = text.replace('bit', '')
    text = text.replace('turn', '')
    text = text.replace('fact', '')
    text = re.sub(r'\bal\b','',text)
    text = re.sub(r'\bst\b','',text)
    text = text.replace('america', '')
    text = re.sub(r'\bation\b','',text)
    text = re.sub(r'\bholi\b','',text)
    text = re.sub(r'\bnt\b','',text)
    text = re.sub(r'\bfall\b','',text)
    text = re.sub(r'\badventure\b','',text)
    text = re.sub(r'\byork\b','',text)
    text = re.sub(r'\bmountain\b','',text)
    text = re.sub(r'\bdeal\b','',text)
    text = re.sub(r'\bals\b','',text)
    text = re.sub(r'\bgr\b','',text)
    text = re.sub(r'\bgett\b','',text)
    text = re.sub(r'\bre\b','',text)
    text = re.sub(r'\bmuseum\b','',text)
    text = re.sub(r'\bwinter\b','',text)
    text = re.sub(r'\bside\b','',text)
    text = re.sub(r'\bdeal\b','',text)
    text = re.sub(r'\blocals\b','',text)
    text = re.sub(r'\briver\b','',text)
    text = re.sub(r'\bors\b','',text)
    text = re.sub(r'\bship\b','',text)
    text = re.sub(r'\bed\b','',text)
    text = re.sub(r'\bmion\b','',text)
    text = re.sub(r'\barea\b','',text)
    text = re.sub(r'\bpoint\b','',text)
    text = re.sub(r'\bcapital\b','',text)
    text = re.sub(r'\bstyle\b','',text)
    text = re.sub(r'\ber\b','',text)
    text = re.sub(r'\bpoint\b','',text)
    text = re.sub(r'\bstyle\b','',text)
    text = re.sub(r'\btsa\b','',text)
    text = re.sub(r'\bspr\b','',text)
    text = re.sub(r'\bservice\b','',text)
    text = re.sub(r'\bhouse\b','',text)
    text = re.sub(r'\bspace\b','',text)
    text = re.sub(r'\bsun\b','',text)
    text = re.sub(r'\bwonder\b','',text)
    text = re.sub(r'\bcountries\b','',text)
    text = re.sub(r'\bnts\b','',text)
    text = re.sub(r'\bcourse\b','',text)
    text = re.sub(r'\bsee\b','',text)
    text = re.sub(r'\bquestion\b','',text)
    text = re.sub(r'\bescape\b','',text)
    text = re.sub(r'\bhi\b','',text)
    text = re.sub(r'\bmoment\b','',text)
    text = re.sub(r'\bbusiness\b','',text)
    text = re.sub(r'\bride\b','',text)
    text = re.sub(r'\bwalk\b','',text)
    text = re.sub(r'\bticket\b','',text)
    text = re.sub(r'\bbreak\b','',text)
    text = re.sub(r'\beye\b','',text)
    text = re.sub(r'\bclass\b','',text)
    text = re.sub(r'\bprice\b','',text)
    text = re.sub(r'\bbeer\b','',text)
    text = re.sub(r'\bneed\b','',text)
    text = re.sub(r'\bclass\b','',text)
    text = re.sub(r'\bname\b','',text)
    text = re.sub(r'\bwine\b','',text)
    text = re.sub(r'\bgetas\b','',text)
    text = re.sub(r'\bletter\b','',text)
    text = re.sub(r'\bfrance\b','',text)
    text = re.sub(r'\bindustry\b','',text)
    text = re.sub(r'\blocation\b','',text)
    text = re.sub(r'\bground\b','',text)
    text = re.sub(r'\bfee\b','',text)
    text = re.sub(r'\bregion\b','',text)
    text = re.sub(r'\bevent\b','',text)
    text = re.sub(r'\bmonth\b','',text)
    text = re.sub(r'\bmeet\b','',text)
    text = re.sub(r'\bfare\b','',text)
    text = re.sub(r'\bthank\b','',text)
    text = re.sub(r'\bly\b','',text)
    text = re.sub(r'\bcenter\b','',text)
    text = re.sub(r'\btree\b','',text)
    text = re.sub(r'\bice\b','',text)
    text = re.sub(r'\bbag\b','',text)
    text = re.sub(r'\blight\b','',text)
    text = re.sub(r'\bmonths\b','',text)
    text = re.sub(r'\bfeet\b','',text)
    text = re.sub(r'\bscene\b','',text)
    text = re.sub(r'\bhand\b','',text)
    text = re.sub(r'\bscape\b','',text)
    text = re.sub(r'\bscape\b','',text)
    text = re.sub(r'\bguest\b','',text)
    text = re.sub(r'\bfeet\b','',text)
    text = re.sub(r'\btrain\b','',text)
    text = re.sub(r'\bsp\b','',text)
    text = re.sub(r'\bcall\b','',text)
    text = re.sub(r'\beurope\b','',text)
    text = re.sub(r'\bcase\b','',text)
    text = re.sub(r'\bsomes\b','',text)
    text = re.sub(r'\bsight\b','',text)
    text = re.sub(r'\bcompany\b','',text)
    text = re.sub(r'\bothers\b','',text)
    text = re.sub(r'\bactivities\b','',text)
    text = re.sub(r'\brule\b','',text)
    text = re.sub(r'\btry\b','',text)
    text = re.sub(r'\bland\b','',text)
    text = re.sub(r'\blandscape\b','',text)
    text = re.sub(r'\bminute\b','',text)
    text = re.sub(r'\bdrink\b','',text)
    text = re.sub(r'\bdin\b','',text)
    text = re.sub(r'\bgroup\b','',text)
    text = re.sub(r'\bcouple\b','',text)
    text = re.sub(r'\bmountains\b','',text)
    text = re.sub(r'\bdeals\b','',text)
    text = re.sub(r'\bpresident\b','',text)
    text = re.sub(r'\bthanks\b','',text)
    text = re.sub(r'\banimal\b','',text)
    text = re.sub(r'\bpresident\b','',text)
    text = re.sub(r'\bcost\b','',text)
    text = re.sub(r'\bspirit\b','',text)
    text = re.sub(r'\bshop\b','',text)
    text = re.sub(r'\bfind\b','',text)
    text = re.sub(r'\bword\b','',text)
    text = re.sub(r'\bchildren\b','',text)
    text = re.sub(r'\beh\b','',text)
    text = re.sub(r'\bsecurity\b','',text)
    text = re.sub(r'\bpass\b','',text)
    text = re.sub(r'\blake\b','',text)
    text = re.sub(r'\bbuild\b','',text)
    text = re.sub(r'\btaste\b','',text)
    text = text.split()
    
    return text

def important_words_politic_issues(words):
    text = ' '.join(string for string in words)
    text = re.sub(r'\btrump\b','',text)
    text = re.sub(r'\bpresident\b','',text)
    text = re.sub(r'\bdonald\b','',text)
    text = re.sub(r'\bdemocrat\b','',text)
    text = re.sub(r'\bstate\b','',text)
    text = re.sub(r'\bhouse\b','',text)
    text = re.sub(r'\brepublican\b','',text)
    text = re.sub(r'\bbill\b','',text)
    text = re.sub(r'\bstate\b','',text)
    text = re.sub(r'\bcandidate\b','',text)
    text = re.sub(r'\bsenator\b','',text)
    text = re.sub(r'\bdemocrat\b','',text)
    text = re.sub(r'\bhouse\b','',text)
    text = re.sub(r'\bbill\b','',text)
    text = re.sub(r'\bcampaign\b','',text)
    text = re.sub(r'\belection\b','',text)
    text = re.sub(r'\bamericans\b','',text)
    text = re.sub(r'\bhouse\b','',text)
    text = re.sub(r'\bsupport\b','',text)
    text = re.sub(r'\bvoter\b','',text)
    text = re.sub(r'\bvote\b','',text)
    text = re.sub(r'\bgop\b','',text)
    text = re.sub(r'\blaw\b','',text)
    text = re.sub(r'\bpoll\b','',text)
    text = re.sub(r'\bparty\b','',text)
    text = re.sub(r'\bgroup\b','',text)
    text = re.sub(r'\bwork\b','',text)
    text = re.sub(r'\bobama\b','',text)
    text = re.sub(r'\bclinton\b','',text)
    text = re.sub(r'\bplan\b','',text)
    text = re.sub(r'\bbernie\b','',text)
    text = re.sub(r'\bsander\b','',text)
    text = re.sub(r'\bsanders\b','',text)
    text = re.sub(r'\bstates\b','',text)
    text = re.sub(r'\brepublicans\b','',text)
    text = re.sub(r'\bdemocrats\b','',text)
    text = re.sub(r'\bgovernment\b','',text)
    text = re.sub(r'\bleader\b','',text)
    text = re.sub(r'\bvoters\b','',text)
    text = re.sub(r'\badministration\b','',text)
    text = re.sub(r'\bgovernment\b','',text)
    text = re.sub(r'\bdebate\b','',text)
    text = re.sub(r'\bcongress\b','',text)
    text = re.sub(r'\breport\b','',text)
    text = re.sub(r'\bpolicy\b','',text)
    text = re.sub(r'\battack\b','',text)
    text = re.sub(r'\bissue\b','',text)
    text = re.sub(r'\breport\b','',text)
    text = re.sub(r'\blawmaker\b','',text)
    text = re.sub(r'\bmonth\b','',text)
    text = re.sub(r'\brace\b','',text)
    text = re.sub(r'\bclaim\b','',text)
    text = re.sub(r'\bcase\b','',text)
    text = re.sub(r'\bnation\b','',text)
    text = re.sub(r'\bcall\b','',text)
    text = re.sub(r'\bofficial\b','',text)
    text = re.sub(r'\bcandidates\b','',text)
    text = re.sub(r'\bproblem\b','',text)
    text = re.sub(r'\bright\b','',text)
    text = re.sub(r'\bquestion\b','',text)
    text = re.sub(r'\baction\b','',text)
    text = re.sub(r'\bjudge\b','',text)
    text = re.sub(r'\bcourt\b','',text)
    text = re.sub(r'\bpolice\b','',text)
    text = re.sub(r'\bpolitics\b','',text)
    text = re.sub(r'\bsenate\b','',text)
    text = re.sub(r'\blawmakers\b','',text)
    text = re.sub(r'\bhelp\b','',text)
    text = re.sub(r'\bamerica\b','',text)
    text = re.sub(r'\brule\b','',text)
    text = re.sub(r'\border\b','',text)
    text = re.sub(r'\bdeal\b','',text)
    text = re.sub(r'\bclimate\b','',text)
    text = re.sub(r'\bofficials\b','',text)
    text = re.sub(r'\bsupporter\b','',text)
    text = re.sub(r'\bleaders\b','',text)
    text = re.sub(r'\bfight\b','',text)
    text = re.sub(r'\bgovernor\b','',text)
    text = re.sub(r'\bmedia\b','',text)
    text = re.sub(r'\bpart\b','',text)
    text = re.sub(r'\byork\b','',text)
    text = re.sub(r'\brights\b','',text)
    text = re.sub(r'\bdecision\b','',text)
    text = re.sub(r'\bpower\b','',text)
    text = re.sub(r'\bsupporters\b','',text)
    text = re.sub(r'\bfund\b','',text)
    text = re.sub(r'\bnominee\b','',text)
    text = re.sub(r'\bsupporters\b','',text)
    text = re.sub(r'\bnominee\b','',text)
    text = re.sub(r'\bhuffpost\b','',text)
    text = re.sub(r'\brise\b','',text)
    text = re.sub(r'\bdeath\b','',text)
    text = re.sub(r'\bsupreme\b','',text)
    text = re.sub(r'\btax\b','',text)
    text = re.sub(r'\bjustice\b','',text)
    text = re.sub(r'\bemail\b','',text)
    text = re.sub(r'\blawyer\b','',text)
    text = re.sub(r'\bsystem\b','',text)
    text = re.sub(r'\bdepartment\b','',text)
    text = re.sub(r'\beffort\b','',text)
    text = re.sub(r'\bviolence\b','',text)
    text = re.sub(r'\bcruz\b','',text)
    text = re.sub(r'\btweet\b','',text)
    text = re.sub(r'\bmeet\b','',text)
    text = re.sub(r'\bpoint\b','',text)
    text = re.sub(r'\bnumber\b','',text)
    text = re.sub(r'\bend\b','',text)
    text = re.sub(r'\bmember\b','',text)
    text = re.sub(r'\bspeech\b','',text)
    text = re.sub(r'\binvestigation\b','',text)
    text = re.sub(r'\bcomment\b','',text)
    text = re.sub(r'\bclaims\b','',text)
    text = re.sub(r'\bidea\b','',text)
    text = re.sub(r'\bhi\b','',text)
    text = re.sub(r'\bprogram\b','',text)
    text = re.sub(r'\brecord\b','',text)
    text = re.sub(r'\bsenators\b','',text)
    text = re.sub(r'\bwashton\b','',text)
    text = re.sub(r'\btues\b','',text)
    text = re.sub(r'\blot\b','',text)
    text = re.sub(r'\bneed\b','',text)
    text = re.sub(r'\bmoney\b','',text)
    text = re.sub(r'\battorney\b','',text)
    text = re.sub(r'\bcrisis\b','',text)
    text = re.sub(r'\bthreat\b','',text)
    text = re.sub(r'\bwoman\b','',text)
    text = re.sub(r'\bconvention\b','',text)
    text = re.sub(r'\bworker\b','',text)
    text = re.sub(r'\bword\b','',text)
    text = text.split()
    return text

def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center', color = 'black')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()
    
    
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
titularesDF['category'].value_counts().head(10).sort_values(ascending=False).plot.bar(color='black')
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

politics_text2 = important_words(words)

# Temas de política más importantes
politics_text = important_words_politic_issues(politics_text2)
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

politics_count = Counter(words_politics.split()).most_common()

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


# Personas famosas aparte de Trump, Beyonce & Taylor Swift.
entertainment_text = important_words(words)
entertainment_text = important_words_personas_famosas(entertainment_text)
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

entertainment_count = Counter(words_entertainment.split()).most_common()

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

# Ciudades importantes
travel_text = important_words(words)
travel_text = important_words_ciudades(travel_text)
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

travel_count = Counter(words_travel.split()).most_common()

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
authors_names_top["authors"].value_counts().sort_values(ascending=False).plot.bar(color='black')
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
plt.barh(category_scores.index, category_scores["compound"], align='center', alpha=1, color= 'black')
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

# Variables más importantes.
f, ax = plt.subplots(figsize=(8,10))
f_importances(abs(clf.coef_[0]), count_vect.get_feature_names(), top=20)
f.savefig('C:/Users/Python/Desktop/features.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# GRÁFICOS!
# 1. Análisis  In-Sample
Y_pred = clf.predict(X_train)
fig = plt.subplots(figsize=(10,10))
cm = pd.DataFrame(confusion_matrix(Y_train, Y_pred))
for i in range(len(cm.index)):
    for j in range(len(cm.columns)):
        if i==j:
            cm.loc[i,j] = 0
cm.columns = clases_list
cm.index = clases_list
sns.heatmap(cm, cmap='gist_gray_r')
# cax = ax.matshow(cm)
plt.title('Matriz de Confusión del Clasificador In-Sample')
# fig.colorbar(cax)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig('C:/Users/Python/Desktop/confusion_matrix_in.jpeg', dpi=300, bbox_inches='tight')
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
fig = plt.subplots(figsize=(12,12))
cm = pd.DataFrame(confusion_matrix(Y_test, Y_pred))
for i in range(len(cm.index)):
    for j in range(len(cm.columns)):
        if i==j:
            cm.loc[i,j] = 0
cm.columns = clases_list
cm.index = clases_list
sns.heatmap(cm, cmap='gist_gray_r', xticklabels=False, yticklabels=False)
plt.title('Matriz de Confusión del Clasificador Out-Sample')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig('C:/Users/Python/Desktop/confusion_matrix_out.jpeg', dpi=300, bbox_inches='tight')
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

## CLASIFICACIÓN CON WEBSCRAPPING ###
# Se leen los links de las noticias para probar mejoras del modelo incluyendo más
# palabras para cada texto.
# Debido a cuestiones de problemas de acceso a internet no se avanza más en este punto,
# pero el siguiente código se puede correr para testear poder predictivo del modelo,
# en un sitio donde la señal de internet sí sea óptima.

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
  
  # Leer entre ET y Download. 
  head, sep, tail = text.partition('Join HuffPost Plus')
  head, sep, tail = tail.partition('ET')
  head, sep, tail = tail.partition('Download')
  content_links.append(head)

lemmatizer = WordNetLemmatizer()
general_corpus = pd.DataFrame(content_links)
general_corpus = general_corpus.apply(word_tokenize)
general_corpus = general_corpus.apply(normalize)
general_corpus2 = general_corpus.tolist()
general_corpus2 = [' '.join(x) for x in general_corpus2]
general_corpus2 = [lemmatizer.lemmatize(w) for w in general_corpus2]
classification_data = pd.concat([titularesDF['category'].to_frame(), pd.DataFrame(general_corpus2)], axis=1)
classification_data.columns = ["category","text"]
classification_data_filtered = classification_data[pd.notnull(classification_data['text'])]
classification_data_filtered = classification_data_filtered[pd.notnull(classification_data_filtered['category'])]
random.seed(12345)
X_train_webscrapping, X_test_webscrapping, Y_train_webscrapping, Y_test_webscrapping = train_test_split(classification_data_filtered['text'], classification_data_filtered['category'], test_size=0.30)

cv = CountVectorizer()
cv.fit(X_train_webscrapping)
X_train2_webscrapping = cv.transform(X_train_webscrapping)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train_webscrapping)
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_webscrapping= tf_transformer.transform(X_train_counts)
X_test_counts = count_vect.transform(X_test_webscrapping)
X_test_webscrapping = tf_transformer.transform(X_test_counts)
labels = LabelEncoder()
y_train_labels_fit = labels.fit(Y_train_webscrapping)
Y_train_webscrapping = labels.transform(Y_train_webscrapping)
labels = LabelEncoder()
y_test_labels_fit = labels.fit(Y_test_webscrapping)
clases_list = list(labels.classes_)
Y_test_webscrapping = labels.transform(Y_test_webscrapping)
linear_svc = LinearSVC()
clf = linear_svc.fit(X_train_webscrapping,Y_train_webscrapping)

Y_pred_webscrapping = clf.predict(X_train_webscrapping)
print('Accuracy: ' + str(accuracy_score(Y_test_webscrapping,Y_pred_webscrapping)))
print('Kappa Statistic: ' + str(cohen_kappa_score(Y_test_webscrapping,Y_pred_webscrapping)))
cnf_matrix = confusion_matrix(Y_test_webscrapping, Y_pred_webscrapping)
sensitivity = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity: ', sensitivity)
specificity = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity: ', specificity)

# 4. ¿Qué se puede decir de los autores? ###############################

# De los autores se puede identificar el tipo de escritura que ellos tienen.
# y qué categoría de noticias ellos escriben.

sid = SentimentIntensityAnalyzer()

total_scores_general = []
for t in range(len(general_corpus2)):
    scores = sid.polarity_scores(general_corpus2[t])
    total_scores_general.append(scores)

total_scores_general = pd.DataFrame(total_scores_general)


# Análisis de sentimiento por autor #

authors_texts = pd.concat([titularesDF['authors'].to_frame(), total_scores_general], axis=1)
authors_texts = authors_texts[pd.notnull(authors_texts['authors'])]
authors_texts = authors_texts.groupby('authors').mean()

authors_names = pd.DataFrame(authors_names.index, columns = ["authors"])
authors_texts_top = pd.merge(authors_names, authors_texts, on='authors', how='left')
authors_texts_top = authors_texts_top.drop(['authors', 'compound'], axis=1)

data_perc = authors_texts_top.divide(authors_texts_top.sum(axis=1), axis=0)
f, ax = plt.subplots(figsize=(10,6))
plt.stackplot(range(data_perc.shape[0]),  data_perc["neg"],  data_perc["neu"],  data_perc["pos"], labels=['Negative','Neutral','Positive'],
              colors = sns.color_palette("Set2"))
plt.legend(loc='right')
plt.margins(0,0)
plt.title('Sentimiento del texto por autor')
plt.savefig('C:/Users/Python/Desktop/sentiments.jpeg' ,dpi=300,bbox_inches="tight")
plt.show()


# Autores con mayor número de artículos por categoría #

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

# Categorías de artículos vs. Top de Autores ##

f, ax = plt.subplots(figsize=(20,6))
authors_categories2 = authors_categories
authors_categories2.index = range(len(authors_names["authors"]))
sns.heatmap(authors_categories2[:200000], vmax = 2, cmap="gist_gray_r")
plt.title('Autores vs. Categorías')
plt.ylabel('Top de autores por artículo')
plt.xlabel('Categorías con muchos artículos                                             <---------------                                     Categorías con pocos artículos')
plt.savefig('C:/Users/Python/Desktop/author_category.jpeg' ,dpi=300,bbox_inches="tight")
plt.show()


#### Matriz de similaridad de textos de autores ####

from sklearn.feature_extraction.text import TfidfVectorizer

titularesDF["TransformedText"] = pd.DataFrame(general_corpus2, columns = ["Text"])

authors_names = titularesDF.groupby('authors').size().sort_values(ascending=False)
authors_names.head(20)
authors_names = authors_names.head(351).iloc[1:]
authors_names = pd.DataFrame(authors_names.index, columns = ["authors"])
Categories = authors_names["authors"]
words_author = []
for t in range(len(Categories)):
    Word = titularesDF[titularesDF["authors"]==Categories[t]]["TransformedText"]
    words_author.append(' '.join(string for string in Word))
    
Texts = pd.DataFrame(words_author, columns = ["Text"])

tfidf = TfidfVectorizer().fit_transform(words_author)
# no need to normalize, since Vectorizer will return normalized tf-idf
pairwise_similarity = (tfidf * tfidf.T).toarray()

f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(pairwise_similarity, vmax = 1, cmap="gist_gray_r")
plt.title('Similaridad entre autores')
plt.ylabel('Top de autores por artículo')
plt.savefig('C:/Users/Python/Desktop/author_similarity.jpeg' ,dpi=300,bbox_inches="tight")
plt.show()
