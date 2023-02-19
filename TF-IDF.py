#Importación de librerías a utilizar
import nltk  
import numpy as np  
import random  
import string
import pandas as pd
import heapq  

import bs4 as bs  
import urllib.request  
import re
from nltk.corpus import stopwords

#Se descarga la librería para tokenizar y eliminar stopwords.
#La ventaja es que NLKTK ya brinda esta función
nltk.download('punkt')
nltk.download('stopwords')

#Se van a anazalizar textos en español, por lo que se descarga una lista de
#stopwords en espsañol
stopwords = nltk.corpus.stopwords.words('spanish')

#Links HTML de las páginas analizadas
raw_html = urllib.request.urlopen('https://www.elfinanciero.com.mx/nacional/2023/01/11/plagio-de-ministra-yasmin-esquivel-por-que-su-titulo-no-puede-ser-invalidado-por-la-unam/')  
#raw_html = urllib.request.urlopen('https://www.washingtonpost.com/es/post-opinion/2023/01/09/ovidio-guzman-detencion-amlo-muertos-sinaloa/')  

 #Se lee la página
raw_html = raw_html.read()                       
article_html = bs.BeautifulSoup(raw_html, 'lxml')

#El texto se lee a través de las etiquetas de parrafo
article_paragraphs = article_html.find_all('p')
article_text = ''

#De acuerdo a los parrafos, se va construyendo una cadena de texto completa
for para in article_paragraphs:  
            article_text += para.text

#Con la cadena de text construida, se empieza a tokenizar mediante oraciones
corpus = nltk.sent_tokenize(article_text)

#Ya que las oraciones fueron tokenizadas se pasan a minúsculas y mediante expresiones
#regulares se eliminan aquellas que no contengan palabras y los espacios en blanco
for i in range(len(corpus )):  
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])
print(corpus)

'''
Se tokeniza por palabras, se guarda en una lista
Se vuelve a recorrer la lista para encontrar la cantidad de veces que aparece
cierta palabra y esto se guarda en un dicccionario mediante {palabra:incidencia}
'''
wordfreq = {}  
for sentence in corpus:  
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

'''
Se elige un rango de palabras más incidentes, en este caso 200 y las
200 palabras más frecuentes se guardan en una lista
Hay que tener en cuenta que en este punto todavía hay stopwords en la lista
por lo que debemos depurarlas
'''
most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

#IMPRIME LAS 200 PALABRAS MÁS FRECUENTES
print("200 PALABRAS MÁS FRECUENTES")

#most_freq = most_freq[~(most_freq["token"].isin(stopwords))]

'''
De acuerdo a la lista que brinda NLTK de stopwords, se recorre la lista de 
palabras más frecuentes y aquellas que no aparezcan en la lista, son palabras
y conectores, preposiciones, conjunciones o artículos.
'''
filtered_sentence = [w for w in most_freq if not w in stopwords] 
  
#filtered_sentence = []

for w in most_freq: 
    if w not in stopwords: 
        filtered_sentence.append(w) 

print(filtered_sentence)

#print(most_freq)

'''
IDF -> Numero total de oraciones / Numero de oraciones que contienen la palabra
---------------------------------------------------------------------------------
Nuevamente se utiliza el corpus que son las oraciones tokenizadas, para encontrar
coincidencias de las palabras tokenizadas, en caso de encontrar incidencia se
incrementa una variable que posteriormete es utilizada en la fórmula para
obtener el valor de IDF.
'''
word_idf_values = {}  
for token in filtered_sentence:  
    doc_containing_word = 0
    for document in corpus:
        if token in nltk.word_tokenize(document):
            doc_containing_word += 1
    word_idf_values[token] = np.log(len(corpus)/(1 + doc_containing_word))



#VALORES IDF [CLAVES] [VALOR]
print("----------Valores IDF----------\n")
print(word_idf_values)
print("-------------------------------\n")
#--------------------------------------------------------------------------------- 
    
'''
TF -> Frecuencia de la palabra en el documento / Total de palabras del documento
---------------------------------------------------------------------------------
Para cada palabra tokenizada, se recorre la tonización de oraciones y se
incrementa contador en caso de haber una incidencia
'''
word_tf_values = {}  
for token in filtered_sentence:  
    sent_tf_vector = []
    for document in corpus:
        doc_freq = 0
        for word in nltk.word_tokenize(document):
            if token == word:
                  doc_freq += 1
        word_tf = doc_freq/len(nltk.word_tokenize(document))
        sent_tf_vector.append(word_tf)
    word_tf_values[token] = sent_tf_vector
#VALORES TF [PALABRA] [VALOR DE CADA PALABRA POR ORACION]
print("----------Valores TF----------\n")
print(word_tf_values)   
print("------------------------------\n") 
#---------------------------------------------------------------------------------

'''
TF*IDF
De acuerdo a las listas de TF e IDF, se recorre cada una de ellas y se multiplican
los valores. Estos valores se agregan a una lista
'''
tfidf_values = []  
for token in word_tf_values.keys():  
    tfidf_sentences = []
    for tf_sentence in word_tf_values[token]:
        tf_idf_score = tf_sentence * word_idf_values[token]
        tfidf_sentences.append(tf_idf_score)
    tfidf_values.append(tfidf_sentences)   
    
'''
Para este momento se tiene una lista de listas, por lo que es necesario 
construir una matriz y transponerla para tener el resultado final
'''
tf_idf_model = np.asarray(tfidf_values)          
tf_idf_model = np.transpose(tf_idf_model)  

TFIDF= pd.DataFrame(tf_idf_model)
TFIDF.to_csv('TFIDF.csv', header=True, index=True)

#LA MATRIZ RESULTANTE CONTIENE 200 COLUMNAS CORRESPONDIENTES A LAS 200 PALABRAS
#                              LAS FILAS CORRESPONDEN

print("-----Matriz transpuesta TFxIDF-----\n")
print(tf_idf_model)
print("-----------------------------------\n")