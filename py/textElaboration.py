import nltk
import csv
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.datasets import load_files
from nltk.corpus import stopwords
#Importing the functions 
from sklearn.datasets import load_files
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer


tweets = []
classes = []

with open("fileFeatures1.csv", 'r', encoding='utf-8') as csvtwitter:
	reader = csv.reader(csvtwitter,delimiter='\t')
	for row in reader:
		tweets.append(row[0])
		classes.append(row[1])
#print(tweets)
#print(classes)


#DEFINISCI UNA PIPELINE DI FILTRI
stemmer = SnowballStemmer('italian')
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
    
text_clf3 = Pipeline([
    #('stopItalianWords', NOME_DEL_FILTRO),
    #('stopUrl', NOME_DEL_FILTRO),
    ('vect', CountVectorizer(analyzer=stemmed_words,stop_words=stopwords.words('italian'))),
    ('tfidf', TfidfTransformer(use_idf=True,max_features=30)),
    ('clf', svm.LinearSVC()),
])

#calculating accuracies in cross-valudation
scores3 = cross_val_score(text_clf3, tweets, classes, cv=5)
print("Accuracy SVM : %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))
"""
print(text_clf3['vect'].fit_transform(tweets))
print(text_clf3['vect'].get_feature_names())
print(text_clf3['tfidf'])
#print(dict(zip(text_clf3['tfidf'].get_feature_names(), idf))
"""

#Pipeline Classifier1
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords.words('italian'))),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

#calculating accuracies in cross-valudation
scores = cross_val_score(text_clf, tweets, classes, cv=5)
print("Accuracy MultinomialNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Pipeline Classifier2
text_clf2 = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords.words('italian'))),
    ('tfidf', TfidfTransformer()),
    ('clf', tree.DecisionTreeClassifier()),
])

#calculating accuracies in cross-valudation
scores2 = cross_val_score(text_clf2, tweets, classes, cv=5)
print("Accuracy Decision Tree : %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

