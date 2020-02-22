import nltk
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import tree
from sklearn import metrics
import stemming

dataset = pd.read_csv("dataset.csv",sep='\t',names=['tweets','target'])
folds = 10

#DEFINISCI UNA PIPELINE DI FILTRI

#Pipeline Classifier1
text_clf = Pipeline([
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', MultinomialNB()),
])

#calculating accuracies in cross-valudation
scores = cross_val_score(text_clf, dataset.tweets, dataset.target, cv=folds)
print("Accuracy MultinomialNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

#Pipeline Classifier2
text_clf2 = Pipeline([
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', tree.DecisionTreeClassifier()),
])

#calculating accuracies in cross-valudation
scores2 = cross_val_score(text_clf2, dataset.tweets, dataset.target, cv=folds)
print("Accuracy Decision Tree : %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))


#Pipeline Classifier3
text_clf3 = Pipeline([
    #('stopItalianWords', NOME_DEL_FILTRO),
    #('stopUrl', NOME_DEL_FILTRO),
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', svm.LinearSVC()),
])

#calculating accuracies in cross-valudation
scores3 = cross_val_score(text_clf3, dataset.tweets, dataset.target, cv=folds)
print("Accuracy SVM : %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))


#Pipeline Classifier4
text_clf4 = Pipeline([
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', KNeighborsClassifier(5)),
])

#calculating accuracies in cross-valudation
scores4 = cross_val_score(text_clf4, dataset.tweets, dataset.target, cv=folds)
print("Accuracy k-NN : %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std() * 2))


#Pipeline Classifier5
text_clf5 = Pipeline([
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', AdaBoostClassifier()),
])

#calculating accuracies in cross-valudation
scores5 = cross_val_score(text_clf5, dataset.tweets, dataset.target, cv=folds)
print("Accuracy Adaboost : %0.2f (+/- %0.2f)" % (scores5.mean(), scores5.std() * 2))


#Pipeline Classifier6
text_clf6 = Pipeline([
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', RandomForestClassifier()),
])

#calculating accuracies in cross-valudation
scores6 = cross_val_score(text_clf6, dataset.tweets, dataset.target, cv=folds)
print("Accuracy Random Forest : %0.2f (+/- %0.2f)" % (scores6.mean(), scores6.std() * 2))

