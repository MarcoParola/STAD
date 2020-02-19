import nltk
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib
import stemming

dataset = pd.read_csv("dataset.csv",sep='\t',names=['tweets','target'])


clf = Pipeline([
    ('vect', stemming.StemmedCountVectorizer(analyzer='word',stop_words=set(stopwords.words('italian')))),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('clf', svm.LinearSVC()),
])


clf.fit(dataset.tweets, dataset.target)


data = pd.read_csv("fileFeatures1.csv",sep='\t',names=['tweets'])


cl0 = 0
cl1 = 0
cl2 = 0

predicted = clf.predict(data.tweets)
i = 0
for w in predicted:
    if w == 0:
        cl0 = cl0 +1
    if w == 1:
        cl1 = cl1 +1
        print(data.tweets[i])
    if w == 2:
        cl2 = cl2 +1
        print(data.tweets[i])
    i = i+1
        
print(cl0)
print(cl1)       
print(cl2)


joblib.dump(clf, 'STAD_model.pkl', compress=9)
