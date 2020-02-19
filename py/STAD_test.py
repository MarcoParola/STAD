import joblib
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
import StemmedCountVectorizer

def load_classifier():
    clf = joblib.load('STAD_model.pkl')
    return clf

dataset = pd.read_csv("fileFeatures1.csv",sep='\t',names=['tweets'])

x = load_classifier()

cl0 = 0
cl1 = 0
cl2 = 0

predicted = x.predict(dataset.tweets)
for w in predicted:
    if w == 0:
        cl0 = cl0 +1
    if w == 1:
        cl1 = cl1 +1
    if w == 2:
        cl2 = cl2 +1
        
print(cl0)
print(cl1)       
print(cl2)
