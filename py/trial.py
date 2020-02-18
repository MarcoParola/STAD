import csv
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn import metrics

"""
stopwords = []
with open("stopwords.txt",'r') as sw:
	reader = csv.reader(sw)
	for row in reader:
		stopwords.append(row[0])
"""

dataset = pd.read_csv("fileFeatures1.csv",sep='\t',names=['tweets','target'])
#print(dataset.head(10))
print("dataset len: " + str(len(dataset)))
print("class 0 len: " + str(len(dataset[dataset.target == 0])))
print("class 1 len: " + str(len(dataset[dataset.target == 1])))
print("class 2 len: " + str(len(dataset[dataset.target == 2])) + '\n')


#splitting Training and Test set
X_train, X_test, y_train, y_test = train_test_split(dataset.tweets, dataset.target, test_size=0.4)
#X_train, X_test, y_train, y_test = dataset['tweets'][:495], dataset['tweets'][496:], dataset['target'][:495], dataset['target'][496:]

def stemming(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
    
italian_stemmer = SnowballStemmer('italian')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([italian_stemmer.stem(w) for w in analyzer(doc)])

#counting the word occurrences 
count_vect = StemmedCountVectorizer(min_df=2, analyzer="word", stop_words = set(stopwords.words('italian')))
#count_vect = CountVectorizer(stop_words=stopwords,analyzer=stemming,min_df=2) #include tokenization and stopword filtering, check parameters
X_train_counts = count_vect.fit_transform(X_train)
#extracted tokens
#print(count_vect.get_feature_names())
  
# Text rapresentation supervised stage on training set
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)# include calculation of TFs (frequencies) 
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count_vect.get_feature_names(),columns=["idf_weights"])
# sort ascending
print(df_idf.sort_values(by=['idf_weights'],ascending=True).head(30))

# TF-IDF extraction on test set
X_test_counts = count_vect.transform(X_test)#tokenization and word counting
X_test_tfidf = tfidf_transformer.transform(X_test_counts)#feature extraction

feature_names = count_vect.get_feature_names()

 
#print the scores
i = 0
while i < 1:
	print('')
	#get tfidf vector for first document
	first_document_vector=X_test_tfidf[i]
	df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
	print(df.sort_values(by=["tfidf"],ascending=False).head(10))
	i = i+1


#Training the second classifier
clf2 = svm.LinearSVC()
clf2.fit(X_train_tfidf, y_train)

#Evaluation on test set
predicted2 = clf2.predict(X_test_tfidf)#prediction
#Extracting statistics and metrics
accuracy=np.mean(predicted2 == y_test)#accuracy extreaction
print('Accuracy SVM:')
print(accuracy)

