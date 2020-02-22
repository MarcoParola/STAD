import csv
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import stemming
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn import metrics

dataset = pd.read_csv("dataset.csv",sep='\t',names=['tweets','target'])
#print(dataset.head(10))
print("dataset len: " + str(len(dataset)))
print("class 0 len: " + str(len(dataset[dataset.target == 0])))
print("class 1 len: " + str(len(dataset[dataset.target == 1])))
print("class 2 len: " + str(len(dataset[dataset.target == 2])) + '\n')


#splitting Training and Test set
X_train, X_test, y_train, y_test = train_test_split(dataset.tweets, dataset.target, test_size=0.3)


#counting the word occurrences 
count_vect = stemming.StemmedCountVectorizer(min_df=2, analyzer="word", stop_words = set(stopwords.words('italian')))
#count_vect = CountVectorizer(stop_words=stopwords,analyzer=stemming,min_df=2)
X_train_counts = count_vect.fit_transform(X_train)
#extracted tokens
#print(count_vect.get_feature_names())
  
# Text rapresentation supervised stage on training set
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)# include calculation of TFs (frequencies) 
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

"""
# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count_vect.get_feature_names(),columns=["idf_weights"])
# sort ascending
print(df_idf.sort_values(by=['idf_weights'],ascending=True).head(30))
"""

# TF-IDF extraction on test set
X_test_counts = count_vect.transform(X_test)#tokenization and word counting
X_test_tfidf = tfidf_transformer.transform(X_test_counts)#feature extraction

"""
# print info_gain of first N documents' features

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
print('')
"""

def evaluate_classifier(clf):
	clf.fit(X_train_tfidf, y_train)
	#Evaluation on test set
	predicted = clf.predict(X_test_tfidf)#prediction
	#Extracting statistics and metrics
	accuracy=np.mean(predicted == y_test)#accuracy extreaction
	print('accuracy : ' + str(accuracy))

	f_score = f1_score(y_test, predicted, average='macro')
	print('f_score : ' + str(f_score) + '\n')

	disp = plot_confusion_matrix(clf, X_test_tfidf, y_test, display_labels=[0,1,2], cmap=plt.cm.Blues, normalize='true')
	disp.ax_.set_title('Confusion Matrix')
	plt.show()


# --------------- RANDOM FOREST ---------------
#Training the sixth classifier
clf6 = RandomForestClassifier()
print('Random Forest:')
evaluate_classifier(clf6)

# --------------- BAYESS ---------------
#Training the first classifier
clf = MultinomialNB()
print('Multinomial NB:')
evaluate_classifier(clf)

# --------------- SVC ---------------
#Training the second classifier
clf2 = svm.LinearSVC()
print('SVM:')
evaluate_classifier(clf2)

# --------------- DECISION TREE ---------------
#Training the third classifier
clf3 = tree.DecisionTreeClassifier()
print('Decision Tree:')
evaluate_classifier(clf3)

# --------------- K-NN ---------------
#Training the forth classifier
k_neighbor = 5
clf4 = KNeighborsClassifier(k_neighbor)
print('k-NN (k = ' + str(k_neighbor) + ') :')
evaluate_classifier(clf4)

# --------------- ADABOOST ---------------
#Training the fifth classifier
clf5 = AdaBoostClassifier()
print('Adaboost:')
evaluate_classifier(clf5)


