import nltk
import csv
import sys
import numpy as np
import relevantStems #file python with the function to calculate the relevant stems
import util	#file python with util functions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# reading KEYWORDS file
#relevant_stems = open("../utils/KEYWORDS.txt", "r").read().splitlines()
relevant_stems, relevant_weights = relevantStems.getFromFile("KEYWORDS.csv")
final_sentences = []
final_array = []
classes = []

filecsv = sys.argv[1]
# preprocessing on the tweets
final_sentences, classes = util.preProcessing(filecsv)
# Stem Filtering
for final_sentence in final_sentences:
	final_words = []
	final_words = [w for w in final_sentence if w in relevant_stems]
	"""
	for w in final_sentence:
		i = 0
		while i < len(relevant_stems):
			if w == relevant_stems[i]:
				final_words.append(relevant_weights[i])
			i = i + 1
	
    	#print(final_words)
	"""
	final_array.append(str(final_words))
#print(final_array)


# Feature Representation
fileOut = open("fileFeatures1.tsv", "w")
fileClasses = open("fileClasses1.tsv", "w")
vectorizer = TfidfVectorizer(use_idf=True)
vectorizer.fit_transform(relevant_stems,relevant_weights)
x = vectorizer.transform(final_array).toarray()
print(util.FromMatrixToString(x))
fileOut.write(util.FromMatrixToString(x))
fileClasses.write( util.FromMatrixToString(classes) )
fileOut.close()
fileClasses.close()
