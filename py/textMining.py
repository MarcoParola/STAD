import nltk
import csv
import sys
import numpy as np
import util	#file python with util functions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# reading KEYWORDS file
relevant_stems = open("../utils/KEYWORDS.txt", "r").read().splitlines()
final_sentences = []
final_array = []
classes = []

filecsv = sys.argv[1]
# preprocessing on the tweets
final_sentences, classes = util.preProcessing(filecsv)
# Stem Filtering
for final_sentence in final_sentences:
    final_words = [w for w in final_sentence if w in relevant_stems]
    print(final_words)

    final_array.append(str(final_words))


# Feature Representation
fileOut = open("fileFeatures1.tsv", "w")
fileClasses = open("fileClasses1.tsv", "w")
vectorizer = CountVectorizer()
vectorizer.fit_transform(relevant_stems)
x = vectorizer.transform(final_array).toarray()
fileOut.write( util.FromMatrixToString(vectorizer.transform(final_array).toarray()) )
fileClasses.write( util.FromMatrixToString(classes) )
fileOut.close()
fileClasses.close()
