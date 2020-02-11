import csv
import math
import nltk
import numpy as np
import pandas as pd	
import re 
import sys
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer
from nltk.stem import SnowballStemmer
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif

def FromArrayToString(array):
	s = ''
	for value in array:
		s = s + str(value) + '\t'
	s = s + '\n'
	return s
	
def FromMatrixToString(matrix):
    str1 = ''
    for items in matrix:
        #str1 = '\t'.join(item)
        for item in items:
            str1 = str1 + str(item) + '\t'
        str1 = str1 + '\n'
    return str1

def FindURLs(string): 
	url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
	return url 


"""
Function that returns an array with all the stems of the twitter in the file file_name
"""
def preProcessing(file_name):
	tokenizer = RegexpTokenizer(r'\w+')
	stop_words = set(stopwords.words('italian'))
	ss = SnowballStemmer('italian')

	final_array = []
	classes = []
	
	with open(file_name, 'r', encoding='utf-8') as csvtwitter:
		reader = csv.reader(csvtwitter)
		for row in reader:
			if row[11] != "class":
				example_sent = row[10]
				
				# Delete all the urls in the tweet
				for url in FindURLs(example_sent):
					example_sent = example_sent.replace(url, '')

				# Save also the class
				classes.append(row[11])

				# Tokenization without punctuaction
				word_tokens = tokenizer.tokenize(example_sent)

				# Stop-word Filtering
				filtered_sentence = [w for w in word_tokens if not w in stop_words]

				# Stemming
				final_sentence = [ss.stem(w) for w in filtered_sentence]
				
				final_array.append(final_sentence)
				
	return final_array, classes
	
