import csv
import math
import nltk
import numpy as np
import pandas as pd	
import re 
import sys
# nltk.download('stopwords')
# nltk.download('punkt')
from guess_language import guess_language
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer
from nltk.stem import SnowballStemmer
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif


def FindURLs(string): 
	url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
	pics = re.findall('pic.twitter.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)
	return url + pics


def isItalian(string):
	if guess_language(string) == 'it':
		return True
	return False

"""
Function that returns an array with all the stems of the twitter in the file file_name
"""
def preProcessing(file_name):
	fileOut = open("fileFeatures1.csv", "w")
	
	with open(file_name, 'r', encoding='utf-8',) as csvtwitter:
		reader = csv.reader(csvtwitter)
		for row in reader:
			if row[10] != "tweet":
				tweet = row[10]
				classValue = row[11]
				
				if isItalian(tweet):
					# Delete all the urls in the tweet
					for url in FindURLs(tweet):
						tweet = tweet.replace(url, '')
					
					
					
					tweet = tweet.replace('\n',' ')
					
					if tweet != '':
						# writing tweet's text and class on a new file
						fileOut.write('\"'+ tweet + '\"\t' + classValue + '\n')
					
	fileOut.close()
	
"""
MAIN
"""
filecsv = sys.argv[1]
preProcessing(filecsv)
	
