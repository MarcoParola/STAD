import nltk
import csv
import sys
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, RegexpTokenizer, TreebankWordTokenizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def toString(array):
	str1 = ''
	for items in array:
		#str1 = '\t'.join(item)
		for item in items:
			str1 = str1 + str(item) + '\t'
		str1 = str1 + '\n'
	return str1

# reading KEYWORDS file 
relevant_stems = open("../utils/KEYWORDS.txt","r").read().splitlines()

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('italian')) 
ss = SnowballStemmer('italian')
final_array = []

filecsv = sys.argv[1]
with open(filecsv,'r', encoding='utf-8') as csvtwitter:
	reader = csv.reader(csvtwitter)
	for row in reader:
	
		example_sent = row[10]

		# Tokenization without punctuaction
		word_tokens = tokenizer.tokenize(example_sent)

		# Stop-word Filtering
		filtered_sentence = [w for w in word_tokens if not w in stop_words] 

		# Stemming
		final_sentence = [ss.stem(w) for w in filtered_sentence]
			
		# Stem Filtering
		final_words = [w for w in final_sentence if w in relevant_stems] 
		print(final_words)

		final_array.append(str(final_words))
		
		
# Feature Representation
fileOut = open("out.tsv","w")
vectorizer = CountVectorizer()
vectorizer.fit_transform(relevant_stems)
fileOut.write(toString(vectorizer.transform(final_array).toarray()))
fileOut.close()
