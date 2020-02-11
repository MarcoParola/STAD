import csv
import math
import operator
import sys
import util	#file python with util functions
from info_gain import info_gain

def idf(stems):
	idf_stems = {}
	N = len(stems)
	print(str(N)+'\n')
	
	idf_stems = { stem:0 for stem in stems if not stem in idf_stems.keys() }
	
	for stem in stems:
		idf_stems[stem] += 1
		
	for stem,val in idf_stems.items():
		idf_stems[stem] = math.log10(N / float(val))
		
	return idf_stems
	

def tf_idf(idf_stems, tweet_stems):
	tf_tweet_stems = { stem:0 for stem,val in idf_stems.items() }
			
	for stem in tweet_stems:
		tf_tweet_stems[stem] += 1
	
	for stem, val in tf_tweet_stems.items():
		if val != 0:
			tf_tweet_stems[stem] = val * idf_stems[stem]
	
	return tf_tweet_stems


def tf_idf_calculate(final_array, tweets_stems, classes):
	idf_stems = []
	tf_tweets_stems = []
	idf_stems = idf(final_array)
	for tweet_stems in tweets_stems:
		tf_tweets_stems.append( tf_idf(idf_stems, tweet_stems) )
	
	#print(tf_tweets_stems)
	#create_stems_csv("stems_table.tsv",tf_tweets_stems, classes)
	return tf_tweets_stems

def info_gain_calculate(tf_tweets_stems, classes):
	keys = []
	info_gains = {}
	for i in tf_tweets_stems[0].keys():
		keys.append(i)
	N = len(tf_tweets_stems)
	M = len(keys)
	i = 0
	j = 0

	while i < M:
		values = []
		while j < N:
			values.append(tf_tweets_stems[j][keys[i]])
			j = j + 1
		ig = info_gain.info_gain(classes, values)
		info_gains[keys[i]] = ig
		"""
		if ig > 0.02:
			print(keys[i] + ': ' + str(ig))
		"""
		i = i + 1
		j = 0
		
	return info_gains
	
	
def choose_relevant_stems(info_gains, limit):
	# create the list of stems sorted by info_gain
	sort_ig_stems = sorted(info_gains.items(), key=operator.itemgetter(1), reverse=True)
	# return only the first 'limit' stems in the list (they are the most relevant)
	rel_stems = sort_ig_stems[:limit]
	for item in rel_stems:
		print(item[0] +': '+ str(item[1]))
	#print(rel_stems)
	return rel_stems
	
"""
# function to write in a file the summary of the stems for each tweet
def create_stems_csv(file_name, tf_tweets_stems, classes):
	fileOut = open(file_name, "w")
	attributes_names = []
	i = 0
	
	for K in tf_tweets_stems[0].keys():
		attributes_names.append(K)
	attributes_names.append( "class" )
	fileOut.write( util.FromArrayToString(attributes_names) )
	
	for tweet_stems in tf_tweets_stems:
		values = []
		for stem,val in tweet_stems.items():
			values.append(val)
		values.append( classes[i] )
		fileOut.write( FromArrayToString(values) )
		i = i + 1
		
	fileOut.close()
"""

def getFromFile(file_name):
	relevant_stems = []
	relevant_weights = []
	with open(file_name, 'r', encoding='utf-8') as csvstems:
		reader = csv.reader(csvstems)
		for row in reader:
			relevant_stems.append(row[0])
			relevant_weights.append(row[1])
			
	return relevant_stems,relevant_weights

"""
MAIN
"""

final_sentences = []
final_array = []
tweets_stems = []
tf_tweets_stems = []
classes = []
info_gains = []

filecsv = sys.argv[1]
final_sentences, classes = util.preProcessing(filecsv)

for final_sentence in final_sentences:			
	for stem in final_sentence:
		final_array.append(stem)
	tweets_stems.append(final_sentence)

# Calculate tf_idf index for each stem found			
tf_tweets_stems = tf_idf_calculate(final_array, tweets_stems, classes)
# Calculate info gain for each stem
info_gains = info_gain_calculate(tf_tweets_stems, classes)
# Choosing the relevant stems
rel_stems = choose_relevant_stems(info_gains, 30)
fileOut = open("KEYWORDS.csv",'w')
for item in rel_stems:
	fileOut.write(item[0] +','+ str(item[1]) +'\n')
fileOut.close()
