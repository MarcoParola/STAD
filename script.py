import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

example_sent = "Alluvione 2014 a Genova, non c'era l'obbligo di dare l'allerta blocco tempesta."

print(example_sent)
ss = SnowballStemmer('italian')
relevant_stems = ['acqua', 'allag', 'allarm', 'allert', 'alluvion', 'blocc', 'chiud', 'chius', 'dann', 'devast', 'disastr', 'disp', 'emergent', 'esond', 'evacu', 'ferm', 'fium', 'fran', 'maltemp', 'mete', 'metr', 'mm', 'mort', 'nubifrag', 'paur', 'pericol', 'pien', 'piogg', 'piov', 'risc', 'scuol', 'sfoll', 'sicurezz', 'sommers', 'tempest', 'temp', 'temporal', 'terribil', 'torrent', 'traged', 'tren', 'vittim']

# Tokenization without punctuaction
tokenizer = RegexpTokenizer(r'\w+')

word_tokens = tokenizer.tokenize(example_sent)
print(word_tokens)

# Stop-word Filtering
stop_words = set(stopwords.words('italian')) 

filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
   
print(filtered_sentence) 

# Stemming
final_sentence = []

for w in filtered_sentence:
    final_sentence.append(ss.stem(w))
    print(ss.stem(w))

print(final_sentence)

vectorizer = CountVectorizer() 

x = vectorizer.fit_transform(relevant_stems)

istToStr = ' '.join(map(str, final_sentence)) 

print(vectorizer.transform([istToStr]).toarray())


'''
# reading csv file 
with open(filename, 'r') as csvfile: 
    reader = csv.reader(csvfile)
    for row in reader:
        print(row[10])
        #mapping this string in a prefixed vector
        print(vectorizer.transform([row[10]]).toarray())


'''
