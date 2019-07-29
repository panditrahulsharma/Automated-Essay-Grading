# For this project, only essay set 1 was used for analysis and model creation.

# Features:
# 1. Bag of Words (BOW) counts (10000 words with maximum frequency)
# 2. Number of characters in an essay
# 3. Number of words in an essay
# 4. Number of sentences in an essay
# 5. Average word length of an essay
# 6. Number of lemmas in an essay
# 7. Number of spellng errors in an essay
# 8. Number of nouns in an essay
# 9. Number of adjectives in an essay
# 10. Number of verbs in an essay
# 11. Number of adverbs in an essay
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score

asses=pd.read_csv("training_set.tsv",delimiter='\t',encoding='latin-1')
dataset=asses[asses['essay_set']==1].iloc[:,[2,6]]

# 1. Number of characters in an essay

def char_count1(p):
	p=p.split()
	total_char=0
	for i in p:
		total_char=total_char+len(i)	
	return total_char

# 2. Number of words in an essay

def word_count(p):
	p=p.split()
	total_word=len(p)
	return total_word

# 3. Number of sentences in an essay
def sentense_count(p):
	tokenized_text=nltk.sent_tokenize(p)	
	return	len(tokenized_text)

# 4. Average word length of an essay

def avg_word_len(p):
	p=p.split()
	a=sum([len(i) for i in p])/len(p)
	return a


# 5. Number of unique_word in an essay

def unique_word(p):
	p=p.split()
	q=[]
	for i in p:
		if i not in q:
			q.append(i)
	return len(q)	
		
	
# 6. Number of spellng errors in an essay

"""
 #big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg 
        and lists of most frequent words from Wiktionary and the British National Corpus.
            It contains about a million words.
  
"""

def count_spell_error(essay):
    
       
	    data = open('big.txt').read()
    
	    words_ = re.findall('[a-z]+', data.lower())
	    
	    word_dict = collections.defaultdict(lambda: 0)
	    #orderd dict ha                   
	    for word in words_:
	        word_dict[word] += 1
	                       
	    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
	    clean_essay = re.sub(r'[0-9]', '', clean_essay)                  
	    mispell_count = 0
	    
	    words = clean_essay.split()
	                        
	    for word in words:
	        if not word in word_dict:
	            mispell_count += 1
	    return mispell_count
	

		
# calculating number of lemmas per essay

def count_lemmas(essay):
    
    tokenized_sentences = nltk.word_tokenize(essay)      
    print(tokenized_sentences)
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence) 
        print(tagged_tokens)
        for token_tuple in tagged_tokens:
        
            pos_tag = token_tuple[1]
            if pos_tag.startswith('N'): 
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
    
    lemma_count = len(set(lemmas))
    
    return lemma_count

#length of paragraph
def paragraph_len(p):
	return len(p)
#paragraph length contains white space comma and othres 

	

# 8. Number of nouns in an essay
# 9. Number of adjectives in an essay
# 10. Number of verbs in an essay
# 11. Number of adverbs in an essay
"""
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
"""
def count_pos(essay):
    
    tokenized_sentences = nltk.word_tokenize(essay)
    #convert paragraph into word
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        #token_touple convert into touple with word and their noun,adjective,verb ,adverd
        for token_tuple in tagged_tokens:
							
            pos_tag = token_tuple[1]
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            
    return noun_count, adj_count, verb_count, adv_count
   
print(count_pos(dataset.iloc[0,0]))



# splitting data into train data and test data (70/30)


def get_count_vectors(essays):
    
    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    
    count_vectors = vectorizer.fit_transform(essays)
    
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, count_vectors




feature_names_cv, count_vectors = get_count_vectors(dataset.iloc[:,0])

X_cv = count_vectors.toarray()

y_cv = dataset.iloc[:,1].as_matrix()

featurs_train, features_test, labels_train, labels_test = train_test_split(X_cv, y_cv, test_size = 0.3,random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import metrics

#first takes linearregression
lr=LinearRegression()
lr.fit(featurs_train,labels_train)
pred_LinearRegression=lr.predict(features_test)

print(pd.DataFrame({'actual':labels_test,'prediction':pred_LinearRegression}))
print("MAE",metrics.mean_absolute_error(labels_test,pred_LinearRegression))
print("MSE",metrics.mean_squared_error(labels_test,pred_LinearRegression))
print("RMSE",np.sqrt(metrics.mean_squared_error(labels_test,pred_LinearRegression)))
print("rmse is less or equal of 10% of mean(labels)")
print("labels mean=",np.mean(y_cv))


x=list(range(0,len(labels_test)))
plt.plot(x,labels_test,color='r',label='actual marks')
plt.plot(x,pred_LinearRegression,label='pred marks')
plt.xlabel("no of assay")
plt.ylabel("marks per assay")
plt.legend()
plt.show()


#2.random forest
rf=RandomForestRegressor(n_estimators=5,random_state=0)
rf.fit(featurs_train,labels_train)
pred_RandomForestRegressor=rf.predict(features_test)


print(pd.DataFrame({'actual':labels_test,'prediction':pred_RandomForestRegressor}))
print("MAE",metrics.mean_absolute_error(labels_test,pred_RandomForestRegressor))
print("MSE",metrics.mean_squared_error(labels_test,pred_RandomForestRegressor))
print("RMSE",np.sqrt(metrics.mean_squared_error(labels_test,pred_RandomForestRegressor)))
print("rmse is less or equal of 10% of mean(labels)")
print("labels mean=",np.mean(y_cv))

x=list(range(0,len(labels_test)))
plt.plot(x,labels_test,color='r',label='actual marks')
plt.plot(x,pred_RandomForestRegressor,label='pred marks')
plt.xlabel("no of assay")
plt.ylabel("marks per assay")
plt.legend()
plt.show()

#3.from sklearn.svm import SVR 
svr=SVR(kernel='rbf',degree=3)
svr.fit(featurs_train,labels_train)
pred_svr=svr.predict(features_test)


print(pd.DataFrame({'actual':labels_test,'prediction':pred_svr}))
print("MAE",metrics.mean_absolute_error(labels_test,pred_svr))
print("MSE",metrics.mean_squared_error(labels_test,pred_svr))
print("RMSE",np.sqrt(metrics.mean_squared_error(labels_test,pred_svr)))
print("rmse is less or equal of 10% of mean(labels)")
print("labels mean=",np.mean(y_cv))


plt.plot(x,labels_test,color='r',label='actual marks')
plt.plot(x,pred_svr,label='pred marks')
plt.xlabel("no of assay")
plt.ylabel("marks per assay")
plt.legend()
plt.show()

#from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(featurs_train,labels_train)
pred_dt=dt.predict(features_test)


print(pd.DataFrame({'actual':labels_test,'prediction':pred_dt}))
print("MAE",metrics.mean_absolute_error(labels_test,pred_dt))
print("MSE",metrics.mean_squared_error(labels_test,pred_dt))
print("RMSE",np.sqrt(metrics.mean_squared_error(labels_test,pred_dt)))
print("rmse is less or equal of 10% of mean(labels)")
print("labels mean=",np.mean(y_cv))


plt.plot(x,labels_test,color='r',label='actual marks')
plt.plot(x,pred_dt,label='pred marks')
plt.xlabel("no of assay")
plt.ylabel("marks per assay")
plt.legend()
plt.show()


#
