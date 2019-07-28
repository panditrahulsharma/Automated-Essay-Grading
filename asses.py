import nltk.data
import pandas as pd
import numpy as np
# Importing packages required for Spell Checking
import re
from collections import Counter
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('punkt')	
nltk.download('brown')	
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import sent_tokenize
	


asses=pd.read_csv('training_set.tsv',delimiter='\t',encoding='latin-1')
dataset=asses[asses['essay_set']==1].iloc[:,[0,1,2,6]]
"""
• Word n-grams - N-grams tokenize the next and treat it as a “bag of words”, where each feature is a count of
how many times a word or combination of words appeared. We usually used unigrams. We applied the tf-idf
transformation to the word counts.
• Part of speech n-grams - We used the Natural Language Toolkit part of speech tagger, and then used these tags
and n-gram features. [2]
• Character counts
• Word counts
• Sentence counts
• Number of mispellings
• Reduced dimention term-vector - We used Latent Semantic Analysis (discussed below) as both an independent
model and a method to reduce the dimention of the term-document matrix, which was then used as features in
the SVM.
"""

import re
def char_count(p):
	p=p.split()
	total_char=0
	for i in p:
		total_char=total_char+len(i)	
	return total_char
	
def word_count(p):
	p=p.split()
	total_word=len(p)
	return total_word

def line_count(p):
	NumberOfLines=len(re.split('\n',p))	
	return NumberOfLines	

def sentense_count(p):
	tokenized_text=sent_tokenize(p)
	return	len(tokenized_text)

def unique_word(p):
	p=p.split()
	q=[]
	for i in p:
		if i not in q:
			q.append(i)
	return len(q)	
		
		
		
		
		
		
		
print("no_char"," word_count","line_count","sentense count","unique_word")	
for i in range(0,len(dataset)):
		print(i," ",char_count(dataset.iloc[i,2])," ",word_count(dataset.iloc[i,2]),"  ",line_count(dataset.iloc[i,2]),"  ",sentense_count(dataset.iloc[i,2]),"  ",unique_word(dataset.iloc[i,2]))
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		








