# Automated-Essay-Grading
Automated essay scoring (AES) is the use of specialized computer programs to assign grades to essays written in an educational setting. It is a method of educational assessment and an application of natural language processing.

"""
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
#12.create a bow model
#13.vectirzation with 10,000 bow features
#14 features extraction
#15. Satndered scalling of features scalling model
#16. add bow and features_extraction 
#17. apply train_test_split 
# use 
 LinearRegression, RandomForestRegressor,SVR ,DecisionTreeRegressor
Lasso,Ridge
"""

"""
step-1 

first extract features using extarct.py files import it 
and pass dataframe in function it automatcally extract features and pass it into standerd scalling

step-2

when we extract features than remove stop word and lemitization from essay

next process CountVectorizer of clean essay

->vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    
->count_vectors = vectorizer.fit_transform(dataset.iloc[:,2])
   
X_cv = count_vectors.toarray()

step-3 

X_cv contains only bog features and now add extract features in X_cv

total features=X_cv+extract_features

#the bow(bag of word data contains 10000) rows and features_set1 contains 9 columns
#so we add column (bom+features_extraction) using np array with axis=1
->features = np.concatenate((np.array(features_set1), X_cv), axis = 1)
->labels = np.array(dataset.iloc[:,1])

step-4
now its times train_test_split and fit into model 

i am use diff of linear model but i think in my own assumptions the linear_resgression model work better among these algo

i have make all model pickle file and load into server.py
