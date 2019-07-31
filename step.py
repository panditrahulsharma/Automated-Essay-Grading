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
and pass dataframe in function it automatcally extract features

step-2

when we extract features than after 