#this is string come to the gui
test_data=pd.Series(dataset.iloc[1,0])

#make dataframe and series using sting
test_dataframe=pd.DataFrame(test_data,columns=['essay'])





def extract_features(data):
    
    features = data.copy()
    
    features['char_count'] = features['essay'].apply(char_count)
    
    features['word_count'] = features['essay'].apply(word_count)
    
    features['sent_count'] = features['essay'].apply(sentense_count)
    
    features['avg_word_len'] = features['essay'].apply(avg_word_len)
    
    #features['lemma_count'] = features['essay'].apply(count_lemmas)
    
    features['spell_err_count'] = features['essay'].apply(count_spell_error)
    
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))
    
    return features



test_dataframe= extract_features(test_dataframe)
dd



m= extract_features(p1)
