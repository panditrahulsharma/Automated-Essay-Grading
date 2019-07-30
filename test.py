#load pickle file
import pickle
scale_pkl=open('scalling.pkl','rb')
sc= pickle.load(scale_pkl)

scale_pkl1=open('extract_feature.pkl','rb')
extract_feature = pickle.load(scale_pkl1)

scale_pkl2=open('vectorizer.pkl','rb')
vectorizer= pickle.load(scale_pkl2)

scale_pkl3=open('regression.pkl','rb')
regression= pickle.load(scale_pkl3)



#this is string come to the gui
test_data=pd.Series(dataset.iloc[1,0])
#make dataframe and series using sting
test_dataframe=pd.DataFrame(test_data,columns=['essay'])



test_dataframe1= extract_feature(test_dataframe)
#method call using pickle file
test_dataframe1=test_dataframe1.iloc[:,1:]

#scalling of the datset
test_dataframe1=sc.transform(test_dataframe1)



essay_set1=[]
for i in range(0,len(test_dataframe)):
	essay=re.sub('[^a-zA-Z]', ' ',test_dataframe['essay'][i])
	essay=essay.lower()
	essay=essay.split()
	#stop word i am ..
	essay = [word for word in essay if not word in set(stopwords.words('english'))]
	ps = PorterStemmer()
	#port stemmer convert word into root word
	essay = [ps.stem(word) for word in essay]
	essay=' '.join(essay)
	essay_set1.append(essay)


#this is clean essay removed stopword and stemming
test_dataframe['essay']=pd.DataFrame(essay_set1)

#now counvactorizer using vectorizer.pickle file
count_vector1=vectorizer.transform(test_dataframe.iloc[:,0])

X_cv1 = count_vector1.toarray()


#now add extract_features and bog
features1= np.concatenate((np.array(test_dataframe1), X_cv1), axis = 1)


pred=regression.predict(features1)













