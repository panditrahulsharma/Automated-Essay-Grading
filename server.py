from flask import Flask, request, render_template, url_for

import pickle
import pandas as pd
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
from extract import extract_features
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

scale_pkl=open('scale.pkl','rb')
sc= pickle.load(scale_pkl)


scale_pkl2=open('vectorizer.pkl','rb')
vectorizer= pickle.load(scale_pkl2)

scale_pkl3=open('regression.pkl','rb')
regression= pickle.load(scale_pkl3)


scale_pkl4=open('Random_forest.pkl','rb')
rf= pickle.load(scale_pkl4)


scale_pkl5=open('SVR.pkl','rb')
svr= pickle.load(scale_pkl5)

scale_pkl6=open('Decision_tree.pkl','rb')
dt= pickle.load(scale_pkl6)

scale_pkl7=open('Ridge.pkl','rb')
ridge= pickle.load(scale_pkl7)

scale_pkl8=open('Lasso.pkl','rb')
lasso= pickle.load(scale_pkl8)

app = Flask(__name__)

@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form["essay"]
    test_data=pd.Series(form_data)
    test_dataframe=pd.DataFrame(test_data,columns=['essay'])
    test_dataframe1= extract_features(test_dataframe)
    test_dataframe1=test_dataframe1.iloc[:,1:]
    test_dataframe1=sc.transform(test_dataframe1)
    essay_set1=[]
    for i in range(0,len(test_dataframe)):
    	essay=re.sub('[^a-zA-Z]', ' ',test_dataframe['essay'][i])
    	essay=essay.lower()
    	essay=essay.split()
    	essay = [word for word in essay if not word in set(stopwords.words('english'))]
    	ps = PorterStemmer()
    	essay = [ps.stem(word) for word in essay]
    	essay=' '.join(essay)
    	essay_set1.append(essay)


    test_dataframe['essay']=pd.DataFrame(essay_set1)
    count_vector1=vectorizer.transform(test_dataframe.iloc[:,0])
    X_cv1 = count_vector1.toarray()
    features1= np.concatenate((np.array(test_dataframe1), X_cv1), axis = 1)
    pred=regression.predict(features1)

    return render_template("index.html",status=pred[0])

if __name__ == "__main__":
    app.run(debug=True)
