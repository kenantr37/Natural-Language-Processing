# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:46:41 2021

@author: Zeno
"""
import pandas as pd
import numpy as np
import re 
import nltk as npl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("D:/Machine Learning Works/Natural Language Processing/gender_classifier.csv",encoding = "Latin1") # reading data
data = pd.concat([data.gender,data.description],axis = 1) # concatinating gender and text columns
data.dropna(axis = 0,inplace = True) # Dropping NaN values
data.gender = [0 if word == "male" else 1 for word in data.gender] # changing "female" and "male" string data types to int (1 and 0)
#%% Cleaning,Preprocessing,Lemmatizating ...
description_list = [] # to create space matrix to count frequency of the words
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description) # Data cleaning 
    description = description.lower() # (Preprocessing) Making all words low cases
    description = npl.word_tokenize(description) # to collect every single word is called "token"
    
    lemma = npl.WordNetLemmatizer() # to pick the tokens roots up, like (lovely ---> love)
    description = [lemma.lemmatize(word) for word in description] # with for loop we look at the root of the word and for this we seperated words to their cases like (love -> "l","o","v","e")
    description = " ".join(description) # by .join method I concat cases to make them roots of the words
    description_list.append(description) # Finally, I collect final words in description list
#%% Bag of words
max_features = 10 # to exhibit the most used 10 words
count_vectorize = CountVectorizer(max_features=max_features,stop_words="english") # to count 10 words are above
space_matrix = count_vectorize.fit_transform(description_list).toarray() # creating space matrix
print("The most used {} words are {} ".format(max_features,count_vectorize.get_feature_names()))
#%% Creating Machine Learning Model
y = data.gender.values # our dependent column is gender (female or male)
x_train,x_test,y_train,y_test = train_test_split(space_matrix,y,test_size = 0.2,random_state = 42) # splitting data as train and test
naive_model = GaussianNB().fit(x_train,y_train) # I used Naive bayes algorithm here
print("Accuracy of the model : ",naive_model.score(x_test,y_test)) # My accuracy is %63.88
#%% Let's look at the Confusion Matrix
y_prediction = naive_model.predict(x_test).reshape(-1,1)
cm = confusion_matrix(y_test, y_prediction)
sns.heatmap(cm,annot =True , fmt = "d")
plt.title("Gender Prediction-Confusion Matrix from Twitter Descriptions")
plt.show()