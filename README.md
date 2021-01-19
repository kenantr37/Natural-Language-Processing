# Natural Language Processing By using Twitter Gender Data
### A short summary for Natural Language Processing Steps
- (1) Data Cleaning:<br>
In data cleaning process, we remove special characters, symbols, punctuation, html tags<> etc from the raw data which contains no information for the model to learn, these are simply noise in our data.
- (2) Preprocessing of data:<br>
Preprocessing of data is a data mining technique that involves transforming the raw data into an understandable format(Like making all words low cases).
- (3) Tokenization:<br>
Tokenization is the process of breaking up text document into individual words called tokens.
- (4) Stop words removal:<br>
Stop words are common words that do not contribute much of the information in a text document. Words like ‘the’, ‘is’, ‘a’ have less value and add noise to the text data.
- (5) Lemmatization:<br>
Lemmatization does the same thing as stemming, converting a word to its root form but with one difference i.e., the root word in this case belongs to a valid word in the language. For example the word caring would map to ‘care’ and not ‘car’ as the in case of stemming.
- (6) Bag of words(BOW):<br>
It is one of the simplest text vectorization techniques. The intuition behind BOW is that two sentences are said to be similar if they contain similar set of words.

### Summary of my project
In my project, I tried to learn Natural Language Processing Basics from Twitter users datas. To use NLP, I choosed gender type as y (male of female) and for x, among the plent of features I choosed users descriptions. At the final result, I could get %63.88 accuracy of the Gender prediction by using Naive Bayes Algroithm. In some of developers examples from internet, they could get better result by Linear Regression but Despite their results, I could get %12 accuracy result by using Linear Regression.

![Screenshot_1](https://user-images.githubusercontent.com/44119225/105025519-79b0fd00-5a5e-11eb-9863-6244313448a7.jpg)


For the last step, I wanted to look at how many genders my algorithm could predict accurate by using Confusion Matrix.My opinion from the confusion matrix is that eventhough my algorithm could predict women genders(1) along with high accuracy, prediction of the men(0) amount is terrible.I need to train my algorithm or I can change the learning model.

![Figure_1](https://user-images.githubusercontent.com/44119225/105025604-9d744300-5a5e-11eb-8805-4d96c551d94e.png)
