# FakeNews Detection from articles

## Text processing 
Before sending our content to any machine learning model, we first have
 to do some basic text processing on the content :  
1. Remove **stopword**
2. Lemmatization:  
    We bring all the words to their base form  
    eg ate ===> eat 
3. Tokenization:  
    We bring all lemma to the root form  
## Text to vector 
In order to be able to use the text we have to transform the to vectors
 and use Machine Learning model for prediction. 
 ### Bag of words
 