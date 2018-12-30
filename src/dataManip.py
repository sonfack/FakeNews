import nltk
import pandas as pd
import re
from nltk.corpus import stopwords

lemma = nltk.WordNetLemmatizer()

def readNews(fileName):
    return pd.read_csv(fileName)

def getCVSHeaders(csvFile):
    heads = []
    for row in csvFile:
        heads.append(row)
    return heads

"""
    Lemmatization : Bringing the word to it base form 
    eg : are === base forme is : be
"""
def lemmatizeWord(word):
    uncapitalize = word.lower()
    return lemma.lemmatize(uncapitalize)

def lemmatizeText(text):
    lemmText = []

    # check the instance of the argurment
    if isinstance(text, str):
        splitText = text.split(" ")
        for word in splitText:
            lemmText.append(lemmatizeWord(word))
    elif isinstance(text, list):
        for word in text:
            lemmText.append(lemmatizeWord(word))
    return lemmText


"""
    Removing stop words from text and create a liste of unique words
    We use nltk.corpus.stopwords of english
    
"""
def removeStopWord(text):
    uniqueWords = []
    splitText = text.lower().split(" ")

    # get list of unique word
    splitText = list(set(splitText))
    # load stop words
    stop_words = stopwords.words('english')
    # Remove stop words
    filterWords = [re.sub('\W+',' ', word) for word in splitText if word not in stop_words]
    filterWords = ' '.join(filterWords).split()

    return filterWords