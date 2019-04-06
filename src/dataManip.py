import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
import csv
from collections import defaultdict

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



def readFileSpecificColumn(file, numberOfLines):
    columns = defaultdict(list)  # each value in each column is appended to a list
    count = 0
    with open(file) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:
            if count < numberOfLines: # read a row as {column1: value1, column2: value2,...}
                for (k, v) in row.items():  # go over each column name and value
                    #columns[k].append(v)  # append the value into the appropriate list
                    # based on column name k
                    if k == "id" or k == "type":
                        print(k,": ", v)
                count = count + 1
            else:
                break


#readFileSpecificColumn('../data/view.csv', 3)

""""
Labeling 
Fake === 0 
Not Fake === 1
"""
def labelType(listType):
    labelType = []
    for i in range(len(listType)):
        if str(listType[i]).lower() in ["fake", "satire", "unreliable", "conspiracy", "rumor"]:
            labelType.append(0)
        else:
            labelType.append(1)
    return labelType