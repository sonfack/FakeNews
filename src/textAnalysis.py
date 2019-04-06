import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from pylab import *
from src.dataManip import readNews, labelType, removeStopWord
from string import punctuation
from collections import Counter

fakenews = readNews("../data/header.csv")
print('Nombre d\'article traite', len(fakenews['content']))

content = fakenews['content']
labelType = labelType(fakenews['type'])


"""
    countFakeLength  all length of fake news
    countNotFakeLength  all length of not fake news 
"""

def countWordInArticles():
    countFakeLength = []
    countNotFakeLength = []
    lenghtWordF = []
    lenghtWordNF = []
    countPointF = 0
    countPointNF = 0
    f = 0
    nf = 0
    # load stop words
    stop_words = stopwords.words('english')
    for i in range(len(content)):
        if labelType[i] == 0:
            countPointF = countPointF + countPointuation(content[i])
            f = f + 1
        elif labelType[i] == 1:
            countPointNF = countPointNF + countPointuation(content[i])
            nf = nf + 1
        splitText = content[i].lower().split(" ")
        text = [re.sub('\W+',' ', word) for word in splitText if word not in stop_words]
        if labelType[i] == 0:
            countFakeLength.append(len(text))
            for i in range(len(text)):
                lenghtWordF.append(len(text[i]))
        elif labelType[i] == 1:
            countNotFakeLength.append(len(text))
            for i in range(len(text)):
                lenghtWordNF.append(len(text[i]))

    print('Moyenne du nombre de mots d\'articles Fake News : ', mean(countFakeLength))
    print('Maximum de mot dans un article Fake News : ', max(countFakeLength))
    print('Moyenne du nombre de mots d\'articles Non Fake News : ', mean(countNotFakeLength))
    print('Maximum de mot dans un article Non Fake News : ', max(countNotFakeLength))
    print('\n')
    print('Moyenne de longeur de mots d\'articles Fake News : ', mean(lenghtWordF))
    print('Moyenne de longeur de mots d\'articles Non Fake News : ', mean(lenghtWordNF))
    print('\n')
    print('Nombre moyen de pointuation dans d\'articles Fake News : ', countPointF/f)
    print('Nombre moyen de pointuation dans d\'articles Non Fake News : ', countPointNF/nf)

def countPointuation(text):
    counts = Counter(text)
    punctuation_counts = {k: v for k, v in counts.items() if k in punctuation}
    countPoint = 0
    for k,v in punctuation_counts.items():
        countPoint = countPoint+int(v)

    return countPoint

countWordInArticles()
#countWordInArticles()