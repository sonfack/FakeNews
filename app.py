import re
import pickle
import datetime
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from src.bagOfWords import bagOfWords
from src.dataManip import readNews, lemmatizeText, removeStopWord, getCVSHeaders

fakenews = readNews("./data/header.csv")
def createCorpus():
    corpus = []
    articleContain = fakenews['content']
    for content in articleContain:
        content = re.sub('\W+',' ', content)
        #content = ' '.join(content)
        corpus.append(content)
    return corpus

"""
https://www.bullshido.net/list-bs-fake-biased-news-sites/

Fake News (tag fake = 0): Sources that entirely fabricate information, disseminate deceptive content, or grossly distort 
actual news reports.

Satire (tag satire = 0): Sources that use humor, irony, exaggeration, ridicule, and false information to comment on current 
events.

Clickbait (tag clickbait = 1): Sources that provide generally credible content, but use exaggerated, misleading, or 
questionable headlines, social media descriptions, and/or images.

Proceed With Caution (tag unreliable = 0): Sources that may be reliable but whose contents require further verification.

Conspiracy Theory (tag conspiracy = 0): Sources that are well-known promoters of kooky conspiracy theories.

Hate News (tag hate): Sources that actively promote racism, misogyny, homophobia, and other forms of discrimination.

Rumor Mill (tag rumor = 0): Sources that traffic in rumors, gossip, innuendo, and unverified claims.

Extreme Bias (tag bias): Sources that come from a particular point of view and may rely on propaganda, decontextualized 
information, and opinions distorted as facts.

*Political (tag political): Sources that provide generally verifiable information in support of certain points of view 
or political orientations.

"""


def labelType(listType):
    labelType = []
    for i in range(len(listType)):
        if str(listType[i]).lower() in ["fake", "satire", "unreliable", "conspiracy", "rumor"]:
            labelType.append(0)
        else:
            labelType.append(1)
    return labelType


def testLR():
    X, y = load_iris(return_X_y=True)
    print(X)
    print(y)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial').fit(X, y)
    print(X[:2, :])
    print(clf.predict(X[:2, :]))
    print(clf.predict_proba(X[:2, :]))
    print(clf.score(X, y))

def logisticRegrestion(X, y):
    N = len(X)
    n = N //3
    X_test = X[:n-1]
    X_train = X[n:]
    y_test = y[:n-1]
    y_train = y[n:]
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial').fit(X_train, y_train)
    file = str(datetime.datetime.now())
    folder = "./model"
    filepath = os.path.join(folder, file)
    if not os.path.exists(folder):
        os.mkdir(folder)
    outfile = open(str(filepath), 'wb+')
    pickle.dump(clf, outfile)
    outfile.close()
    print(clf.predict(X_test))
    print(y_test)
    print(clf.predict_proba(X_test))
    print(clf.score(X_train, y_train))

def main():
    #print(getCVSHeaders(fakenews))
    label = labelType(fakenews['type'])
    print(label)
    corpus = createCorpus()
    vectorize = bagOfWords(corpus)

    logisticRegrestion(vectorize, label)

    #print(vectorize.shape)
    #print(vectorize)


if __name__ == '__main__':
    main()
