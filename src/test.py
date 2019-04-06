import pandas as pd
import re
import pickle
import datetime
import os

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Decission Tree
from sklearn import tree
from sklearn.metrics import confusion_matrix

from src.bagOfWords import bagOfWords
from src.dataManip import readNews, lemmatizeText, removeStopWord, getCVSHeaders, readFileSpecificColumn
from src.tf_idf import tf_idf

"""
key google api : AIzaSyAHM2BO5oMzQ3QAnR_go6WxDMJ9TJGpJSE

pip install google-api-python-client

<script>
  (function() {
    var cx = '017411662308276300777:dhoerkg7vwi';
    var gcse = document.createElement('script');
    gcse.type = 'text/javascript';
    gcse.async = true;
    gcse.src = 'https://cse.google.com/cse.js?cx=' + cx;
    var s = document.getElementsByTagName('script')[0];
    s.parentNode.insertBefore(gcse, s);
  })();
</script>
<gcse:search></gcse:search>

"""
fakenews = readNews("./data/header.csv")


def createCorpus():
    corpus = []
    articleContain = fakenews['content']
    for content in articleContain:
        content = re.sub('\W+', ' ', content)
        # content = ' '.join(content)
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

Rumor Mill (tag   = 0): Sources that traffic in rumors, gossip, innuendo, and unverified claims.

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


# Test Logistic regrestion
def testLR(X, numberToTest, modelfile):
    X_test = X[:numberToTest]
    folder = "./model"
    filepath = os.path.join(folder, modelfile)
    pickle_off = open(filepath, "rb")
    model = pickle.load(pickle_off)
    clf = model['model']
    print(clf.predict(X_test))
    print(clf.predict_proba(X_test))


# Logistic regrestion
def logisticRegrestion(X, y, vocabulary, extracteur):
    N, col = X.shape
    n = N // 3
    X_test = X[:3]

    X_train = X[n:]

    y_test = y[:3]
    y_train = y[n:]

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

    file = "LogisticRegrestion" + "_" + extracteur + "_" + str(datetime.datetime.now())
    folder = "./model"
    filepath = os.path.join(folder, file)
    if not os.path.exists(folder):
        os.mkdir(folder)
    outfile = open(str(filepath), 'wb+')
    myModel = {'model': clf, 'vocabulary': vocabulary}
    pickle.dump(myModel, outfile)
    outfile.close()
    print(myModel['vocabulary'])
    print(clf.predict(X_test))
    print(y_test)
    print(clf.predict_proba(X_test))
    print(clf.score(X_train, y_train))


# Test decission tree
def testDT(X_test, y_test, modelfile):
    folder = "./model"
    # filepath = os.path.join(folder, modelfile)

    # pickle_off = open(filepath, "rb")
    # clf = pickle.load(pickle_off)

    filepath = os.path.join(folder, modelfile)
    pickle_off = open(filepath, "rb")
    model = pickle.load(pickle_off)
    clf = model['model']
    # vocabulary = model['vocabulary']

    print(clf.score(X=X_test, y=y_test))
    y_predict = clf.predict(X_test)
    print(y_predict)
    print(clf.predict_proba(X_test))
    print(pd.DataFrame(
        confusion_matrix(y_test, y_predict),
        columns=['Predicted Fake', 'Predicted Not Fake'],
        index=['True Fake', 'True Not Fake']
    ))


# Decission Tree
def decissionTree(X, y, extracteur):
    N, col = X.shape
    n = N // 3
    # X_test = X[:10]

    X_train = X[n:]

    # y_test = y[:10]
    y_train = y[n:]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X=X_train, y=y_train)
    file = "DecisionTree" + "_" + extracteur + "_" + str(datetime.datetime.now())
    folder = "./model"
    filepath = os.path.join(folder, file)
    if not os.path.exists(folder):
        os.mkdir(folder)
    outfile = open(str(filepath), 'wb+')
    myModel = {'model': clf, 'vocabulary': ""}
    pickle.dump(myModel, outfile)
    outfile.close()


def main():
    label = labelType(fakenews['type'])

    corpus = createCorpus()
    vectorizebow, vocabulary = bagOfWords(corpus)
    vectorizetf = tf_idf(corpus)

    X_testtf = vectorizetf[:10]
    X_testbow = vectorizebow[:10]
    y_test = label[:10]
    print(y_test)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    testDT(X_testtf, y_test, "DecisionTree_tf")
    print("******************************************")
    testDT(X_testbow, y_test, "DecisionTree_bow")
    print("******************************************")
    testLR(X_testtf, 10, "LogisticRegrestion_tf")
    print("******************************************")
    testLR(X_testbow, 10, "LogisticRegrestion_bow")

    '''
    label = labelType(fakenews['type'])

    corpus = createCorpus()
    vectorizebow, vocabulary = bagOfWords(corpus)
    vectorizetf = tf_idf(corpus)
    decissionTree(vectorizebow, label, "bow")
    decissionTree(vectorizetf, label, "tf")
    logisticRegrestion(vectorizebow, label, vocabulary, "bow")
    logisticRegrestion(vectorizetf, label, vocabulary, "tf")


    bel = labelType(fakenews['type'])
    print(label[:10])

    corpus = createCorpus()
    vectorize, vocabulary = bagOfWords(corpus)
    decissionTree(vectorize, label)

    #####################################################

    #print(getCVSHeaders(fakenews))
    label = labelType(fakenews['type'])

    #print(label)
    corpus = createCorpus()
    vectorize, vocabulary = bagOfWords(corpus)

    #testLR(vectorize, 5, "modelLR")

    filecsv = "./data/header.csv"
    #readFileSpecificColumn(filecsv, 5)


    logisticRegrestion(vectorize, label, vocabulary)
    #readFileSpecificColumn(filecsv, 3)
    #print(vectorize.shape)
    #print(vectorize)
    '''


if __name__ == '__main__':
    main()
