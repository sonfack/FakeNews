from sklearn.feature_extraction.text import TfidfVectorizer

'''
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
'''

def tf_idf(corpus):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(corpus)

#print(tf_idf(corpus).todense())