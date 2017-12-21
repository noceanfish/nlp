import string
import os
import nltk
import pandas as pd
import matplotlib as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def text_process(message):
    nopunc = [c for c in message if c not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


if __name__ == '__main__':

    messages1 = [line.rstrip() for line in open('D:/dev/udemy/SMSSpamCollection')]
    print(messages1[1])

    messages2 = pd.read_csv('d:/dev/udemy/SMSSpamCollection', sep='\t', names=['labels', 'message'])
    messages2.describe()
    messages2.groupby('labels').describe()
    messages2['length'] = messages2['message'].apply(len)
    print(messages2.head())

    messages2['length'].plot.hist(bins=200)
    plt.pyplot.show()

    # messages2.hist(column='length', by='labels', bins=100)
    # plt.pyplot.show()

    mess = 'simple messages! Notice : is has punctuation.'
    print(text_process(mess))

    # bow
    bow_transformer = CountVectorizer(analyzer=text_process).fit(messages2['message'])
    print(len(bow_transformer.vocabulary_))
    bow4 = bow_transformer.transform([messages2['message'][3]])
    print(bow4)
    messages_bow = bow_transformer.transform(messages2['message'])
    print(messages_bow.nnz)

    # tfidf
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    tfidf4 = tfidf_transformer.transform(bow4)
    print(tfidf4)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    # classifier
    spam_detect_model = MultinomialNB().fit(messages_tfidf, messages2['labels'])

    # using pipeline
    msg_train, msg_test, label_train, label_test = train_test_split(messages2['message'], messages2['labels'],
                                                                    test_size=0.3)
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(msg_train, label_train)
    predictions = pipeline.predict(msg_test)
    print(classification_report(label_test, predictions))

    print('finish!!!')

