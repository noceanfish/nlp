import string
import os
import nltk
import pandas as pd
import matplotlib as plt
import seaborn as sns

from nltk.corpus import stopwords


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

    messages2.hist(column='length', by='labels', bins=100)
    plt.pyplot.show()

    mess = 'simple messages! Notice : is has punctuation.'
    print(text_process(mess))



