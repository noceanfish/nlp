import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

yelp = pd.read_csv('d:/Dev/udemy/yelp.csv')
print(yelp.head())
print(yelp.info())
print(yelp.describe())

yelp['text length'] = yelp['text'].apply(len)

sns.set_style('white')
g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, 'text length', bins=50)
plt.show()

sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')
plt.show()

sns.countplot(x='stars', data=yelp, palette='rainbow')
plt.show()

stars = yelp.groupby('stars').mean()
print(stars)

print(stars.corr())

sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)
plt.show()

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
print(yelp_class.info())

x = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

nb = MultinomialNB()
nb.fit(x_train, y_train)
predictions = nb.predict(x_test)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

pipe = Pipeline([('bow', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB())])

x = yelp_class['text']
y = yelp_class['stars']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

pipe.fit(x_train, y_train)
predictions = pipe.predict(x_test)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))