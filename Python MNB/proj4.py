import sys
import nltk 
from nltk.corpus import movie_reviews
nltk.download('movie_reviews')
from random import shuffle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import string

def build_raw_data():
	raw_data = []
	target_names = []
	stopWords = set(stopwords.words('english'))
	puncWords = string.punctuation

	for category in movie_reviews.categories():
		for fileid in movie_reviews.fileids(category):
			review_words = movie_reviews.words(fileid)
			review_text = ''
			for word in review_words:
				if word not in stopWords and word not in puncWords:
					review_text += ' ' + word
			tokenizer = nltk.tokenize.WhitespaceTokenizer()
			tokens = tokenizer.tokenize(review_text)
			stemmer = nltk.stem.WordNetLemmatizer()
			tokens = " ".join(stemmer.lemmatize(token) for token in tokens)
			review_dictionary = {'text':tokens, 'sentiment':category}
			raw_data.append(review_dictionary) 
	return raw_data

def feature_selection(train_text):
	tfidf = TfidfVectorizer(min_df = 70, max_df = 1300, ngram_range = (1,2))
	tfidfm = tfidf.fit(train_text) 
	return tfidfm


def text_to_vector(texts, tfidfm):
	X = tfidfm.transform(texts) 
	return X
	
def split_data(all_data):
	np.random.seed(10)
	np.random.shuffle(all_data)
	testSet = []
	trainSet = []
	for i in range(2000):
		if(0 <= i <= 1499):
			trainSet.append(all_data[i])
		else:
			testSet.append(all_data[i])
	return trainSet, testSet

def model():
	data = build_raw_data()
	trainSet, testSet = split_data(data)
	train_text = []
	train_sentiment = []
	test_text = []
	test_sentiment = []
	for i in trainSet:
		train_text.append(i.get('text'))
		train_sentiment.append(i.get('sentiment'))

	for i in testSet:
		test_text.append(i.get('text'))
		test_sentiment.append(i.get('sentiment'))

	tfidfm = feature_selection(train_text)
	X = text_to_vector(train_text, tfidfm)
	Y = train_sentiment
	

	clf = MultinomialNB()
	clf.fit(X, Y)

	a = tfidfm.transform(test_text)
	prediction = (clf.predict(a))
	acc = sklearn.metrics.accuracy_score(test_sentiment, prediction, normalize=True)
	print("Classifier Accuracy = %s" % acc)


if __name__ == '__main__':	model()