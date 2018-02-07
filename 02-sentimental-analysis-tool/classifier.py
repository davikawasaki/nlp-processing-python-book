import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

# Lemmatization: turning a word into a base form (dogs -> dog)
wordnet_lemmatizer = WordNetLemmatizer()

# Setting stopwords from list to reduce processing time and dimensionality
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# Loading reviews with a XML parser (BeautifulSoup with lxml)
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), "lxml")
positive_reviews = positive_reviews.find_all('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), "lxml")
negative_reviews = negative_reviews.find_all('review_text')

# Random positive reviews and get same quantity from negative reviews - BALANCED CLASSES
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

def my_tokenizer(s):
    s = s.lower()  # downcase
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2]  # remove short words
    # todo: misspellings
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # putting words in base form
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    return tokens

# Setting base for matrix X with index dictionary and token arrays
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

# Iterate positive_reviews, insert tokens into array and map word index in object
for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

# Iterate negative_reviews, insert tokens into array and map word index in object
for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

# Convert each set of tokens into a data vector
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)  # setting bag of words frequency then increase last column for true label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()  # normalize it before setting true label
    x[-1] = label
    return x


# (N x D+1 matrix - keeping positive-negative together so shuffle more easily later
N = len(positive_tokenized) + len(negative_tokenized)

# Initialize data matrix with zero frequencies
data = np.zeros((N, len(word_index_map) + 1))
i = 0

# Get positive and negative words frequencies and add to data matrix
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i, :] = xy
    i += 1
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i, :] = xy
    i += 1

# Shuffle data and create train/test splits
np.random.shuffle(data)
# X as data matrix except true label column; Y as true label column
X = data[:, :-1]
Y = data[:, -1]

# Last 100 rows will be used as test
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]

# Classifying with LogisticRegression
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print "Classification rate: ", model.score(Xtest, Ytest)

# Getting the sentiment of each individual word through logistic regression weights
# Weight with absolute value less than 0.5 are considered neutral
threshold = 0.5
for word, index in word_index_map.iteritems():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print word, weight
