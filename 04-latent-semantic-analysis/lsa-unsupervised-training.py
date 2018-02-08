import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()
titles = [line.rstrip() for line in open('all_book_titles.txt')]

stopwords = set(w.rstrip() for w in open('stopwords.txt'))
# Add more stopwords specific to this problem
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package',
    'plus', 'etext', 'brief', 'vol',
    'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth'
})

def my_tokenizer(s):
    s = s.lower()  # downcase each string
    tokens = nltk.tokenize.word_tokenize(s)  # split title string into words
    tokens = [t for t in tokens if len(t) > 2]  # filter to remove short words
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]  # remove any digits (3rd edition)
    return tokens


# Creating word index mapping and other arrays
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []

# Iterate titles, appending results to arrays and main dictionary
for title in titles:
    try:
        title = title.encode('ascii', 'ignore')  # throw exception for bad characters
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        # Check and insert token in dictionary
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print e
        print title

# Turning tokens into feature vector
# PCA/SVD are unsupervised algorithms - just structure the data, without label
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x


# Creating DxN data matrix (N as rows-documents and D as columns-terms)
N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))  # Terms will go along rows, documents along columns
i = 0
for tokens in all_tokens:
    X[:, i] = tokens_to_vector(tokens)
    i += 1

# Creating scatterplot of the data with SVD, annotating each feature within sectors
svd = TruncatedSVD()
Z = svd.fit_transform(X)
plt.scatter(Z[:, 0], Z[:, 1])
for i in xrange(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
plt.show()