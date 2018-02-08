import nltk
import random
from bs4 import BeautifulSoup

# Extracting positive reviews text with XML parser
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), "lxml")
positive_reviews = positive_reviews.find_all('review_text')

# Extracting trigrams and adding to dictionary
# (w1, w3) is the key, [ w2 ] are the values
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in xrange(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])  # the key will be the first/last words
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# Turning each array of middle words into a probability vector
trigram_probabilities = {}
for k, words in trigrams.iteritems():
    # Create a dictionary of word frequency
    if len(set(words)) > 1:  # Different possibilities for a middle word case only
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1  # Increment word appearance
            n += 1  # Increment total of words
        for w, c in d.iteritems():
            d[w] = float(c) / n  # Frequency calculus for each middle word
        trigram_probabilities[k] = d  # Dictionary with frequencies for each middle word


# Randomly sample the dictionary
# Choose a random sample from dict where values are the probabilites
def random_sample(d):
    r = random.random()
    cumulative = 0
    for w, p in d.iteritems():
        cumulative += p
        if r < cumulative:
            return w


# Testing the spinner (needs more context)
def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print "Original: ", s
    tokens = nltk.tokenize.word_tokenize(s)
    for i in xrange(len(tokens) - 2):
        if random.random() < 0.2:  # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print "Spun:"
    print " ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!")


if __name__ == '__main__':
    test_spinner()