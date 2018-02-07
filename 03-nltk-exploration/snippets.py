import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Getting words role (parts-of-speech)
# see: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
print nltk.pos_tag("Machine learning is great".split())

# Steemming (more crude) x Lemmatization
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
print porter_stemmer.stem('wolves')
print wordnet_lemmatizer.lemmatize('wolves')

# Displaying a tree of named Entity Recognition (NER)
nltk.ne_chunk(nltk.pos_tag("Albert Einstein was born on March 14, 1879.".split())).draw()

# NNP (noun as a person or organization)
nltk.ne_chunk(nltk.pos_tag("Steve Jobs was the CEO of Apple Corp.".split())).draw()
