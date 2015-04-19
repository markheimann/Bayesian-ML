from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import *
import numpy as np
import lda
#import lda.datasets

#NMF for topic modeling: http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf.html
#(cf. NMF: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 10

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

t0 = time()
print("Loading dataset and extracting TF-IDF features...")
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

#vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
#                             stop_words='english')
vectorizer = CountVectorizer(stop_words='english')

counts_vectorizer = CountVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(dataset.data[:n_samples])
print("done in %0.3fs." % (time() - t0))

counts_vectorizer = CountVectorizer(stop_words='english')
counts = counts_vectorizer.fit_transform(dataset.data[:n_samples])
# Fit the NMF model
print("Fitting the NMF model with n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


#LDA: https://pypi.python.org/pypi/lda


#document-term matrix 
#(rows: documents, columns: terms, entries: #occurrences in document of a term)
#X = lda.datasets.load_reuters()
#vocab = lda.datasets.load_reuters_vocab()
#if we ever want the titles here they are
#titles = lda.datasets.load_reuters_titles() 

#test LDA
model = lda.LDA(n_topics=n_topics, n_iter=200, random_state=1)
#model.fit(dataset)
model.fit(counts)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
#for i, topic_dist in enumerate(topic_word):
#   topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
#   print('Topic {}: {}'.format(i, ' '.join(topic_words)))
feature_names = counts_vectorizer.get_feature_names()
for topic_idx, topic in enumerate(model.components_):
   print("Topic #%d:" % topic_idx)
   # print topic.argsort()
   print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    
