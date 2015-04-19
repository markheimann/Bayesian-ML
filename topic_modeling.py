from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
import numpy as np
import lda

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 10

#dataset: 20 newsgroups
print("Loading dataset and extracting document term features...")
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

#get features from dataset
#vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
#                             stop_words='english')
vectorizer = CountVectorizer(stop_words='english')
td_matrix = vectorizer.fit_transform(dataset.data[:n_samples])
feature_names = vectorizer.get_feature_names()

#Fit NMF
#for reference: http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf.html
print("Fitting the NMF model with %d samples and %d features..."
      % (n_samples, n_features))
nmf = NMF(n_components=n_topics, random_state=1).fit(td_matrix)

#print out topics
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

#Fit LDA
#For reference: https://pypi.python.org/pypi/lda
print ("Fitting LDA for %d topics..." % n_topics)
model = lda.LDA(n_topics=n_topics, n_iter=200, random_state=1)
model.fit_transform(td_matrix)  

#print out topics
topic_word = model.components_ # model.topic_word_ also works
for topic_idx, topic in enumerate(model.components_):
   print("Topic #%d:" % topic_idx)
   print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
   print() 
