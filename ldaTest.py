import numpy as np
import lda
import lda.datasets

#Python references
#LDA: https://pypi.python.org/pypi/lda
#NNMF: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

#for NNMF
from sklearn.decomposition import *

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
print X.shape

#test LDA
model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 9
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))
	
#test NNMF
model = ProjectedGradientNMF(n_components=2, sparseness='components', init='random', random_state=0)
model.fit(X)
print "Reconstruction error of NNMF: ",
print model.reconstruction_err_