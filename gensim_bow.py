#https://gist.github.com/aronwc/8248457
import numpy as np
import time
 
from gensim import matutils, corpora
from gensim.models.ldamodel import LdaModel
from sklearn import linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
 
def print_features(clf, vocab, n=10):
   """ Print sorted list of non-zero features/weights. """
   coef = clf.coef_[0]
   print 'positive features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0]))
   print 'negative features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0]))
 
 
def fit_classifier(X, y, C=0.1):
   """ Fit L1 Logistic Regression classifier. """
   # Smaller C means fewer features selected.
   clf = linear_model.LogisticRegression(penalty='l1', C=C)
   clf.fit(X, y)
   return clf

#Use classifier to make predictions on data
#Can return those predictions and compare to true labels to calculate error''' 
def make_predictions(clf,test_data):
   return clf.predict(test_data)

#Compute and return classification error by comparing predictions to labels 
def compute_error(predictions,labels):
   comp_preds = predictions - labels
   errorRate = np.count_nonzero(comp_preds)*1.0/np.size(comp_preds)
   return errorRate 

 
def fit_lda(X, vocab, num_topics=5, passes=20):
   """ Fit LDA from a scipy CSR matrix (X). """
   print 'fitting lda...'
   
   return LdaModel(matutils.Sparse2Corpus(X), num_topics=num_topics,
   #return LdaModel(X, num_topics=num_topics,    
                passes=passes,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))
 
 
def print_topics(lda, vocab, n=10):
   """ Print the top words for each topic. """
   topics = lda.show_topics(num_topics=n, formatted=False)
   for ti, topic in enumerate(topics):
      print 'topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic))
 
 
if (__name__ == '__main__'):
   # Load data.
   #allows you to specify which categories you want
   #shuffles the data so it's in random order
   rand = np.random.mtrand.RandomState(8675309)
   cats = ['rec.sport.baseball', 'sci.crypt']
   all_data = fetch_20newsgroups(categories=cats,shuffle=True,random_state=rand, remove=('headers', 'footers', 'quotes'))
#   all_data = fetch_20newsgroups(shuffle=True,random_state=rand, remove=('headers', 'footers', 'quotes'))
#   vec = CountVectorizer(min_df = 10, stop_words='english')

   #Create bag of words representation of corpus
   #TODO: load and save dictionary, LDA model instead of recomputing each time (use gensim save and load methods)
   #https://radimrehurek.com/gensim/tut1.html   
   stoplist = set('for a of the and to in'.split())
   texts = [[word for word in document.lower().split() if word not in stoplist] for document in all_data.data]

   from collections import defaultdict
   frequency = defaultdict(int)
   for text in texts:
      for token in text:
         frequency[token] += 1
   
   texts = [[token for token in text if frequency[token] > 1] for text in texts]
   dictionary = corpora.Dictionary(texts)
   corpus = [dictionary.doc2bow(text) for text in texts]
   
   #Create term document matrix: each document is a row, columns correspond to terms
#   tdm_all = vec.fit_transform(all_data.data)
#   vocab = vec.get_feature_names()
   
   #Topic modeling 
   #Fit LDA.
   num_topics = 5
   num_passes = 1
   
   # Time LDA
   start_time = time.time()
#   lda_orig = fit_lda(tdm_all, vocab, num_topics, num_passes)
   lda = LdaModel(corpus, id2word=dictionary,num_topics=num_topics,passes=num_passes)
   print("Performed LDA in %f seconds" % (time.time() - start_time))

   #print out LDA topics
#   print_topics(lda_orig,vocab,num_topics)
   print_topics(lda,dictionary.keys(),num_topics)
   
   num_docs = len(all_data.data)

   print "distribution of topics for documents: "
   lda_topic_dists = [lda[corpus[i]] for i in range(0,num_docs)]
   lda_dists_matrix = [ [term[1] for term in topic_dist ] for topic_dist in lda_topic_dists]
   print len(lda_dists_matrix)
   print len(lda_dists_matrix[0])  
   #Classification using LDA topics as features
   #Possibly compare to document classification using the term-document counts as features (which is much higher dimensional)?

   #Split into training and test data
   #In this case we just split document-term matrix into training and test components
   training_proportion = 0.8 #can set elsewhere
   #Since documents are shuffled randomly, just take the first training_proportion*num_docs as training, rest as test
   num_train = int(training_proportion*num_docs)
   print "Number of training examples: ", num_train
   train_docs = corpus[0:num_train]
   train_lda_features = lda_dists_matrix[0:num_train] #[lda[corpus[i]] for i in range(0,num_train)] #lda[train_docs]
   print type(train_lda_features)   
   test_docs= corpus[num_train:num_docs]
   test_lda_features = lda_dists_matrix[num_train:num_docs]#[lda[corpus[i]] for i in range(num_train,num_docs)] #lda[test_docs]
  
   print len(train_lda_features)
   print len(all_data.target[0:num_train]) 
   print train_lda_features[0]
   print all_data.target[0:num_train][0]

   #Fit classifier.
   clf = fit_classifier(train_lda_features,all_data.target[0:num_train],C=0.1)   
   
   #Test classifier
   test_preds = make_predictions(clf,test_lda_features)

   #Compute and display error rat
   test_error = compute_error(test_preds, all_data.target[num_train:num_docs])
   print "Test error using LDA topic distributions as features: ", test_error
