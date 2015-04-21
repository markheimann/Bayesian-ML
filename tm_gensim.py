#https://gist.github.com/aronwc/8248457
import numpy as np
import time
 
from gensim import matutils
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
   vec = CountVectorizer(min_df=10, stop_words='english')
#   vec = CountVectorizer(stop_words='english')
   tdm_all = vec.fit_transform(all_data.data)
   vocab = vec.get_feature_names()
   
   #Topic modeling 
   # Fit LDA.
   num_topics = [2,4,6,8,10,20]
   num_passes = 10

   for num in num_topics:
      # Time the fit_lda process
      start_time = time.time()
      lda = fit_lda(tdm_all, vocab, num, num_passes)
      print("fit_lda ran in %s seconds" % (time.time() - start_time))

      #print out LDA topics
      print_topics(lda,vocab,num_topics)


   #Classification using LDA topics as features
   #Possibly compare to document classification using the term-document counts as features (which is much higher dimensional)?

   #Split into training and test data

   
   #Fit classifier.

   
   #Test classifier


   #Compute and display error rate
