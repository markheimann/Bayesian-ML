#https://gist.github.com/aronwc/8248457
import numpy as np
import time

from gensim import matutils, corpora
from gensim.models.ldamodel import LdaModel
from sklearn import linear_model
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


'''Machine learning methods''' 

def print_features(clf, vocab, n=10):
   """ Print sorted list of non-zero features/weights. """
   coef = clf.coef_[0]
   print 'positive features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0]))
   print 'negative features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0]))
 
 
def fit_classifier(X, y, C=0.1):
   # Smaller C means fewer features selected.
   #clf = linear_model.LogisticRegression(penalty='l1', C=C)
   clf = svm.SVC()
   clf.fit(X, y)
   return clf

#Use classifier to make predictions on data
#Can return those predictions and compare to true labels to calculate error''' 
def make_predictions(clf,data):
   return clf.predict(data)

#Compute and return classification error by comparing predictions to labels 
def compute_error(predictions,labels):
   comp_preds = predictions - labels
   errorRate = np.count_nonzero(comp_preds)*1.0/np.size(comp_preds)
   return errorRate 


'''LDA methods'''

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
 

def getTopicDistributions(lda, documents):
   topic_dists = [lda[document] for document in documents]
   return topic_dists

def getTopicDistributionFeatures(topic_distributions, num_topics):
   topicDist_features =  [ [(topic_dist[i][1] if i < len(topic_dist) and topic_dist[i][0] == i else 0) for i in range(num_topics)] for topic_dist in topic_distributions]
   return topicDist_features

'''Text processing methods'''

#Read in and stop words from a file and save them as a set
def getStopWords():
   stoplist = list()
   for word in open('stopwords.txt', 'r'):
      stoplist.append(word.strip()) 
   return stoplist

def trainTest_split(full_data, train_proportion):
   data = full_data.data
   labels = full_data.target
   num_docs = len(data)
   num_train = int(train_proportion * num_docs)
   train_data = data[0:num_train]
   test_data = data[num_train:num_docs]
   train_labels = labels[0:num_train]
   test_labels = labels[num_train:num_docs]
   return (train_data, test_data, train_labels, test_labels)

def process_documents(documents):
   #standard list of stopwords from http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
   #read in from file called stopwords.txt
   #each stopword on its own line
   stoplist = getStopWords()
   texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

   frequency = defaultdict(int)
   for text in texts:
      for token in text:
         frequency[token] += 1
   
   processed_texts = [[token for token in text if frequency[token] > 1] for text in texts]
   return processed_texts

def getDictionary(texts):
   return corpora.Dictionary(texts)

def getCorpus(texts, dictionary):
   corpus = [dictionary.doc2bow(text) for text in texts]
   return corpus



'''Run program'''
 
if (__name__ == '__main__'):
   # Load data.
   #allows you to specify which categories you want
   #shuffles the data so it's in random order
   rand = np.random.mtrand.RandomState(8675309)
   cats = ['rec.sport.baseball', 'sci.crypt']
   all_data = fetch_20newsgroups(categories=cats,shuffle=True,random_state=rand, remove=('headers', 'footers', 'quotes'))
   train_proportion = 0.8
   train_data, test_data, train_labels, test_labels = trainTest_split(all_data, train_proportion)

   #Create bag of words representation of corpus
   #TODO: load and save dictionary, LDA model instead of recomputing each time (use gensim save and load methods)
   #https://radimrehurek.com/gensim/tut1.html   
   
   train_texts = process_documents(train_data)
   test_texts = process_documents(test_data) 
   
   #Dictionary and corpus used to fit LDA model must be based only off training data
   dictionary = getDictionary(train_texts)
   corpus = getCorpus(train_texts, dictionary)
   test_bow_docs = getCorpus(test_texts, dictionary)

   #Fit and time LDA.
   num_topics = 5
   num_passes = 5
   
   start_time = time.time()
   lda = LdaModel(corpus, id2word=dictionary,num_topics=num_topics,passes=num_passes)
   print("Performed LDA in %f seconds" % (time.time() - start_time))

   #print out LDA topics
   print_topics(lda,dictionary.keys(),num_topics)

   #Get distribution of topics for training documents
   train_topic_dists = getTopicDistributions(lda, corpus) 
   train_topicDist_features = getTopicDistributionFeatures(train_topic_dists, num_topics)
  
   #Get distribution of topics for training documents
   test_topic_dists = getTopicDistributions(lda, test_bow_docs) 
   test_topicDist_features = getTopicDistributionFeatures(test_topic_dists, num_topics)
   
   #Fit classifier.
   clf = fit_classifier(train_topicDist_features,train_labels)   
   
   #Test classifier
   train_preds = make_predictions(clf, train_topicDist_features)
   test_preds = make_predictions(clf,test_topicDist_features)
   
   #Compute and display error rate
   train_error = compute_error(train_preds, train_labels)
   print "Training error using LDA topic distributions as features: ", train_error

   test_error = compute_error(test_preds, test_labels)
   print "Test error using LDA topic distributions as features: ", test_error
