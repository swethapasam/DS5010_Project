import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import csv
import math
import os
import time
import seaborn as sb
import nltk
# nltk.download('vader_lexicon')
import en_core_web_sm
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
# %matplotlib inlineunprocessed_file
nlp = spacy.load("en_core_web_sm")
# import hello_world 
import sklearn

def train_test_data(product):
    raw_file = os.getcwd()+"/kindle_reviews.csv"

    p_new_df = pd.read_csv(raw_file)
    print(p_new_df.head())

    p_new_df=p_new_df[p_new_df['asin']==str(product)]
    p_new_df=p_new_df[['asin','reviewText','overall']]
    train = p_new_df[:int(len(p_new_df)*0.9)]
    test = p_new_df[int(len(p_new_df)*0.9):]
#     When the Bag of Words algorithm considers only single unique words in the vocabulary, the feature set is said to be UniGram. Let’s define train Logistic Regression classifier on unigram features:-


    return train, test

def train_test_lables(train,test):
     
    train_labels = [1 if overall>=3 else 0 for overall in train['overall']]
    test_labels = [1 if overall>=3 else 0 for overall in test['overall']]
    print (len(train_labels), len(test_labels))
    return train_labels,test_labels
    
#Logistic Regression classifier with unigram bag of words features    
def log_reg_tf_unigram(product):
    
    train,test=train_test_data(product)
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,1))

    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))  ## Even astype(str) would work

    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print(tfidf_features_train.shape, tfidf_features_test.shape)
    train_labels,test_labels=train_test_lables(train,test)

    #unigram
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(tfidf_features_train, train_labels)
    print (clf)
    predictions = clf.predict(tfidf_features_train)


    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

    #Unigram plus bigram

def log_reg_tf_uni_bigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))


def log_reg_tf_uni_bi_trigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))
    
    
#      Linear Support Vector Machine (LSVM) for sentiment analysis

def lsvm_tf_unigram(product):
    # lsvm
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,1))
    #by increaing the ngram_range we can create bigram and trigrams

    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = sklearn.svm.LinearSVC()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

def lsvm_tf_uni_bigram(product):
    # lsvm
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    #by increaing the ngram_range we can create bigram and trigrams

    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = sklearn.svm.LinearSVC()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

def lsvm_tf_uni_bi_trigram(product):
    # lsvm
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,3))
    #by increaing the ngram_range we can create bigram and trigrams

    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = sklearn.svm.LinearSVC()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))
    
    
    
from sklearn.naive_bayes import MultinomialNB

# Naive Bayes for sentiment analysis unigram
def navie_bayes_tf_unigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    from sklearn.naive_bayes import MultinomialNB
 
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,1))
    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = MultinomialNB()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

# Naive Bayes for sentiment analysis bigram
    
def navie_bayes_tf_uni_bigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    from sklearn.naive_bayes import MultinomialNB
 
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,2))
    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = MultinomialNB()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))
    

def navie_bayes_tf_uni_bi_trigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    from sklearn.naive_bayes import MultinomialNB
 
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False,ngram_range=(1,3))
    tf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tf_features_train.shape, tf_features_test.shape)

    clf = MultinomialNB()
    clf.fit(tf_features_train, train_labels)

    predictions = clf.predict(tf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))
    
# Time to feed these tf-idf features into the classification algorithms.
def log_tfidf_unigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,1))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

def log_tfidf_uni_bigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,2))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

    
def log_tfidf_uni_bi_trigram(product):
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,3))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

def lvsm_tf_idf_unigram(product):
    #Create features
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
 
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,1))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.svm.LinearSVC()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

def lvsm_tf_idf_uni_bigram(product):
    #Create features
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
 
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,2))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.svm.LinearSVC()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

def lvsm_tf_idf_uni_bi_trigram(product):
    #Create features
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
 
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,3))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.svm.LinearSVC()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))
# Multinomial Naive Bayes (MNB)

def naive_bayes_tfidf_unigram(product):
    #Create features
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
 
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,1))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))

def naive_bayes_tfidf_uni_bigram(product):
    #Create features
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
 
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,2))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))
def naive_bayes_tfidf_uni_bi_trigram(product):
    #Create features
    train,test=train_test_data(product)
    train_labels,test_labels=train_test_lables(train,test)
 
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True,ngram_range=(1,3))
    tfidf_features_train = vectorizer.fit_transform(train['reviewText'].values.astype('U'))
    tfidf_features_test = vectorizer.transform(test['reviewText'].values.astype('U'))
    print (tfidf_features_train.shape, tfidf_features_test.shape)

    #train model
    clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(tfidf_features_train, train_labels)

    #evaluation
    predictions = clf.predict(tfidf_features_test)
    print(sklearn.metrics.classification_report(test_labels, predictions, target_names=['Negative', 'Positive']))
    print(sklearn.metrics.confusion_matrix(test_labels, predictions, labels=[0, 1]))
