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
# hello_world.check()

import sentiment_intensity_analysis as s
import vizualization as viz 
import basic_functions as b 
import tf_tfidf_bag_of_words.py as t


#Create variables for different files and products
global raw_csv_file 
raw_csv_file = os.getcwd()+"/product_reviews.csv"

global processed_csv_file 
processed_csv_file = os.getcwd()+"/processed_product_reviews.csv"

global temp_csv_file 
temp_csv_file = os.getcwd()+"/temp_product_reviews.csv"

global products 
#products = TV, Gaming Laptop, Head Phones, Roku TV stick, Ear Phones
products = ['128004808','27299919','300604667','43945914','44487595','441520359']

# This function generates lemma and Sentiment Intenstity
s.gen_lemmas_sIntensity()
# print the data
b.print_data()
# print products list
b.print_product_ids()
# add review
b.add_review()

# This function creates a word cloud for a certain product 
viz.create_wordcloud(128004808)
# Sentiment intensity plot of a product
viz.intensity_plots(128004808)


products_list_for_bag_of_words=['B000F83SZQ','B00BT0J8ZS','B00BTIDW4S','B000FA64QO','B000FBFMVG','B00JDYC5OI']
t.log_reg_tf_unigram('B006GWO5WK')
t.log_reg_tf_uni_bigram('B006GWO5WK')
t.log_reg_tf_uni_bi_trigram('B006GWO5WK')
# Linear support vector machine(LSVM)
t.lsvm_tf_unigram("B006GWO5WK")
t.lsvm_tf_uni_bigram("B006GWO5WK")
t.lsvm_tf_uni_bi_trigram("B006GWO5WK")
# Multinomial Naive Bayes algorithm
t.navie_bayes_tf_unigram('B006GWO5WK')
t.navie_bayes_tf_uni_bigram('B006GWO5WK')
t.navie_bayes_tf_uni_bi_trigram('B006GWO5WK')

t.log_tfidf_unigram('B006GWO5WK')
t.log_tfidf_uni_bigram('B006GWO5WK')
t.log_tfidf_uni_bi_trigram('B006GWO5WK')
t.lvsm_tf_idf_unigram('B006GWO5WK')
t.lvsm_tf_idf_uni_bigram('B006GWO5WK')
t.lvsm_tf_idf_uni_bi_trigram('B006GWO5WK')
t.naive_bayes_tfidf_unigram('B006GWO5WK')
t.naive_bayes_tfidf_uni_bigram('B006GWO5WK')
t.naive_bayes_tfidf_uni_bi_trigram('B006GWO5WK')










