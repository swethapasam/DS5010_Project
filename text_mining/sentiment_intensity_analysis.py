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
nltk.download('vader_lexicon')
import en_core_web_sm
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
# %matplotlib inlineunprocessed_file
nlp = spacy.load("en_core_web_sm")
global raw_csv_file 
raw_csv_file = os.getcwd()+"/product_reviews.csv"

global processed_csv_file 
processed_csv_file = os.getcwd()+"/processed_product_reviews.csv"

global temp_csv_file 
temp_csv_file = os.getcwd()+"/temp_product_reviews.csv"


def gen_lemmas_sIntensity():
    #Load the csv and create DataFrame
    try:
        print(raw_csv_file)
        p_df = pd.read_csv(raw_csv_file)
        p_df = p_df.astype(str)

    except Exception as e: 
        sys.exit("********ERROR : Cannot find/open product_reviews.csv file ")

    #Use NLP to generate Lemmas (keywords/aspects) for each user review
    print("Generating Lemmas and Sentiment Intensity for reviews")
    p_df["Lemmas"] = [" ".join([token.lemma_ if token.lemma_ != "-PRON-" else token.lower() for sentence in nlp(speech).sents for token in sentence if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "X"} and token.is_stop == False]) for speech in p_df['User Reviews']]

    sia = SentimentIntensityAnalyzer()
    sentiment_score=[]
    for i in range(len(p_df)):
        ss = sia.polarity_scores(p_df.Lemmas[i])
        sentiment_score.append(ss)

    compound=[sentiment_score[i]['compound'] for i in range(len(sentiment_score))]
    p_df['SIntensity']=compound

    #Write Lemmas and sentiment intensity to csv file
    try:
        p_df.to_csv(processed_csv_file, index=False)
#         os.remove(temp_csv_file)
        print("Processed data file generated.")
    except Exception as e:
        sys.exit("********ERROR********")