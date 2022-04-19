# Word count dictionary. This function is only for verification and not used in the code. 
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


global processed_csv_file 
processed_csv_file = os.getcwd()+"/processed_product_reviews.csv"

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts
#Create a wordcloud visualization of the keywords/aspects found in the reviews.
def create_wordcloud(product):
    words = ""
    p_new_df = pd.read_csv(processed_csv_file)
    filt = p_new_df['productid'] == int(product)
    p_df = p_new_df[filt]
    p_df=p_df.reset_index(drop=True)


    #Use Lemmas field from the data set
    for l in p_df['Lemmas']:
        words = str(l) + "" + str(words)
        
    # Create and generate a word cloud image
    wordcloud = WordCloud().generate(words)

    # Display the generated image:
    plt.figure(figsize=(30,30))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def intensity_plots(product):
    p_new_df = pd.read_csv(processed_csv_file)
    filt1 = p_new_df['productid'] == int(product)
#     print(filt1)

    filt2 = p_new_df['SIntensity'] < -0.05
    filt3 = p_new_df['SIntensity'] > 0.05
    filt4 = (p_new_df['SIntensity'] > -0.05) & (p_new_df['SIntensity'] < 0.05)
#     print(
    neg_df = p_new_df[filt1 & filt2]
    pos_df = p_new_df[filt1 & filt3]
    neu_df = p_new_df[filt1 & filt4]
    neg_cnt = neg_df['SIntensity'].count()
    pos_cnt = pos_df['SIntensity'].count()
    neu_cnt = neu_df['SIntensity'].count()
#     print("Printingneg_cnt)

    total=neg_cnt+pos_cnt+neu_cnt
    neg_percentage=neg_cnt/total*100
    pos_percentage=pos_cnt/total*100
    nue_percentage=neu_cnt/total*100
    count = [pos_percentage,neg_percentage,nue_percentage]
#     print(senti)

    senti = ["+ve: "+str(round(pos_percentage))+"%","-ve: "+str(round(neg_percentage))+"%","Neutral: "+str(round(nue_percentage))+"%"]
    print(senti)


    sb.barplot(x=senti,y=count)
    plt.title("Percentage of different Sentiment of reviews for the product")
    plt.xlabel('\nReview Sentiment')
    plt.ylabel('Percentage(%)')
    plt.show()