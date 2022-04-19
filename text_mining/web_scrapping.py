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

#Create variables for different files and products
global raw_csv_file 
raw_csv_file = os.getcwd()+"\\product_reviews.csv"

global processed_csv_file 
processed_csv_file = os.getcwd()+"\\processed_product_reviews.csv"

global temp_csv_file 
temp_csv_file = os.getcwd()+"\\temp_product_reviews.csv"

global products 
#products = TV, Gaming Laptop, Google Home Device, Security Camera, Kitchen Aid Beater
products =['36907838','708236785','831078728','540689246','268234325','441520359']

#Look for the url and make a BeautifulSoup Object
def get_request(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")
        if response.status_code == 200:
            return soup
        else:
            sys.exit(response.status_code)
    
    except Exception as e:
        print("Cannot get request")
        sys.exit(e)

#Get user reviews
def get_reviews(soup):
    try:
        return(soup.p.text.replace("[This review was collected as part of a promotion.]",""))
    except Exception as e:
        #Return None if there is no user review
        return "None"

#Extract the sub tag required for reviews
def get_semi_soup(soup):
    try:
        return(soup.find_all("div", {"class":"Grid ReviewList-content"}))
    except Exception as e:
        sys.exit(e)

#Get individual user ratings
def get_user_rating(soup):
    return (soup.find("span", {"class":"visuallyhidden seo-avg-rating"}).string)
# Get dates of each reviews 

def get_review_dates(soup):
    date = soup.find('span',{'class':'review-footer-submissionTime'}).string
    return date  
# Get all customer nick names corresponding to reviews 
def get_user_names(soup):
    user=soup.find('span',{'class':'review-footer-userNickname'}).string
    return user
#Get total number of ratings. Use this to determine total number of pages to scrape.
def get_rating_count(soup):
    return(soup.find('div',{'class':'product-review-stars-container-ratings-row'}).string.replace("ratings",""))
def clean_up_dataset():
    try:
        c_df = pd.read_csv(raw_csv_file)
    
        c_df['User Reviews'].replace('None', np.nan, inplace=True)
        c_df=c_df.dropna()
        c_df.to_csv(temp_csv_file, index=False)
    
    except Exception as e: 
        sys.exit("********ERROR : Cannot find/open product_reviews.csv file ")
def scrape():
    #Here we do pagination --> go to every page to get reviews and other details
    prod_dict = {}
    #Create a csv file. 
    try:
        csv_file = open(raw_csv_file, 'w', encoding="UTF-8", newline="")
        writer = csv.writer(csv_file)
    except Exception as e:
         sys.exit(str(e))

    start = time.time()

    print("********GET REVIEWS START********")
    #Create the column names in csv file
    writer.writerow(['Product id','User','Date','User Rating','User Reviews'])
    for p in products:
        print ("Getting reviews from Walmart for product id : " + str(p))
        #Get total number of ratings to calculate the number of pages to scrape for information
        url_page = "https://www.walmart.com/reviews/product/"+str(p)
        soup = get_request(url_page)

        #Number of pages
        page_count = math.ceil(int(get_rating_count(soup))/20)
        page = 1

        #Iterate the pages to get info
        while page <= page_count:

            #Create the page url
            url_page = "https://www.walmart.com/reviews/product/"+str(p)+"?page="+str(page)
            soup = get_request(url_page)
            #Get the outermost tag for reviews 
            review_soup = get_semi_soup(soup)

            #Iterate over the outermost tage to get information for the review 

            for reviews in review_soup:

                #User name
                user = get_user_names(reviews)

                #User rating
                user_rating = get_user_rating(reviews)

                #Date of review
                review_date = get_review_dates(reviews)

                #Review 
                review = get_reviews(reviews)

                prod_dict['Product id'] = str(p)
                prod_dict['User'] = user
                prod_dict['Date'] = review_date
                prod_dict['User Rating'] = user_rating
                prod_dict['User Reviews'] = review

                #Write to csv file
                writer.writerow(prod_dict.values())
            page = page+1
    print("********GET REVIEWS DONE********")    
    csv_file.close() 
    print('Time taken to scrape reviews and write to CSV file : {} mins'.format(round((time.time() - start) / 60, 2)))