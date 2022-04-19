# DS5010_Project

**** Objective of the Package **-
- To understand the sentiment of the customers for the products on walmart based on the reviews submitted by the user on the website using natural language processing techniques **i.e by using the classification techniques available on NLTK package(SentimentIntensityAnalyzer) **
- To understand the sentiment of the customer for the products on amazon books based on the reviews submitted by the user **using differnt bag of words techniques like Logistic Regression, Linear SVM and MultiNomial Naive Bayes by using both term frequency and term frequency-inverse document frequency**
 
**Programming language ***
Python

**Python Packages used **
- Basic python Packages - Pandas, csv , numpy , math , os , time ,sys 
- Visualization - Seaborn, wordcloud, matplotlib
- Sentiment Analysis - nltk, sklearn
- Web Scrapping - BeautifulSoap , requests

**Classes in the Pacakge and a quick description **

**Test.py **
- This is the important module of the package. 
- This is test/main code in which all the modules are imported and are tested.  

**basic_functions.py**
- In this module the functions like printing the webscrapped data , printing the unique number of products list available with us are outputted and adding a review to the exiting reviews available based on user inputs

**sentiment_intensity_analysis.py**
- This module creates the processed file by taking in input the raw file(web scrapping input) by generating lemma and Sentiment Intenstity

**tf_tfidf.py**
- This code divides the universe universe to train and test and generates a matrix of accuracy for  using differnt bag of words techniques like Logistic Regression, Linear SVM and MultiNomial Naive Bayes by using both term frequency and term frequency-inverse document frequency


<img width="959" alt="image" src="https://user-images.githubusercontent.com/17700749/164024249-af8e15df-3b10-4516-915b-d0dec4395dc7.png">
<img width="959" alt="image" src="https://user-images.githubusercontent.com/17700749/164024395-50acc071-7f20-4c6b-81da-916de52fd4c5.png">
<img width="983" alt="image" src="https://user-images.githubusercontent.com/17700749/164024705-a21810e3-78c2-493f-8b2b-c661ac7fb456.png">
<img width="914" alt="image" src="https://user-images.githubusercontent.com/17700749/164024782-1b8b8cce-5f0c-46f3-817a-ced29dea443b.png">


**** webscrapping.py** 
- This module contains various functions which helps in scrapping the walmart website and writing them as a csv , which will be used in further process

**** visualization.py**
- This module has functions which helps in creating the word cloud and the overall sentiment of the user based onclassification techniques available on NLTK package(SentimentIntensityAnalyzer)
- Takes in productid as the input
<img width="487" alt="image" src="https://user-images.githubusercontent.com/17700749/164025201-cb7baac4-f4ec-4233-8596-1191531defc8.png">


- This module has functions which helps in creating the word cloud and the overall sentiment of the user based onclassification techniques available on NLTK package(SentimentIntensityAnalyzer)

![Figure_1](https://user-images.githubusercontent.com/17700749/164023885-0249bdf6-3862-4437-b3ef-fcc3040680ec.png)




**Note: We have picked up datasets from the kaggle 
**
