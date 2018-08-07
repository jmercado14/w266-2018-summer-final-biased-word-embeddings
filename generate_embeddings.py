from w266_common import utils
import unittest
import csv
import pandas as pd
from nltk.corpus import stopwords 
from nltk import tokenize
import nltk
from gensim.models import Word2Vec
import gensim.utils
import re
import datetime

if __name__ == "__main__":
    
    print ("Loading news article data")
    print(datetime.datetime.now())
    articles = pd.read_csv("./data/RANE/2018_articles.csv")
    
    # preprocess article text (this will take hours)
    preprocessed = []
    print ("Beginning preprocessing")
    print(datetime.datetime.now())
    print(articles.shape)
    for i in range(articles.shape[0]):
    # for i in range(5):
        if i%100000 == 0:
            print (i, "articles preprocessed")
            print(datetime.datetime.now())
        text = gensim.utils.simple_preprocess(articles.iloc[i]['article'])
        preprocessed.append(text)
    
    # create model (this will take hours)
    print ("Beginning model generation")
    print(datetime.datetime.now())
    model = Word2Vec(preprocessed, iter=2, size=300, workers=4) # num dimensions = 300
    
    # write model to file
    print ("Writing embedding matrix to file")
    print(datetime.datetime.now())
    model.wv.save_word2vec_format('./embeddings/RANE/RANE_300d.txt', binary=False)
    
    # end
    print ("Finished!")
    print(datetime.datetime.now())