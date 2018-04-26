#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 13:36:53 2018

@author: aditya
"""
import nltk
from nltk.corpus import stopwords
import numpy as np
import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import pandas as pd

start = datetime.datetime.now()

def rouge_metrics(system_list,reference_list):
    reference_word_count = len(reference_list)
    system_word_count = len(system_list)
    if (system_word_count == 0) or (reference_word_count == 0):
        rouge_recall = 0
        rouge_precision = 0
    else:
        rouge_recall = len(intersection(system_list,reference_list))*1.0/reference_word_count
        rouge_precision = len(intersection(system_list,reference_list))*1.0/system_word_count
    return rouge_precision, rouge_recall



def intersection(system_lst, ref_lst):
    intersection_lst = [value for value in system_lst if value in ref_lst]
    return intersection_lst



def create_ngrams(text_list,n=2):
	iterations = len(text_list)-n
	ngrams = []
	gram = []
	for i in range(iterations+1):
		gram = text_list[i:n+i]
		ngrams.append(gram)
	return ngrams

def f_score(precision,recall):
    if precision + recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision + recall)
    
def compute_cosine_similarity(vector1, vector2):
    if np.linalg.norm(vector1) * np.linalg.norm(vector2) == 0:
        return 0
    else:
        return float((np.dot(vector1, vector2))/(np.linalg.norm(vector1) * np.linalg.norm(vector2)))
    

    
stop_words = set(stopwords.words('english'))

# SPLITTING THE DATA INTO 80% TRAIN, 20% TEST
# RUN THIS FIRST TO CREATE THE FILES
# =============================================================================
# X_data = []
# with open("X_data_train_5K.txt","r") as f:
#     data = f.read().split("\n")
#     for line in data:
#         X_data.append(line)
 
# y_data = []
# with open("y_data_train_5K.txt","r") as f:
#     data = f.read().split("\n")
#     for line in data:
#         y_data.append(line)   

# with open("supervised_X_data_train.txt","w") as f:
#     for line in X_data[:4000]:
#         f.write(line)
#         f.write("\n")

# with open("supervised_y_data_train.txt","w") as f:
#     for line in y_data[:4000]:
#         f.write(line)
#         f.write("\n")

# with open("supervised_X_data_test.txt","w") as f:
#     for line in X_data[4000:]:
#         f.write(line)
#         f.write("\n")

# with open("supervised_y_data_test.txt","w") as f:
#     for line in y_data[4000:]:
#         f.write(line)
#         f.write("\n")
# =============================================================================

# X_data = []
# with open("supervised_X_data_train.txt","r") as f:
#     data = f.read().split("\n")
#     for line in data:
#         X_data.append(line)
 
y_data = []  
with open("supervised_y_data_test.txt","r") as f:
    data = f.read().split("\n")
    for line in data:
        y_data.append(line) 


entity_sentence = []
article_num = []
with open("../entity_scores_test.txt","r") as f:
    data = f.read().split("\n")
    for line in data:
        article_num.append(int(line.split("@@@")[0].strip()))
        entity_sentence.append(line.split("@@@")[2].strip())

article_set = set(article_num)
        
# nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat', 'tokenizer'])
nlp = spacy.load('en')
features_labels = []

best_sentences_list = []

for article in article_set:
    print(article)
    X_data_sentences_original = []
    X_data_sentences = []
    
    for i,j in zip(article_num,entity_sentence):
        if i == article:
            X_data_sentences_original.append(j)
            X_data_sentences.append(j)
    


    #X_data_sentences = [a for a in X_data_sentences if len(set(a.split()) - stop_words)> 2]
    reference_2grams = create_ngrams(y_data[article-1].split(),2)
    system_2grams = [create_ngrams(a.split(),2) for a in X_data_sentences]

    precision_recall = [rouge_metrics(a,reference_2grams) for a in system_2grams]
    f_score_list = [f_score(a[0],a[1]) for a in precision_recall]
    best_sentences = X_data_sentences[np.argmax(f_score_list)]   
    #print("1",best_sentences)

    X_data_sentences = list(set(X_data_sentences) - set([best_sentences]))
    X_data_sentences_1 = [best_sentences + "\n" + a for a in X_data_sentences]
    system_2grams = [create_ngrams(a.split(),2) for a in X_data_sentences_1]
    precision_recall = [rouge_metrics(a,reference_2grams) for a in system_2grams]
    f_score_list = [f_score(a[0],a[1]) for a in precision_recall]
    best_sentences = X_data_sentences_1[np.argmax(f_score_list)]
    #print("2",best_sentences)
    
    X_data_sentences_1 = list(set(X_data_sentences_1) - set([best_sentences]))
    X_data_sentences = list(set(X_data_sentences) - set(best_sentences.split("\n")))
    X_data_sentences_1 = [best_sentences + "\n" + a for a in X_data_sentences]
    system_2grams = [create_ngrams(a.split(),2) for a in X_data_sentences_1]
    precision_recall = [rouge_metrics(a,reference_2grams) for a in system_2grams]
    f_score_list = [f_score(a[0],a[1]) for a in precision_recall]


    best_sentences = X_data_sentences_1[np.argmax(f_score_list)].split("\n")
    
    best_summary = " ".join(best_sentences)
    best_ngrams = create_ngrams(best_summary.split(),2)
    print("3",best_summary)
    original_summary = y_data[article-1]
    original_ngrams = create_ngrams(original_summary.split(),2)
    print("3.1", original_summary)
    print(original_ngrams)
    print(best_ngrams)
    rouges = rouge_metrics(original_ngrams,best_ngrams)
    print(rouges)
    fscore = f_score(1.0*rouges[0],1.0*rouges[1])
    print(fscore)

    best_sentences_list.append(" ".join(best_sentences))
    sentences = X_data_sentences_original
    


end = datetime.datetime.now()
duration = end - start
print("Duration - " + str(duration))