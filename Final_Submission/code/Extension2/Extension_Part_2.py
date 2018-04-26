#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:52:35 2018

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
import csv
from sklearn.svm import SVC

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
    
def compute_cosine_similarity(vector1, vector2):
    if np.linalg.norm(vector1) * np.linalg.norm(vector2) == 0:
        return 0
    else:
        return float((np.dot(vector1, vector2))/(np.linalg.norm(vector1) * np.linalg.norm(vector2)))

X_data_train = []
y_data_train = []
with open("Extension_2_Features.csv","r") as f:
    data= csv.reader(f)
    for row in data:
        X_data_train.append(row[:6] + row[7:])
        y_data_train.append(row[6])
        
X_data_train = X_data_train[1:]
y_data_train = y_data_train[1:]


for i in range(len(X_data_train)):
    for j in range(len(X_data_train[i])):
        if X_data_train[i][j].strip() != "":
            X_data_train[i][j] = float(X_data_train[i][j].strip())
        else:
            X_data_train[i][j] = 0.0
            
y_data_train = [int(a) for a in y_data_train]

X_data_train = np.array(X_data_train[:10000])
y_data_train = np.array(y_data_train[:10000])

clf = SVC(kernel = "rbf",probability = True)
clf.fit(X_data_train,y_data_train)


# X_data_test = []
# with open("supervised_X_data_test.txt","r") as f:
#     data = f.read().split("\n")
#     for line in data:
#         X_data_test.append(line)

entity_scores = []
entity_sentence = []
article_num = []
with open("../entity_scores_test.txt","r") as f:
    data = f.read().split("\n")
    for line in data:
        article_num.append(int(line.split("@@@")[0].strip()))
        entity_scores.append(float(line.split("@@@")[1].strip()))
        entity_sentence.append(line.split("@@@")[2].strip())

article_set = set(article_num)

nlp = spacy.load('en') 

summary_pred = []

for article in article_set:
    print(article)
    X_data_sentences_original = []
    X_data_sentences = []
    X_data_entity_score = []
    for i,j,k in zip(article_num,entity_sentence,entity_scores):
        if i == article:
            X_data_sentences.append(j)
            X_data_sentences_original.append(j)
            X_data_entity_score.append(k)
            
    tfidf_vect_uni = TfidfVectorizer(stop_words="english",lowercase = True)
    tfidf_vect_bi = TfidfVectorizer(stop_words="english",ngram_range=(2,2),lowercase = True)
    vect_uni = CountVectorizer(stop_words="english",lowercase = True)
    vect_bi = CountVectorizer(stop_words="english",ngram_range=(2,2),lowercase = True)
    vect_uni_1 = CountVectorizer(stop_words="english",lowercase = True,max_features = 30)
    vect_bi_1 = CountVectorizer(stop_words="english",ngram_range=(2,2),lowercase = True,max_features=30)
    tfidf_matrix_uni = tfidf_vect_uni.fit_transform(X_data_sentences)
    tfidf_matrix_bi = tfidf_vect_bi.fit_transform(X_data_sentences)
    tf_matrix_uni = vect_uni.fit_transform(X_data_sentences)
    tf_matrix_bi = vect_bi.fit_transform(X_data_sentences)
    tf_matrix_uni_1 = vect_uni_1.fit_transform(X_data_sentences)
    tf_matrix_bi_1 = vect_bi_1.fit_transform(X_data_sentences)
    
    sentences = X_data_sentences_original
    
    first_sentence_vect = np.asarray(tf_matrix_uni[0].todense())
    text_features = []
    for i in range(len(sentences)):
        test_features = []
        position = 1/(i+1.0)
        if i == 0:
            doc_first = 1
        else:
            doc_first = 0
        length = len(sentences[i].split())
        quote = len(" ".join(re.findall("[\"\'][a-zA-Z0-9 ]*[\"\']",sentences[i])).strip().split())
        Centroid_uni = np.sum(tfidf_matrix_uni[i].todense())
        Centroid_bi = np.sum(tfidf_matrix_bi[i].todense())
        SigTerm_Uni = len(np.where(tf_matrix_uni[i].todense() > 0)[0])
        SigTerm_Bi = len(np.where(tf_matrix_bi[i].todense() > 0)[0])
        FreqWord_Uni = np.sum(tf_matrix_uni_1[i].todense())
        FreqWord_Bi = np.sum(tf_matrix_bi_1[i].todense())
        text = nlp(sentences[i].decode("utf8"))
        event_features_length = len('{0}'.format(text.ents).split())
        current_sentence_vect = np.transpose(np.asarray(tf_matrix_uni[i].todense()))
        FirstRel_Doc = compute_cosine_similarity(first_sentence_vect,current_sentence_vect)
        test_features.append(Centroid_bi)
        test_features.append(Centroid_uni)
        test_features.append(event_features_length)
        test_features.append(FirstRel_Doc)
        test_features.append(FreqWord_Bi)
        test_features.append(FreqWord_Uni)
        test_features.append(SigTerm_Bi)
        test_features.append(SigTerm_Uni)
        test_features.append(doc_first)
        test_features.append(X_data_entity_score[i])
        test_features.append(length)
        test_features.append(position)
        test_features.append(quote)
        text_features.append(test_features)
    text_features = np.array(text_features)
    prob = clf.predict_proba(text_features)
    prob_one = [a[1] for a in prob]
    prob_sentence_tuple = []
    for j in range(len(prob_one)):
        prob_sentence_tuple.append((sentences[j],prob_one[j]))
    
    ordered_prob_sentence = sorted(prob_sentence_tuple, key=lambda x: x[1])
    summary = ""
    for j in range(3):
        summary = summary + ordered_prob_sentence[j][0] + " "
    summary_pred.append(summary)


with open("y_pred_summary.txt","w") as f:
    for line in summary_pred:
        f.write(line)
        f.write("\n")


        
end = datetime.datetime.now()
duration = end - start
print("Duration - " + str(duration))