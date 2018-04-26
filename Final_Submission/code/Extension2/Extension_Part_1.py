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
with open("supervised_y_data_train.txt","r") as f:
    data = f.read().split("\n")
    for line in data:
        y_data.append(line) 

entity_scores = []
entity_sentence = []
article_num = []
with open("../entity_scores_train.txt","r") as f:
    data = f.read().split("\n")
    for line in data:
        article_num.append(int(line.split("@@@")[0].strip()))
        entity_scores.append(float(line.split("@@@")[1].strip()))
        entity_sentence.append(line.split("@@@")[2].strip())

article_set = set(article_num)
        
# nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat', 'tokenizer'])
nlp = spacy.load('en')
features_labels = []

for article in article_set:
    print(article)
    X_data_sentences_original = []
    X_data_sentences = []
    X_data_entity_score = []
    for i,j,k in zip(article_num,entity_sentence,entity_scores):
        if i == article:
            X_data_sentences_original.append(j)
            X_data_sentences.append(j)
            X_data_entity_score.append(k)

    tfidf_vect_uni = TfidfVectorizer(stop_words="english",lowercase = True)
    tfidf_vect_bi = TfidfVectorizer(stop_words="english",ngram_range=(2,2),lowercase = True)
    vect_uni = CountVectorizer(stop_words="english",lowercase = True)
    vect_bi = CountVectorizer(stop_words="english",ngram_range=(2,2),lowercase = True)
    vect_uni_1 = CountVectorizer(stop_words="english",lowercase = True,max_features = 30)
    vect_bi_1 = CountVectorizer(stop_words="english",ngram_range=(2,2),lowercase = True,max_features=30)
    tfidf_matrix_uni = tfidf_vect_uni.fit_transform(X_data_sentences) # Used for calculating Centroid_Uni
    tfidf_matrix_bi = tfidf_vect_bi.fit_transform(X_data_sentences) # Used for calculating Centroid_Bi
    tf_matrix_uni = vect_uni.fit_transform(X_data_sentences)# Used for calculating SigTerm_Uni
    tf_matrix_bi = vect_bi.fit_transform(X_data_sentences) #  Used for calculating SigTerm_Bi
    tf_matrix_uni_1 = vect_uni_1.fit_transform(X_data_sentences) 
    tf_matrix_bi_1 = vect_bi_1.fit_transform(X_data_sentences)
    

    #X_data_sentences = [a for a in X_data_sentences if len(set(a.split()) - stop_words)> 2]
    reference_2grams = create_ngrams(y_data[article-1].split(),2)
    system_2grams = [create_ngrams(a.split(),2) for a in X_data_sentences]
    precision_recall = [rouge_metrics(a,reference_2grams) for a in system_2grams]
    f_score_list = [f_score(a[0],a[1]) for a in precision_recall]
    best_sentences = X_data_sentences[np.argmax(f_score_list)]   

    X_data_sentences = list(set(X_data_sentences) - set([best_sentences]))
    X_data_sentences_1 = [best_sentences + "\n" + a for a in X_data_sentences]
    system_2grams = [create_ngrams(a.split(),2) for a in X_data_sentences_1]
    precision_recall = [rouge_metrics(a,reference_2grams) for a in system_2grams]
    f_score_list = [f_score(a[0],a[1]) for a in precision_recall]
    best_sentences = X_data_sentences_1[np.argmax(f_score_list)]
    
    X_data_sentences_1 = list(set(X_data_sentences_1) - set([best_sentences]))
    X_data_sentences = list(set(X_data_sentences) - set(best_sentences.split("\n")))
    X_data_sentences_1 = [best_sentences + "\n" + a for a in X_data_sentences]
    system_2grams = [create_ngrams(a.split(),2) for a in X_data_sentences_1]
    precision_recall = [rouge_metrics(a,reference_2grams) for a in system_2grams]
    f_score_list = [f_score(a[0],a[1]) for a in precision_recall]
    best_sentences = X_data_sentences_1[np.argmax(f_score_list)].split("\n")
    sentences = X_data_sentences_original
    first_sentence_vect = np.asarray(tf_matrix_uni[0].todense())
    for j in range(len(sentences)):
        position = 1/(j+1)
        if j == 0:
            doc_first = 1
        else:
            doc_first = 0
        length = len(sentences[j].split())
        quote = len(" ".join(re.findall("[\"\'][a-zA-Z0-9 ]*[\"\']",sentences[j])).strip().split())
        Centroid_uni = np.sum(tfidf_matrix_uni[j].todense())
        Centroid_bi = np.sum(tfidf_matrix_bi[j].todense())
        SigTerm_Uni = len(np.where(tf_matrix_uni[j].todense() > 0)[0])
        SigTerm_Bi = len(np.where(tf_matrix_bi[j].todense() > 0)[0])
        FreqWord_Uni = np.sum(tf_matrix_uni_1[j].todense())
        FreqWord_Bi = np.sum(tf_matrix_bi_1[j].todense())
        if sentences[j] in best_sentences:
            sentence_label = 1
        else:
            sentence_label = 0
        text = nlp(sentences[j].decode("utf8"))
        event_features_length = len('{0}'.format(text.ents).split())
        current_sentence_vect = np.transpose(np.asarray(tf_matrix_uni[j].todense()))
        FirstRel_Doc = compute_cosine_similarity(first_sentence_vect,current_sentence_vect)
        features_labels.append({"position":position,"doc_first":doc_first,"length":length,"quote":quote,"Centroid_Uni":Centroid_uni,"Centroid_Bi":Centroid_bi,"SigTerm_Uni":SigTerm_Uni,"SigTerm_Bi":SigTerm_Bi,"FreqWord_Uni":FreqWord_Uni,"FreqWord_Bi":FreqWord_Bi,"Event_Features":event_features_length,"FirstRel_Doc":FirstRel_Doc,"Label":sentence_label,"entity_score":X_data_entity_score[j]})           

features_label_df = pd.DataFrame(features_labels)
features_label_df.to_csv("Extension_2_Features.csv",index=False)

end = datetime.datetime.now()
duration = end - start
print("Duration - " + str(duration))
        